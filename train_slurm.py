# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import datetime
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import subprocess

from utils.metric_util import per_class_iu, fast_hist_crop
from utils import common_utils
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        print("is SLURM")
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29523"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)

        print("WORLD_SIZE:"+str(world_size))
        print("RANK:"+str(rank % num_gpus))
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )

# 通过 srun 产生的程序在环境变量中会有 SLURM_JOB_ID， 以此判断是否为slurm的调度方式
# rank 通过 SLURM_PROCID 可以拿到
# world size 实际上就是进程数， 通过 SLURM_NTASKS 可以拿到
# IP地址通过 ``subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")`` 巧妙得到，栗子来源于 [MMCV](https://github.com/open-mmlab/mmcv)
# 否则，就使用launch进行调度，直接通过 os.environ["RANK"] 和 os.environ["WORLD_SIZE"] 即可拿到 rank 和 world size

def main(args):
    #pytorch_device = torch.device('cuda:0')
    setup_distributed(backend="nccl")
    # torch.distributed.init_process_group(
    #     backend = 'nccl'
    #     #init_method='env://'
    # )

    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)


    pytorch_device = torch.device(f'cuda:{local_rank}')
    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)
    #my_model = torch.nn.DataParallel(my_model)
    my_model = torch.nn.parallel.DistributedDataParallel(my_model, device_ids=[local_rank],output_device=local_rank)

    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    output_dir = os.path.join('/mnt/lustre/xiaozeqi.vendor/output/'+args.outdir)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    tmp_dir = os.path.join(output_dir, 'tmp')
    summary_dir = os.path.join(output_dir, 'summary')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir, exist_ok=True)

    log_file = os.path.join(output_dir, ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    logger.info('lr: %f' % train_hypers["learning_rate"])
    logger.info('batch_size: %d' % train_dataloader_config['batch_size'])
    
    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']

    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        if local_rank == 0:
            pbar = tqdm(total=len(train_dataset_loader), dynamic_ncols=True)
        #pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        # lr_scheduler.step(epoch)

        if local_rank == 0:
            if epoch >= 0:
                my_model.eval()
                hist_list = []
                val_loss_list = []
                with torch.no_grad():
                    for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                            val_dataset_loader):

                        val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                            val_pt_fea]
                        val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                        val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

                        predict_labels = my_model(val_pt_fea_ten, val_grid_ten,val_label_tensor.shape[0])#, val_batch_size)
                        # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                        loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                                ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
                        predict_labels = torch.argmax(predict_labels, dim=1)
                        predict_labels = predict_labels.cpu().detach().numpy()
                        for count, i_val_grid in enumerate(val_grid):
                            hist_list.append(fast_hist_crop(predict_labels[
                                                                count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                                val_grid[count][:, 2]], val_pt_labs[count],
                                                            unique_label))
                        val_loss_list.append(loss.detach().cpu().numpy())
                my_model.train()
                iou = per_class_iu(sum(hist_list))
                # print('Validation per class iou: ')
                # for class_name, class_iou in zip(unique_label_str, iou):
                #     print('%s : %.2f%%' % (class_name, class_iou * 100))
                logger.info('Validation per class iou: ')
                for class_name, class_iou in zip(unique_label_str, iou):
                    logger.info('%s : %.2f%%' % (class_name, class_iou * 100))
                val_miou = np.nanmean(iou) * 100
                del val_vox_label, val_grid, val_pt_fea, val_grid_ten

                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    torch.save(my_model.state_dict(), os.path.join(ckpt_dir,
                            'checkpoint_epoch_{}_{}.pth'.format(epoch, str(best_val_miou))))

                # print('Current val miou is %.3f while the best val miou is %.3f' %
                #         (val_miou, best_val_miou))
                # print('Current val loss is %.3f' %
                #         (np.mean(val_loss_list)))
                logger.info('Current val miou is %.3f while the best val miou is %.3f' %
                        (val_miou, best_val_miou))
                logger.info('Current val loss is %.3f' %
                        (np.mean(val_loss_list)))

        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
            #vox_cor(in cat), vox_label, grid(in polar), label(raw), decorate feature
            #在数据集中已经part和decorate好了

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]#是一个长度为N的list 
            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]#
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

            # forward + backward + optimize
            # 只需要decorate point和vox grid
            outputs = my_model(train_pt_fea_ten, train_vox_ten, point_label_tensor.shape[0])#,train_batch_size)
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=0) + loss_func(outputs, point_label_tensor)

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            optimizer.zero_grad()
            global_iter += 1
            if local_rank == 0:
                if global_iter % 100 == 0:
                    if len(loss_list) > 0:
                        print('global iter: %d, epoch %d iter %5d, loss: %.3f\n' %
                                (global_iter, epoch, i_iter, np.mean(loss_list)))
                        logger.info('global iter: %d, epoch %d iter %5d, loss: %.3f\n' %
                                (global_iter, epoch, i_iter, np.mean(loss_list)))
                    else:
                        print('loss error')
                pbar.set_postfix({'loss': loss, 'lr': train_hypers["learning_rate"],'epoch':epoch})
                pbar.update(1)

                if global_iter % check_iter == 0:
                    if len(loss_list) > 0:
                        print('epoch %d iter %5d, loss: %.3f\n' %
                            (epoch, i_iter, np.mean(loss_list)))
                    else:
                        print('loss error')
        if local_rank == 0:
            pbar.close()
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    parser.add_argument('--outdir', default='Cylinder', help='node rank for distributed training')

    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
