import os
import argparse
import torch
import distributed_utils ######
import numpy as np
import torch.backends.cudnn as cudnn
import time
from distributed_utils import is_main_process, MetricLogger ######
from typing import Iterable, Optional
import torch.optim as optim
import datetime
from optim_factory import create_optimizer ######

from dataset import Action_Dataset, data_preprocessing ######
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logging
from torch import nn
from model.network import * # 导入网络模型DG-STA


# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('training and evaluation script for image matting', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--log_dir', default='./distributed_log/base_line2',
                        help='path where to tensorboard log')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Optimization parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")  

    ####### warm up
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--lr', type=float, default=5e-5, help='Learning Rate. Default=0.00001')
    parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    # parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--logname', type=str, default='train_log', help="name of the logging file")
    parser.add_argument('--save_ckpt_num', default=30, type=int)
    parser.add_argument('--finetune', type=str, default='./saved_models/u2net/u2net.pth', help="name of the logging file")
    parser.add_argument('--model_name', type=str, default='eff3', help="model_name")
    # parser.add_argument('--class_names', type=list, default=['No','Yes'], help="class_names")
    parser.add_argument('--class_names',type=str, default='No, Yes', help="class_names")
    parser.add_argument('--data_path', type=str, default='./data/Flower', help="data_path")
    return parser

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def load_model(args): 

    class_num = 7
    time_len = 41
    # dp_rate = 0.1
    # dp_rate = 0.4
    # dp_rate = 0.7

    dp_rate = 0.9
    # coordinates_nums = 4 
    coordinates_nums = 5 
    model = DG_STA(class_num, dp_rate, time_len, coordinates_nums)


    start_epoch = 0
    return model, start_epoch

def load_dataset(args):
    train_data, test_data, weights_train = data_preprocessing()
    args.weights_train = weights_train
    print('train_len: ', len(train_data), " ", 'valid_len: ', len(test_data))


    dataset_train = Action_Dataset(train_data)

    dataset_val = Action_Dataset(test_data)

    return dataset_train, dataset_val

def tensor_print(res):
    for t in res:
        print(torch.max(t), torch.min(t))


def train(args, model, data_loader: Iterable, optimizer, device: torch.device, epoch,
    log_writer=None, start_steps=None, num_training_steps_per_epoch=None, update_freq=None, lr_schedule_values=None, sampler_train=None):

    metric_logger = MetricLogger(delimiter="  ")
    model.train(True)
    print_freq = int(num_training_steps_per_epoch / 2)

    header = 'Epoch Train: [{}]'.format(epoch)
    optimizer.zero_grad()
    sampler_train.set_epoch(epoch+1)
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration

        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] 
                    # * param_group["lr_scale"]


        batch_new = []
        for item in batch:
            item = Variable(item).cuda(device, non_blocking=True)
            batch_new.append(item)
        [inputs_v, labels_v] = batch_new
        # image_tensor, mask_tensor, edge_tensor, body_tensor
        # forward + backward + optimize
        output = model(inputs_v)
        weight = torch.from_numpy(np.array(args.weights_train)).float()
        criterion = nn.CrossEntropyLoss(weight=weight.to(device))
        loss = criterion(output, labels_v)

            
        loss_value = loss.item()

        loss /= update_freq

        torch.autograd.set_detect_anomaly(True)
        loss.backward()    

        class_acc = (output.max(-1)[-1] == labels_v).float().mean()
        metric_logger.update(class_acc=class_acc)

        # nn.utils.clip_grad_norm_(model.parameters(), 2)

        if (data_iter_step + 1) % update_freq == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        # del temporary outputs and loss
        del inputs_v, labels_v, loss


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if log_writer is not None:
        args.logging.info("Train GFM-Epoch[{}/{}] Lr:{:.8f} Loss:{:.5f}".format(
                epoch, args.epochs, optimizer.param_groups[0]['lr']
                , stats['loss']))

        for key in stats.keys():
            log_writer.update_epoch(**{key:stats[key]}, head="epoch_train")

        # log_writer.set_step_epoch()
    return stats

def eval(args, model, data_loader: Iterable, optimizer, device: torch.device, epoch, log_writer=None, start_steps=None, num_training_steps_per_epoch=None, update_freq=None):

    metric_logger = MetricLogger(delimiter="  ")
    model.eval()
    print_freq = int(num_training_steps_per_epoch / 1)
    header = 'Epoch Val: [{}]'.format(epoch)
    with torch.no_grad():
        for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            step = data_iter_step // update_freq
            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step  # global training iteration

            batch_new = []
            for item in batch:
                item = Variable(item).cuda(device, non_blocking=True)
                batch_new.append(item)
            [inputs_v, labels_v] = batch_new
            # image_tensor, mask_tensor, edge_tensor, body_tensor
            # forward + backward + optimize
            output = model(inputs_v)
            weight = torch.from_numpy(np.array(args.weights_train)).float()
            criterion = nn.CrossEntropyLoss(weight=weight.to(device))
            loss = criterion(output, labels_v)
            loss_value = loss.item()

            class_acc = (output.max(-1)[-1] == labels_v).float().mean()
            metric_logger.update(class_acc=class_acc)

            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)

            del inputs_v, labels_v, loss

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if log_writer is not None:
        args.logging.info("Val GFM-Epoch[{}/{}] Lr:{:.8f} Loss:{:.5f}".format(
                epoch, args.epochs, optimizer.param_groups[0]['lr']
                , stats['loss']))

        for key in stats.keys():
            log_writer.update_epoch(**{key:stats[key]}, head="epoch_val")
  
    return stats

def main(args):
    distributed_utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + distributed_utils.get_rank()
    torch.manual_seed(seed)
    ####np.random.seed(seed)#####数据增强需要---需要注释掉

    cuda_deterministic = False
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    num_tasks = distributed_utils.get_world_size()
    global_rank = distributed_utils.get_rank()


    ###### 保存event日志
    if global_rank == 0 and args.log_dir is not None:

        os.makedirs(args.log_dir, exist_ok=True)
        now = datetime.datetime.now()
        logging_filename = args.log_dir+"/" + args.logname+'_'+now.strftime("%Y-%m-%d-%H:%M")+'.log'
        logging.basicConfig(filename=logging_filename, level=logging.INFO)
        args.logging = logging
        log_writer = distributed_utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    #####################加载自己的数据
    dataset_train, dataset_val = load_dataset(args)
    #####分布式训练与验证
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(2 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    ####创建模型
    model, start_epoch = load_model(args)

    # 引入SyncBN这句代码---会将普通BN替换成SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    total_batch_size = args.batch_size * args.update_freq * distributed_utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    # torch.distributed.init_process_group('gloo', init_method='env://', world_size=1, rank=0) ####新增的
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    min_val_loss = None

    #### base_line_data_enhancement
    ## optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer = None, filter_bias_and_bn=False,
        get_layer_scale = None)
    print("Use Cosine LR scheduler")
    lr_schedule_values = distributed_utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    for epoch in range(start_epoch, args.epochs+start_epoch):
        data_loader_train.sampler.set_epoch(epoch)
        data_loader_val.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            log_writer.set_step_epoch(epoch)
        train(args, model, data_loader_train, optimizer, device, epoch,
        log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch, num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
        lr_schedule_values = lr_schedule_values, sampler_train=sampler_train
        )
        #### 保存模型
        if args.log_dir and args.save_ckpt:
            if (epoch) % args.save_ckpt_freq == 0 or epoch == args.epochs:
                distributed_utils.save_model(args, epoch, model_without_ddp, optimizer)
        eval_stats = eval(args, model, data_loader_val, optimizer, device, epoch, log_writer=log_writer,start_steps=epoch * num_training_steps_per_epoch,num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq)
        if min_val_loss is None or min_val_loss >= eval_stats["loss"]:
            min_val_loss = eval_stats["loss"]
            print("Save Best model %d epochs" % epoch)
            if args.log_dir and args.save_ckpt:
                distributed_utils.save_model(args, "best", model_without_ddp, optimizer)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser('Distributed training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.class_names = [str(item) for item in args.class_names.split(',')]
    main(args)