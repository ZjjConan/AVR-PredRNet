import argparse
import os
import random
import time
import warnings
import copy
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from utils import AverageMeter, ProgressMeter, ToTensor, accuracy, normalize_image, parse_gpus
from report_acc_regime import init_acc_regime, update_acc_regime
from loss import BinaryCrossEntropy, ContrastiveLoss
from checkpoint import save_checkpoint, load_checkpoint
from thop import profile
from networks import create_net


parser = argparse.ArgumentParser(description='PredRNet: Neural Prediction Errors for Abstract Visual Reasoning')

# dataset settings
parser.add_argument('--dataset-dir', default='datasets/',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='RAVEN-FAIR',
                    help='dataset name')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--image-size', default=80, type=int,
                    help='image size')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')

# network settings
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('--enable-rc', action='store_true',
                    help="whether to use row and col rules for prediction (e.g. PGM)")
parser.add_argument('--num-extra-stages', default=0, type=int,
                    help='number of extra normal residue blocks or predictive coding blocks')
parser.add_argument('--classifier-hidreduce', default=4, type=int,
                    help='classifier hidden dimension scale')
parser.add_argument('--block-drop', default=0.0, type=float,
                    help="dropout within each block")
parser.add_argument('--classifier-drop', default=0.0, type=float,
                    help="dropout within classifier block")
parser.add_argument('--num-filters', default=32, type=int,
                    help="basic filters for backbone network")
parser.add_argument('--in-channels', default=1, type=int,
                    help="input image channels")

# training settings
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# others settings
parser.add_argument("--ckpt", default="./ckpts/", 
                    help="folder to output checkpoints")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default="0",
                    help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument("--fp16", action='store_true',
                    help="whether to use fp16 for training")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--show-detail', action='store_true',
                    help="whether to show detail accuracy on all sub-types")
parser.add_argument('--subset', default='None', type=str,
                    help='subset selection for dataset')


# Contrastive Loss
parser.add_argument('--c-margin', default=1.0, type=float,
                    help='margin for contrastive loss')


# seed the sampling process for reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loader(args, data_split = 'train', transform = None, subset = None):

    if args.dataset_name in ('RAVEN', 'RAVEN-FAIR', 'I-RAVEN'):
        from data import RAVEN as create_dataset
    elif 'PGM' in args.dataset_name:
        from data import PGM as create_dataset
    elif 'VAD' in args.dataset_name:
        from data import VAD as create_dataset
    elif 'CLEVR-Matrix' in args.dataset_name:
        from data import CLEVR_MATRIX as create_dataset
    elif 'new_split' in args.dataset_name:
        from data import RAVEN as create_dataset
    else:
        raise ValueError(
            "not supported dataset_name = {}".format(args.dataset_name)
        )

    # assert False
    dataset = create_dataset(
        args.dataset_dir, data_split = data_split, image_size = args.image_size, 
        transform = transform, subset = subset
    )

    if args.seed is not None:
        g = torch.Generator()
        g.manual_seed(args.seed)
    else:
        g = None

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size = args.batch_size, shuffle = (data_split == "train"),
        num_workers = args.workers, pin_memory = True, sampler = None,
        generator = g, worker_init_fn = seed_worker,
    )

    return data_loader


best_acc1 = 0

def main():
    args = parser.parse_args()

    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)

    args.ckpt += args.dataset_name
    args.ckpt += "-" + args.arch
    if "pred" in args.arch:
        args.ckpt += "-prb" + str(args.num_extra_stages)
    else:
        args.ckpt += "-ext" + str(args.num_extra_stages)

    if args.block_drop > 0.0 or args.classifier_drop > 0.0:
        args.ckpt += "-b" + str(args.block_drop) + "c" + str(args.classifier_drop)
        
    args.ckpt += "-imsz" + str(args.image_size)
    args.ckpt += "-wd" + str(args.weight_decay)
    args.ckpt += "-ep" + str(args.epochs)
    args.gpu = parse_gpus(args.gpu)
    if args.gpu is not None:
        args.device = torch.device("cuda:{}".format(args.gpu[0]))
    else:
        args.device = torch.device("cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        args.ckpt += '-seed' + str(args.seed)
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)


    main_worker(args)


def main_worker(args):
    global best_acc1

    # create model
    model = create_net(args)

    log_path = os.path.join(args.ckpt, "log.txt")

    if os.path.exists(log_path):
        log_file = open(log_path, mode="a")
    else:
        log_file = open(log_path, mode="w")
    
    for key, value in vars(args).items():
        log_file.write('{0}: {1}\n'.format(key, value))

    args.log_file = log_file

    model_flops = copy.deepcopy(model)
    if "Analogy" in args.dataset_name:
        x = torch.randn(1, 9, args.image_size, args.image_size)
    elif "CLEVR-Matrix" in args.dataset_name:
        x = torch.randn(1, 16, 3, args.image_size, args.image_size)
    else:
        x = torch.randn(1, 16, args.image_size, args.image_size)
    flops, params = profile(model_flops, inputs=(x,))

    print("model [%s] - params: %.6fM" % (args.arch, params / 1e6))
    print("model [%s] - FLOPs: %.6fG" % (args.arch, flops / 1e9))
        
    args.log_file.write("--------------------------------------------------\n")
    args.log_file.write("Network - " + args.arch + "\n")
    args.log_file.write("Params - %.6fM" % (params / 1e6) + "\n")
    args.log_file.write("FLOPs - %.6fG" % (flops / 1e9) + "\n")
    
    if args.evaluate == False:
        print(model)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.device)
        model = model.to(args.gpu[0])
        model = torch.nn.DataParallel(model, args.gpu) 

    # define loss function (criterion) and optimizer
    bce_loss = BinaryCrossEntropy().cuda(args.gpu)
    ctr_loss = ContrastiveLoss(margin=args.c_margin).cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    if args.resume:
        model, optimizer, best_acc1, start_epoch = load_checkpoint(args, model, optimizer)
        args.start_epoch = start_epoch

    # Data loading code
    tr_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        ToTensor()
    ])
    ts_transform = transforms.Compose([
        ToTensor()
    ])
    
    tr_loader = get_data_loader(args, data_split='train', transform=tr_transform, subset=args.subset)
    vl_loader = get_data_loader(args, data_split='val',   transform=ts_transform, subset=args.subset)
    ts_loader = get_data_loader(args, data_split='test',  transform=ts_transform, subset=args.subset)

    args.log_file.write("Number of training samples: %d\n" % len(tr_loader.dataset))
    args.log_file.write("Number of validation samples: %d\n" % len(vl_loader.dataset))
    args.log_file.write("Number of testing samples: %d\n" % len(ts_loader.dataset))

    args.log_file.write("--------------------------------------------------\n")
    args.log_file.close()

    if args.evaluate:
        acc = validate(ts_loader, model, bce_loss, ctr_loss, args, valid_set="Test")
        return

    if args.fp16:
        args.scaler = torch.cuda.amp.GradScaler()

    cont_epoch = 0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        
        args.log_file = open(log_path, mode="a")

        # train for one epoch
        train(tr_loader, model, bce_loss, ctr_loss, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(ts_loader, model, bce_loss, ctr_loss, args, valid_set="Test")

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc": best_acc1,
            "optimizer" : optimizer.state_dict(),
            }, is_best, epoch, save_path=args.ckpt)

        if is_best:
            cont_epoch = 0
            best_epoch = epoch
        else:
            cont_epoch += 1

        epoch_msg = ("----------- Best Acc at [{}]: Test {:.3f} Continuous Epoch {} -----------".format(
            best_epoch, best_acc1, cont_epoch)
        )
        print(epoch_msg)

        args.log_file.write(epoch_msg + "\n")
        args.log_file.close()


def train(data_loader, model, bce_loss, ctr_loss, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':4.3f')
    data_time = AverageMeter('Data', ':4.3f')
    losses = AverageMeter('loss', ':4.3f')

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses],
        prefix = "Epoch: [{}]".format(epoch))

    param_groups = optimizer.param_groups[0]
    curr_lr = param_groups["lr"]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _, meta_target, structure_encoded, data_file) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        if args.gpu is not None:
            images = images.to(args.device, non_blocking=True)

        images = normalize_image(images)

        # compute output
        if args.fp16:
            with torch.cuda.amp.autocast():
                output, target = model(images)
            ctrs = [ctr_loss(oe, target) for oe in output]
            loss = sum(ctrs) / len(ctrs)
            args.scaler.scale(loss).backward()
            args.scaler.step(optimizer)
            args.scaler.update()
        else:
            output, target = model(images)
            ctrs = [ctr_loss(oe, target) for oe in output]
            loss = sum(ctrs) / len(ctrs)
            loss.backward()
            optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(data_loader) - 1:
            epoch_msg = progress.get_message(i+1)
            epoch_msg += ("\tLr  {:.4f}".format(curr_lr))
            print(epoch_msg)

            args.log_file.write(epoch_msg + "\n")


def validate(data_loader, model, bce_loss, ctr_loss, args, valid_set='Valid'):
    
    if 'RAVEN' in args.dataset_name:
        acc_regime = init_acc_regime(args.dataset_name)
    else:
        acc_regime = None

    batch_time = AverageMeter('Time', ':4.3f')
    losses = AverageMeter('loss', ':4.3f')
    top1 = AverageMeter('Acc', ':4.2f')
    
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, losses, top1],
        prefix = valid_set + ': ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, meta_target, structure_encoded, data_file) in enumerate(data_loader):
           
            if args.gpu is not None:
                images = images.to(args.device, non_blocking=True)
            if torch.cuda.is_available():
                target = target.to(args.device, non_blocking=True)

            images = normalize_image(images)

            # compute outputs
            output = model(images)[-1]
            ctrs = ctr_loss(output, target)
            losses.update(ctrs.item(), images.size(0))

            # measure accuracy and record loss
            acc1 = accuracy(-output, target)
            
            top1.update(acc1[0][0], images.size(0))

            if acc_regime is not None:
                update_acc_regime(args.dataset_name, acc_regime, -output, target, structure_encoded, data_file)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(data_loader) - 1:
                epoch_msg = progress.get_message(i+1)
                print(epoch_msg)

        if acc_regime is not None:
            for key in acc_regime.keys():
                if acc_regime[key] is not None:
                    if acc_regime[key][1] > 0:
                        acc_regime[key] = float(acc_regime[key][0]) / acc_regime[key][1] * 100
                    else:
                        acc_regime[key] = None

            mean_acc = 0
            for key, val in acc_regime.items():
                mean_acc += val
            mean_acc /= len(acc_regime.keys())
        else:
            mean_acc = top1.avg

        epoch_msg = '----------- {valid_set} Acc {mean_acc:.3f} -----------'.format(
            valid_set=valid_set, mean_acc=mean_acc
        )

        print(epoch_msg)
        
        if args.evaluate == False:
            args.log_file.write(epoch_msg + "\n")


    if args.show_detail:
        for key, val in acc_regime.items():
            print("configuration [{}] Acc {:.3f}".format(key, val))


    return mean_acc


if __name__ == '__main__':
    main()