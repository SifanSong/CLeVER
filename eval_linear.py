# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits

def eval_linear(args):

    if args.dataset_type == "Imagenet":
        args.num_labels = 1000
    elif args.dataset_type == "IN100":
        args.num_labels = 100
    print("args.num_labels", args.num_labels)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if 'vmamba' in args.arch:
        from vmamba.config import get_config
        from vmamba.models import build_model
        ## vmamba args
        args.cfg='./vmamba/configs/'+args.arch.split('-')[0]+'/'+args.arch.split('-')[1]+'.yaml'
        args.opts=None
        args.batch_size=args.batch_size_per_gpu
        config = get_config(args)
        model = build_model(config)
        print(model)
        embed_dim = model.dims[-1]
        print("--------------embed_dim---------------", embed_dim) # 768
        DVR_dim = embed_dim - int(embed_dim * args.hp1)
        DIR_dim = embed_dim - DVR_dim
        print("--------------DIR_dim---------------", DIR_dim) # 614
        print("--------------DVR_dim---------------", DVR_dim) # 154
    else:
        if args.arch in vits.__dict__.keys():
            model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
            print(model)
            print("--------------model.embed_dim---------------", model.embed_dim) # 384
            embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
            if args.net == "dino":
                pass
            else:
                DVR_dim = model.embed_dim - int(model.embed_dim * args.hp1)
                DIR_dim = embed_dim - DVR_dim
                print("--------------DIR_dim---------------", DIR_dim) # 1536-77=1549
                print("--------------DVR_dim---------------", DVR_dim) # 77
            print("--------------embed_dim---------------", embed_dim) # 1536
        # if the network is a XCiT
        elif "xcit" in args.arch:
            model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
            embed_dim = model.embed_dim
        # otherwise, we check if the architecture is in torchvision models
        elif args.arch in torchvision_models.__dict__.keys():
            model = torchvision_models.__dict__[args.arch]()
            embed_dim = model.fc.weight.shape[1]
            model.fc = nn.Identity()
            DVR_dim = embed_dim - int(embed_dim * args.hp1)
            DIR_dim = embed_dim - DVR_dim
            print("--------------DIR_dim---------------", DIR_dim) # 2048-410 = 1638
            print("--------------DVR_dim---------------", DVR_dim) # 410
        else:
            print(f"Unknow architecture: {args.arch}")
            sys.exit(1)
    model.cuda()
    model.eval()

    # from fvcore.nn import FlopCountAnalysis
    # from fvcore.nn import flop_count_table
    # inp = (torch.randn((1,3,224,224)).cuda(non_blocking=True))
    # flops = FlopCountAnalysis(model, inp)
    # flops.total()
    # print(flop_count_table(flops))
    # sys.exit(0)

    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    if args.net == "dino":
        linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels, args=args)
    else:
        if args.else_part == "ALL":
            linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels, args=args)
        elif args.else_part == "main_part":
            linear_classifier = LinearClassifier(DIR_dim, num_labels=args.num_labels, args=args)
        elif args.else_part == "else_part":
            linear_classifier = LinearClassifier(DVR_dim, num_labels=args.num_labels, args=args)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])
    print(linear_classifier)

    # ============ preparing data ... ============

    # Data loading code
    if args.test_aug_type == "ori":
        # test_transforms = pth_transforms.Compose([
        #     pth_transforms.Resize(256),
        #     pth_transforms.CenterCrop(224),
        #     pth_transforms.ToTensor(),
        #     pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif args.test_aug_type == "basic":
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256),
            pth_transforms.CenterCrop(224),
            pth_transforms.RandomApply([
                pth_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=1.0),
            #transforms.RandomGrayscale(p=1.0),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1":
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256),
            pth_transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            pth_transforms.RandomHorizontalFlip(p=1.0),
            pth_transforms.RandomApply([
                pth_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=1.0),
            #transforms.RandomGrayscale(p=1.0),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1_2":
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256),
            pth_transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            #transforms.RandomHorizontalFlip(p=1.0),
            pth_transforms.RandomRotation(degrees=(-90, 90)),
            pth_transforms.RandomApply([
                pth_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=1.0),
            #transforms.RandomGrayscale(p=1.0),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1_3":
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256),
            pth_transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            #transforms.RandomHorizontalFlip(p=1.0),
            pth_transforms.RandomRotation(degrees=(-180, 180)),
            pth_transforms.RandomApply([
                pth_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=1.0),
            #transforms.RandomGrayscale(p=1.0),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1_4_1":
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256),
            pth_transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            #transforms.RandomHorizontalFlip(p=1.0),
            pth_transforms.RandomRotation(degrees=(-90, 90)),
            pth_transforms.RandomApply([
                pth_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=1.0),
            #transforms.RandomGrayscale(p=1.0),
            pth_transforms.RandomApply([
                pth_transforms.ElasticTransform(alpha=50.0)
            ], p=1.0),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1_4_2":
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256),
            pth_transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            #transforms.RandomHorizontalFlip(p=1.0),
            pth_transforms.RandomRotation(degrees=(-90, 90)),
            pth_transforms.RandomApply([
                pth_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=1.0),
            #transforms.RandomGrayscale(p=1.0),
            pth_transforms.RandomApply([
                pth_transforms.ElasticTransform(alpha=100.0)
            ], p=1.0),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.test_aug_type == "aug1_4_3":
        val_transform = pth_transforms.Compose([
            pth_transforms.Resize(256),
            pth_transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            #transforms.RandomHorizontalFlip(p=1.0),
            pth_transforms.RandomRotation(degrees=(-90, 90)),
            pth_transforms.RandomApply([
                pth_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=1.0),
            #transforms.RandomGrayscale(p=1.0),
            pth_transforms.RandomApply([
                pth_transforms.ElasticTransform(alpha=150.0)
            ], p=1.0),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    print("args.test_aug_type", args.test_aug_type)

    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        utils.load_pretrained_linear_weights(linear_classifier, args.arch, args.patch_size, args.final_eval_weights)
        test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    if args.optimizer == "lars":
        print("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)
    else:
        print("=> use SGD optimizer.")

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    # utils.restart_from_checkpoint(
    #     os.path.join(args.output_dir, "checkpoint.pth.tar"),
    #     run_variables=to_restore,
    #     state_dict=linear_classifier,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    # )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model number of params (M): %.2f' % (n_parameters / 1.e6))
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print("model len(parameters)", len(parameters))
    for _, p in model.named_parameters():
        p.requires_grad = False
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model number of params (M): %.2f' % (n_parameters / 1.e6))
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print("model len(parameters)", len(parameters))
    n_parameters = sum(p.numel() for p in linear_classifier.parameters() if p.requires_grad)
    print('linear_classifier number of params (M): %.2f' % (n_parameters / 1.e6))
    parameters = list(filter(lambda p: p.requires_grad, linear_classifier.parameters()))
    print("linear_classifier len(parameters)", len(parameters))

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens, args)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            if args.else_part == "ALL":
                torch.save(save_dict, args.output_dir+"/"+"checkpoint_linear_"+str(args.lr)+"_"+str(args.epochs)+"_"+args.trial_name+".pth.tar")
            elif args.else_part == "main_part":
                torch.save(save_dict, args.output_dir+"/"+"checkpoint_linear_hp0.8_"+str(args.lr)+"_"+str(args.epochs)+"_"+args.trial_name+".pth.tar")
            elif args.else_part == "else_part":
                torch.save(save_dict, args.output_dir+"/"+"checkpoint_linear_hp0.8_else_"+str(args.lr)+"_"+str(args.epochs)+"_"+args.trial_name+".pth.tar")
            
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool, args):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                # print("------------len(intermediate_output)------------", len(intermediate_output)) # 4
                # print("------------intermediate_output[0].shape)------------", intermediate_output[0].shape) # torch.Size([128, 197, 384])
                # print("------------intermediate_output[0][:, 0].shape)------------", intermediate_output[0][:, 0].shape) # torch.Size([128, 384])
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1) ## only cls_token ## concate last 4 cls tokens
                # print("-----------------output3.shape-----------------", output.shape) # torch.Size([128, 1536])
                if avgpool: # avgpool=False default
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        # print("-----------------output4.shape-----------------", output.shape) # torch.Size([128, 1000])

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000, args=None):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        self.dim = dim
        self.args = args

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        if "resnet" in args.arch or "vmamba" in args.arch:
            if args.else_part == "main_part":
                x = x[:, :self.dim] ## DIR_dim
            elif args.else_part == "else_part":
                x = x[:, -self.dim:] ## DVR_dim
        else:
            if args.avgpool_patchtokens == False:
                if args.else_part == "main_part":
                    x = x[:, :self.dim] ## DIR_dim
                elif args.else_part == "else_part":
                    x = x[:, -self.dim:] ## DVR_dim
            else:
                x1, x2 = x[:, :int((x.shape[-1])/2)], x[:, int((x.shape[-1])/2):] ## DIR_dim
                # print("avg, x.shape, int((x.shape[-1])/2)", x.shape, int((x.shape[-1])/2))
                if args.else_part == "main_part":
                    if args.arch == "vit_small":
                        clip_dim = int(384 * args.hp1)
                    elif args.arch == "vit_base":
                        clip_dim = int(768 * args.hp1)
                    x = torch.cat((x1[:, :clip_dim], x2), dim=-1)
                elif args.else_part == "else_part":
                    x = x1[:, -self.dim:]
            # print("linear x.shape", x.shape)
            # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    ## CLeVER
    parser.add_argument('--net', default='dino', type=str) ## dino, CLeVER
    parser.add_argument('--trial_name', default='', type=str) ## for save model
    parser.add_argument('--hp1', default=1.0, type=float, help='hp1')
    parser.add_argument('--else_part', default='main_part', type=str, choices=['ALL', 'main_part', 'else_part'])
    parser.add_argument('--li_aug', default='_li_aug1', type=str, choices=['_li_aug1', '_li_aug1_2', '_li_aug1_4_2'])
    ## dataset_type
    parser.add_argument('--dataset_type', default='Imagenet', type=str, choices=['Imagenet', 'IN100'])
    parser.add_argument('--ds_mode', default='linear', type=str) #, choices=['linear', 'semi_1', 'semi_10', 'ds_ft'])
    # parser.add_argument('--backbone_lr', default=0.1, type=float, help='backbone lr for semi/finetune')
    parser.add_argument('--test_aug_type', default='ori', type=str, help='test_aug_type')
    parser.add_argument('--final_eval_weights', default=None, type=str, help='final_eval_weights for only eval')

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'lars'])

    args = parser.parse_args()
    eval_linear(args)
