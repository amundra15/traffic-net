
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR 
import torch.nn as nn

from torchvision import transforms
from utils.accuracy import accuracy
from utils.averagemeter import AverageMeter

from models.lenet import LeNet5
from dataloader.trafficnet import trafficnet_dataset

import argparse
import os
import shutil
from tqdm import tqdm

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def train(model, train_loader, optimizer, criterion, writer, epoch):
    train_acc   = AverageMeter()
    train_loss  = AverageMeter()
    model.train()
    num_iter    = len(train_loader) * epoch 

    for i, (inputs, targets) in tqdm(enumerate(train_loader)):
        inputs = inputs.cuda()
        targets = targets.cuda()

        _,outputs = model(inputs)
        loss    = criterion(outputs, targets)
        
        acc1    = accuracy(outputs.data, targets.data, topk=(1,))
        
        train_loss.update(loss.data.item(), inputs.size(0))
        train_acc.update(acc1[0].item(), inputs.size(0))
        writer.add_scalar("train_loss_iter", train_loss.avg, num_iter + i)
        writer.add_scalar("train_acc_iter", train_acc.avg, num_iter + i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return (train_loss.avg, train_acc.avg)


def test(model, test_loader, criterion, writer, epoch):
    test_acc    = AverageMeter()
    test_loss   = AverageMeter()
    model.eval()
    num_iter    = len(train_loader) * epoch 
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(test_loader)):
            inputs = inputs.cuda()
            targets = targets.cuda()

            _,outputs = model(inputs)
            loss    = criterion(outputs, targets)
            
            acc1    = accuracy(outputs.data, targets.data, topk=(1,))
            
            test_loss.update(loss.data.item(), inputs.size(0))
            test_acc.update(acc1[0].item(), inputs.size(0))
            writer.add_scalar("test_loss_iter", test_loss.avg, num_iter + i)
            writer.add_scalar("test_acc_iter", test_acc.avg, num_iter + i)
    
    return (test_loss.avg, test_acc.avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training network to detect traffic congestion")

    # data
    parser.add_argument("--datapath", type=str, default="/home/kaustubh/kaustubh_imp/courses/DS21/Project/dataset/trafficnet_dataset_v1/", help="path to trafficnet dataset")
    parser.add_argument("--checkpoint", type=str, default="/home/kaustubh/kaustubh_imp/courses/DS21/Project/experiments/lenet/", help="path where to save the logs, resuts, models etc")
    
    # training
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate for the experiment")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of the optimizer")
    parser.add_argument("--schedule", type=int, nargs="+", default=None, help="Milestones of step LR if given else step LR not used")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR factor of LR scheduler")
    parser.add_argument("--wd", type=float, default=0.0001, help="weight decay")
    parser.add_argument("--train_batch", type=int, default=128, help="train batch size")
    parser.add_argument("--test_batch", type=int, default=128, help="test batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--model", type=str, default="lenet", help="Backbone network, one of [lenet, resnet50]")
    
    # resume
    parser.add_argument("--resume", action="store_true", default=False)

    # Misc
    parser.add_argument("--manual_seed", default=None, type=int, help="Initial seed of the experiment")

    args = parser.parse_args()

    # Defining the dataloader
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = trafficnet_dataset(datapath=args.datapath, train=True, train_trainsform=train_transforms)
    test_set  = trafficnet_dataset(datapath=args.datapath, train=False, train_trainsform=test_transforms)

    train_loader = data.DataLoader(train_set, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers)
    test_loader  = data.DataLoader(test_set, batch_size=args.train_batch, shuffle=False, num_workers=args.num_workers)

    model = LeNet5(2)
    model.cuda()

    start_epoch = 0
    best_acc    = 0
    start_iter  = 0

    optimizer = SGD(model.parameters(), lr = args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd)
    scheduler = None
    criterion = nn.CrossEntropyLoss().cuda()

    if args.schedule is not None:
        scheduler = MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=args.gamma)

    if args.resume:
        checkpoint  = torch.load(os.path.join(args.checkpoint + "checkpoint.pth.tar"))
        model       = checkpoint['model']
        best_acc    = checkpoint['best_acc']
        optimizer   = checkpoint['optimizer']
        scheduler   = checkpoint['scheduler']
        start_epoch = checkpoint['epoch']
        start_iter  = checkpoint['iter']

    writer = SummaryWriter(log_dir=args.checkpoint)

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, writer, epoch)
        test_loss, test_acc   = test(model, test_loader, criterion, writer, epoch)
        
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer,
                'scheduler': scheduler,
            }, is_best, checkpoint=args.checkpoint)
        writer.add_scalar('Epoch_Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch_Train_acc', train_acc, epoch)
        writer.add_scalar('Epoch_test_Loss', test_loss, epoch)
        writer.add_scalar('Epoch_test_acc', test_acc, epoch)

        writer.flush()
        print("Done epoch: ", epoch, " train_acc:", train_acc, " train_loss:", train_loss, " test_acc:", test_acc, " test_loss:", test_loss)
        if scheduler is not None:
            scheduler.step()
        
writer.close()
        

