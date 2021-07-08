import torch, torch.nn as nn
import argparse, sys
import os, os.path as osp
import random
from torch.utils.data import DataLoader

from dataset import datasets
from architecture import models
from utils.RAdam import RAdam
from utils.Logger import Logger
from utils.val_loss import val_loss

N_ITER = 50

def parse_args():
    parser = argparse.ArgumentParser(description='Garbage Classification Configuration')

    parser.add_argument('-d', '--dataset', type=str, default='GarbageClass',
                        choices=datasets.keys(),  help=f"options: {datasets.keys()}")
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to store logs and trained models')
    parser.add_argument('-model', type=str, default='resnet-18', choices=models.keys(), help=f'Options: {models.keys()}')
    parser.add_argument('--lr', type=float, default=1e-4, help=f'Learning Rate')
    parser.add_argument('--max-epoch', type=int, default='100')
    parser.add_argument('--patience', default=-1, type=int,
                        help="number of epochs without model improvement (-1 to disable it)")
    parser.add_argument('--seed', type=int, default=64678, help="manual seed")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help="root directory where part/fold of previous train are saved")
    parser.add_argument('--mode', type=str, default='train', choices=['train','test'], help='Options: [train, test]') #test after training though

    return vars(parser.parse_args())

def train(args):
    try:
        os.makedirs(args['log_dir'])
    except FileExistsError:
        pass

    train_set = datasets[args['dataset']](mode='train') #pass other args too
    val_set = datasets[args['dataset']](mode='val')

    train_loader = DataLoader(train_set
                                , batch_size=args['batch_size']
                                , shuffle=True
                                #, sampler=valSampler
                                , pin_memory=True
                                , drop_last=True)

    val_loader = DataLoader(val_set
                            , batch_size=1
                            , shuffle=True
                            #, sampler=valSampler
                            , pin_memory=True
                            , drop_last=False)

    net = models[args['model']](train_set.get_nclass()) #pass args
    net.to(args['device'])

    optim = RAdam(net.parameters(), lr=args['lr'])

    f_loss = nn.CrossEntropyLoss()

    if args['resume']:
        sys.stdout = Logger(osp.join(args['log_dir'], 'log_train.txt'), mode='a') #todo print é salvo em um arquivo tambem
        state_dict = torch.load(osp.join(args['log_dir'], 'last_state.pth'))
        initial_epoch = state_dict['epoch']
        loss_data = state_dict['loss_data']
        net.load_state_dict(state_dict['net'])
        optim.load_state_dict(state_dict['optimizer'])
        best_val = state_dict['best_val']
        best_model = state_dict['best_model']
        print(f'Resuming training. Epoch {initial_epoch}')
    else:
        sys.stdout = Logger(osp.join(args['log_dir'], 'log_train.txt'), mode='w')
        initial_epoch = -1
        iteration = 0
        best_val = sys.maxsize
        best_model = None
        loss_data = {'loss': [], 'iteration': []}

    acu_loss_gran = 0
    acu_loss_epoch = 0

    for epoch in range(initial_epoch+1, args['max_epoch']): #train_loop
        for ite, (img, label) in enumerate(train_loader):
            iteration += 1

            optim.zero_grad()
            out = net(img.to(args['device']))
            loss = f_loss(out, label.to(args['device']))
            acu_loss_gran += loss.item()
            acu_loss_epoch += loss.item()
            loss.backward()
            optim.step()

            if N_ITER and iteration%N_ITER==0:
                loss_data['iteration'].append(iteration)
                loss_data['loss'].append(acu_loss_gran/N_ITER)
                print(f'Iteration: {iteration}. Loss: {acu_loss_gran/N_ITER:.4f}')
                acu_loss_gran = 0

        acu_loss_epoch = acu_loss_epoch/(ite+1)

        state_dict = {
            'epoch': epoch
            , 'net': net.state_dict()
            , 'optimizer': optim.state_dict()
            , 'loss_data': loss_data
            , 'best_val': best_val
            , 'best_model': best_model
        }
        #validate, record progress
        validation_loss = val_loss(net, val_loader, f_loss, args['device'])
        #validate model
        if validation_loss < best_val:
            best_val = validation_loss
            best_model = epoch
            state_dict['best_val'] = validation_loss
            state_dict['best_model'] = best_model
            torch.save(state_dict, osp.join(args['log_dir'], 'best_state.pth'))
        torch.save(state_dict, osp.join(args['log_dir'], 'last_state.pth'))
        print(f'\nEpoch: {epoch+1}/{args["max_epoch"]}. Train Loss: {acu_loss_epoch:.4f}. Val Loss: {validation_loss}. Best_model: {epoch}')
        
def test(args):
    pass

if __name__ == '__main__':
    args = parse_args()
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    random.seed(args['seed'])
    if args['mode'] == 'train':
        train(args)
    test(args)