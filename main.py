import torch, torch.nn as nn
import argparse, sys
import os, os.path as osp
import random
from torch.utils.data import DataLoader

from dataset import datasets
from architecture import models
from utils.RAdam import RAdam
from utils.Logger import Logger

def parse_args():
    parser = argparse.ArgumentParser(description='Garbage Classification Configuration')

    parser.add_argument('-d', '--dataset', type=str, default='GarbageClass',
                        choices=datasets.keys(),  help=f"options: {datasets.keys()}")
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to store logs and trained models')
    parser.add_argument('-model', type=str, default='resnet-20', choices=models.keys(), help=f'Options: {models.keys()}')
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
                                , drop_last=False)

    val_loader = DataLoader(val_set
                            , batch_size=1
                            , shuffle=True
                            #, sampler=valSampler
                            , pin_memory=True
                            , drop_last=False)

    net = models[args['model']]() #pass args
    net.to(args['device'])

    optim = RAdam(net.parameters(), lr=args['lr'])

    f_loss = nn.CrossEntropyLoss()

    if args['resume']:
        sys.stdout = Logger(osp.join(args['log_dir'], 'log_train.txt'), mode='a') #todo print é salvo em um arquivo tambem
        #carregar statedict
        pass
    else:
        sys.stdout = Logger(osp.join(args['log_dir'], 'log_train.txt'), mode='w')
        initial_epoch = -1

    for epoch in range(initial_epoch+1, args['max_epoch']): #train_loop
        for ite, data in enumerate(train_loader):
            #train process
                #zero o gradiente do optmizer
                #passa imagens pela rede, calcula o custo, dá um passo do optmizer 
            #print progress
            pass

        #validate, record progress
        state_dict = {
            'epoch': epoch
            , 'net': net.state_dict()
            , 'optimizer': optim.state_dict()
        }
        torch.save(state_dict, osp.join(args['log_dir'], 'last_state.pth'))
        #validate model
        best_val = False
        if best_val:
            torch.save(state_dict, osp.join(args['log_dir'], 'best_state.pth'))
        
def test(args):
    pass

if __name__ == '__main__':
    args = parse_args()
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(args)
    random.seed(args['seed'])
    if args['mode'] == 'train':
        train(args)
    test(args)