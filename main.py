import torch, torch.nn as nn
import argparse, sys
import os, os.path as osp
import random
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from dataset import datasets
from architecture import models
from utils.RAdam import RAdam
from utils.Logger import Logger
from utils.val_loss import val_loss
from utils.plot_loss import plot_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Garbage Classification Configuration')

    parser.add_argument('-d', '--dataset', type=str, default='GarbageClass',
                        choices=datasets.keys(),  help=f"options: {datasets.keys()}")
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory to store logs and trained models')
    parser.add_argument('--model', type=str, default='resnet-18', choices=models.keys(), help=f'Options: {models.keys()}')
    parser.add_argument('--lr', type=float, default=1e-4, help=f'Learning Rate')
    parser.add_argument('--pretrained', action='store_true') #always pass it, even during training
    parser.add_argument('--max-epoch', type=int, default='100')
    parser.add_argument('--patience', default=-1, type=int,
                        help="number of epochs without model improvement (-1 to disable it)")
    parser.add_argument('--seed', type=int, default=64678, help="manual seed")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help="root directory where part/fold of previous train are saved")
    parser.add_argument('--mode', type=str, default='train', choices=['train','test','plot'], help='Options: [train, test, plot]') #test after training though

    return vars(parser.parse_args())

def train(args):
    try:
        os.makedirs(args['log_dir'])
    except FileExistsError:
        pass

    train_set = datasets[args['dataset']](mode='train', size=models[args['model']][1], args=args) #pass other args too
    val_set = datasets[args['dataset']](mode='val', args=args, size=models[args['model']][1])

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

    net = models[args['model']][0](train_set.get_nclass(), pretrained=args['pretrained']) #pass args
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
        iteration = state_dict['iteration']
        print(f'\nResuming training. Epoch {initial_epoch+1}\n')
    else:
        sys.stdout = Logger(osp.join(args['log_dir'], 'log_train.txt'), mode='w')
        initial_epoch = -1
        iteration = 0
        best_val = sys.maxsize
        best_model = None
        loss_data = {'loss_train': [], 'loss_val': [], 'epoch': []}

    for epoch in range(initial_epoch+1, args['max_epoch']): #train_loop
        acu_loss_epoch = 0
        for ite, (img, label) in enumerate(train_loader):
            iteration += 1
            img = img.to(args['device'])
            label = label.to(args['device'])
            optim.zero_grad()
            if args['model'] == 'inception-v3':
                out, aux_out = net(img)
                loss = f_loss(out, label) + 0.4*f_loss(aux_out, label)
            else:
                out = net(img)
                loss = f_loss(out, label)
            acu_loss_epoch += loss.item()
            loss.backward()
            optim.step()

        acu_loss_epoch = acu_loss_epoch/(ite+1)
        validation_loss, val_acc = val_loss(net, val_loader, f_loss, args['device'])
        loss_data['epoch'].append(epoch+1)
        loss_data['loss_train'].append(acu_loss_epoch)
        loss_data['loss_val'].append(validation_loss)

        plot_loss(loss_data, args['log_dir'])

        state_dict = {
            'epoch': epoch
            , 'net': net.state_dict()
            , 'optimizer': optim.state_dict()
            , 'loss_data': loss_data
            , 'best_val': best_val
            , 'best_model': best_model
            , 'iteration': iteration
        }
        #validate model
        if validation_loss < best_val:
            best_val = validation_loss
            best_model = epoch+1
            state_dict['best_val'] = validation_loss
            state_dict['best_model'] = best_model
            torch.save(state_dict, osp.join(args['log_dir'], 'best_state.pth'))
        torch.save(state_dict, osp.join(args['log_dir'], 'last_state.pth'))
        print(f'Epoch: {epoch+1}/{args["max_epoch"]}. Train Loss: {acu_loss_epoch:.4f}. Val Loss: {validation_loss:.4f}. Val Acc: {val_acc:.4f}. Best_model: {best_model}')
        sys.stdout.close_open()

def test(args):
    if not osp.isdir(args['log_dir']):
        raise Exception(f'Missing directory: {args["log_dir"]}')

    test_set = datasets[args['dataset']](mode='test', size=models[args['model']][1], args=args)
    test_loader = DataLoader(test_set
                            , batch_size=1
                            , shuffle=False
                            #, sampler=valSampler
                            , pin_memory=True
                            , drop_last=False)

    net = models[args['model']][0](test_set.get_nclass(), pretrained=args['pretrained']) #pass args
    net.to(args['device'])

    sys.stdout = Logger(osp.join(args['log_dir'], 'log_test.txt'), mode='w')
    state_dict = torch.load(osp.join(args['log_dir'], 'best_state.pth'))
    net.load_state_dict(state_dict['net'])
    net.eval()

    gts = []
    preds = []

    for i, (img, label) in enumerate(test_loader):
        preds.append(torch.argmax(net(img.to(args['device']))).item())
        gts.append(label.item())

    print(f'Testing model: {args["model"]}\n')
    print(f'Accuracy: {accuracy_score(gts, preds):.6f}')
    print(f'Precision: {precision_score(gts, preds, average="macro"):.6f}')
    print(f'Recall: {recall_score(gts, preds, average="macro"):.6f}')
    print(f'F1 Score: {f1_score(gts, preds, average="macro"):.6f}')
    print('\n')
    print('Confusion Matrix')
    print(confusion_matrix(gts, preds))

def plot(args):
    loss = torch.load(osp.join(args['log_dir'], 'last_state.pth'))['loss_data']
    plot_loss(loss, args['log_dir'], 15)

if __name__ == '__main__':
    args = parse_args()
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args['mode'] == 'plot':
        plot(args)
        
    random.seed(args['seed'])
    if args['mode'] == 'train':
        train(args)
    if args['mode'] in ['train', 'test']:
        test(args)
