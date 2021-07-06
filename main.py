import torch
import argparse
import os, os.path as osp

from dataset import datasets
from architecture import models

def parse_args():
    parser = argparse.ArgumentParser(description='Garbage Classification Configuration')

    parser.add_argument('-d', '--dataset', type=str, default='ucf-cc-50',
                        choices=datasets.keys(),  help=f"options: {datasets.keys()}")
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to store logs and trained models')
    parser.add_argument('-model', type=str, default='resnet-20', choices=models.keys(), help=f'Options: {models.keys()}')
    parser.add_argument('--lr', type=float, default=1e-4, help=f'Learning Rate')
    parser.add_argument('--max-epoch', type=int, default='100')
    parser.add_argument('--patience', default=-1, type=int,
                        help="number of epochs without model improvement (-1 to disable it)")
    parser.add_argument('--seed', type=int, default=64678, help="manual seed")
    parser.add_argument('--resume', type=str, default='', metavar='PATH', help="root directory where part/fold of previous train are saved")

    return parser.parse_args()

def train(args):
    pass

if __name__ == '__main__':
    args = parse_args()
    train(args)