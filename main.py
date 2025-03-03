import argparse
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import numpy as np
import torch
from tqdm import tqdm
import timm
from vit_pytorch import ViT
from torchvision import transforms
from torchvision.datasets import CIFAR10

"""
    python3 inference.py --model_name models/1.cpt
    のようにして使用
"""

def main(args):
    """
    データセットを利用して，訓練済みのモデルの性能を評価する
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device =  'cpu'
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--num_class',  default=10)
    parser.add_argument('--output_dir', default='output')
    args = parser.parse_args()
    main(args)