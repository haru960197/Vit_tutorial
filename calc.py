import torch
from vit_pytorch import ViT
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import numpy as np
import timm.optim.optim_factory as optim_factory
import argparse
import os
import types

from trainer import trainer, validater
from params import params_set

"""
    複数の組み合わせのパラメータで学習を行い，
    結果を各ディレクトリに保存する
"""

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def do(args, patch_size, dim, depth, heads, mlp_dim, model_dir):
    model = ViT(
        image_size=32,
        patch_size=patch_size,
        num_classes=10,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
    )
    model.to(device)

    # データの前処理とデータセットの準備
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_dataset = CIFAR10(root='./data', train=True,
                            transform=transform, download=True)

    train_size = int(len(train_dataset) * 0.9)
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchsize)

    # optimizer 最適化手法の定義
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    param_groups = optim_factory.param_groups_weight_decay(model, 0.05)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.1)

    # loss 損失関数
    criterion = torch.nn.CrossEntropyLoss()

    # 最初のエポック数
    epoch_init = 0

    # 誤差が小さくなったかの管理用
    early_stopping = [np.inf, 30, 0]

    if args.cpt_file != "":
        cpt = torch.load(args.cpt_file)
        stdict_m = cpt['model_state_dict']
        stdict_o = cpt['opt_state_dict']
        early_stopping[0] = cpt['valid_loss']
        epoch_init = cpt['iter'] + 1
        model.load_state_dict(stdict_m)
        optimizer.load_state_dict(stdict_o)

    # 学習ループ
    for epoch in range(epoch_init, args.epochs):
        print(f'-------- epoch {epoch} --------')
        # train
        train_loss = trainer(train_dataloader, model, device,
                             optimizer, epoch, criterion, args)
        # validate
        with torch.no_grad():
            valid_loss = validater(valid_dataloader, model, device, criterion)

        # 出力先のディレクトリを作成
        p = model_dir
        if not os.path.isdir(p):
            os.makedirs(p)

        # early stopping
        output_filename = str(epoch) + '.cpt'
        if valid_loss < early_stopping[0]:
            # 誤差が前回より小さくなっている
            early_stopping[0] = valid_loss
            output_filename = str(epoch) + '-updated.cpt'
        print(f"valid_loss = {valid_loss}")

        # 今回のepochでの結果をmodelsディレクトリに保存
        torch.save({'iter': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    }, os.path.join(p, output_filename))


if __name__ == '__main__':
    

    args = types.SimpleNamespace()
    args.batchsize = 64
    args.epochs = 1
    args.lr = 1e-4
    args.min_lr = 1e-8
    args.warmup = 10
    args.num_class = 10
    # 過去のモデルをロードしない
    args.cpt_file = ""

    for param_name in params_set:
        print(f'-------- change [{param_name}] --------')
        for i in range(len(params_set[param_name])):
            params = params_set[param_name][i]
            out_dir = "./model-" + str(param_name) + "/param_set-"  + str(i)
            print(f'-------- case [{i}] --------')
            do(args, patch_size=params["patch_size"], dim=params["dim"], depth=params
               ["depth"], heads=params["heads"], mlp_dim=params["mlp_dim"], model_dir=out_dir)
