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
    python3 main.py --model_name models/1.cpt
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
    print(f"device is {device}")
    
    # ここでハイパーパラメータを変える
    model = ViT(
        image_size=32,
        patch_size=8,
        num_classes=10,
        dim=256,
        depth=12,
        heads=8,
        mlp_dim=2048,
    )
    
    cpt = torch.load(args.model_name)
    stdict_m = cpt['model_state_dict']
    model.load_state_dict(stdict_m)
    model.to(device)

    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    test_dataset = CIFAR10(root='./data', train=False,
                            transform=transform, download=True)
    # indices = np.random.choice(
    #     len(test_dataset), size=5000, replace=False)
    # test_subset = Subset(test_dataset, indices)
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False
    )
    try:
        with tqdm(enumerate(test_dataloader), total=len(test_dataloader), ncols=100) as pbar:
            result = np.zeros((args.num_class, args.num_class))
            t_p, f_p, t_n, f_n, = np.zeros(args.num_class), np.zeros(args.num_class), np.zeros(args.num_class), np.zeros(args.num_class)
            model.eval()

            for i, batch in pbar:
                image, label = batch
                image = image.to(device)
                label = label.to(device)
                logit = model(image)
                predict = torch.argmax(torch.softmax(logit, dim=-1)).cpu().numpy()
                label = label.cpu().numpy()
                result[label, predict] += 1
            
            for posclass in range(args.num_class):
                total = result.sum()
                p = result[:, posclass].sum()
                n = total - p
                t_p[posclass] = result[posclass, posclass]
                f_p[posclass] = p - t_p[posclass]
                tp_and_fn = result[posclass, :].sum()
                f_n[posclass] = tp_and_fn - t_p[posclass]
                t_n[posclass] = n - f_n[posclass]

            accuracy, precision, recall = 0, 0, 0
            for cls in range(args.num_class):
                accuracy += (t_p[cls])/(total)
                precision += (t_p[cls])/(t_p[cls] + f_p[cls])
                recall += (t_p[cls])/(t_p[cls] + f_n[cls])
            accuracy = accuracy
            precision /= args.num_class
            recall /= args.num_class
            f_score = 2/((1/precision) + (1/recall))
            
            print(f'accuracy:{accuracy}, precision:{precision}, recall:{recall}, F_Score:{f_score}')
    except ValueError:
        pass
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--num_class',  default=10)
    parser.add_argument('--output_dir', default='output')
    args = parser.parse_args()
    main(args)