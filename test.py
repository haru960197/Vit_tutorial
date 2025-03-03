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
import os

from params import params_set

"""
    python3 inference.py --model_name models/1.cpt
    のようにして使用
"""

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = CIFAR10(root='./data', train=False,
                            transform=transform, download=True)
indices = np.random.choice(
    len(test_dataset), size=3000, replace=False)
test_subset = Subset(test_dataset, indices)
test_dataloader = DataLoader(
    dataset=test_subset,
    batch_size=1,
    shuffle=False
)


def main(args, patch_size, dim, depth, heads, mlp_dim, out_dir, model_name, model_num):
    """
    データセットを利用して，訓練済みのモデルの性能を評価する
    """
    model = ViT(
        image_size=32,
        patch_size=patch_size,
        num_classes=args.num_class,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    cpt = torch.load(model_name)
    stdict_m = cpt['model_state_dict']
    model.load_state_dict(stdict_m)
    model.to(device)

    try:
        with tqdm(enumerate(test_dataloader), total=len(test_dataloader), ncols=100) as pbar:
            result = np.zeros((args.num_class, args.num_class))
            t_p, f_p, t_n, f_n, = np.zeros(args.num_class), np.zeros(
                args.num_class), np.zeros(args.num_class), np.zeros(args.num_class)
            model.eval()
            for i, batch in pbar:
                image, label = batch
                image = image.to(device)
                label = label.to(device)
                logit = model(image)
                predict = torch.argmax(
                    torch.softmax(logit, dim=-1)).cpu().numpy()
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

            # 出力先のディレクトリを作成
            p = out_dir
            if not os.path.isdir(p):
                os.makedirs(p)

            output_filename = str(model_num) + '.cpt'

            # 結果をディレクトリに保存
            torch.save({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'F_Score': f_score,
            }, os.path.join(p, output_filename))
    except ValueError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--num_class',  default=10)
    args = parser.parse_args()

    for param_name in params_set:
        print(f'-------- inference [{param_name}] --------')
        for i in range(len(params_set[param_name])):
            params = params_set[param_name][i]
            out_dir = "./results/" + str(param_name) + "/param_set-" + str(i)
            print(f'-------- case [{i}] --------')
            for j in range(0, len(os.listdir("./models/" + param_name + "/param_set-" + str(i))), 5):
                print(f'-------- load {j}.cpt --------')
                model_name = "./models/" + param_name + "/param_set-" + str(i) + "/" + str(j) + ".cpt"
                main(args, patch_size=params["patch_size"], dim=params["dim"], depth=params["depth"],
                     heads=params["heads"], mlp_dim=params["mlp_dim"], out_dir=out_dir, model_name=model_name, model_num=j)
