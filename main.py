import argparse

import torch

from train import train_model
from modules import utils
from modules import MVTecTrainDataset, MVTecTestDataset
from modules import FastFlow

dataset_path = '../../../datasets/MVTec'



def run(backbone, dataset_path, classname, show):
    dataloader = MVTecTrainDataset(dataset_path, classname).get_dataloader(num_workers=8)
    test_dataloader = MVTecTestDataset(dataset_path, classname).get_dataloader()

    if show:
        for id_batch, data in enumerate(dataloader):
            utils.batch2grid(data, id_batch)

    model = FastFlow(backbone)
    train_model(model, dataloader, test_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='wide_resnet50_2')
    parser.add_argument('--dataset', type=str, default='../../../datasets/MVTec')
    parser.add_argument('--classname', type=str, default='pill')
    parser.add_argument('--show-batch', action='store_true', help = 'keeps in tests folder how the batch looks')

    args = parser.parse_args()

    run(args.backbone, args.dataset, args.classname, args.show_batch)











