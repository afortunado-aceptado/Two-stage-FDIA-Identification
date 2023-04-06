# import sys
# sys.path.append('/root/zwg/Deep-SVDD-PyTorch/src/')
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import numpy as np
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx
import random
import os

def convert_label(labels):
    return np.where(np.sum(labels, axis=-1) == 0, 0, 1)


def gauss_noisy(x):
    """
    对输入数据加入高斯噪声
    :param x: x轴数据
    :param y: y轴数据
    :return:
    """
    mu = 0
    sigma = 0.2
    for i in range(len(x)):
        x[i] += random.gauss(mu, sigma)
        # y[i] += random.gauss(mu, sigma)


class AnomalyDataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class=0):
        super().__init__(root)
        self.root = root
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = [1]

        train_set = Anomaly(data_path=self.root, train=True)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.labels.clone().data.numpy(),
                                                self.normal_classes)  # .clone().data.cpu().numpy()
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = Anomaly(data_path=self.root, train=False)

import os
class Anomaly(Dataset):
    def __init__(self, data_path, train=True) -> None:
        super().__init__()
        self.train = train

        with open(os.path.join(data_path, "train_X"), "rb") as rf:
            train_data = np.load(rf)
        with open(os.path.join(data_path, "test_X"), "rb") as rf:
            test_data = np.load(rf)
        with open(os.path.join(data_path, "train_Y"), "rb") as rf:
            train_label = np.load(rf)
        with open(os.path.join(data_path, "test_Y"), "rb") as rf:
            test_label = np.load(rf)
        
        node_num = train_data.shape[1]
        
        train_labels = torch.from_numpy(convert_label(train_label))
        test_labels = torch.from_numpy(convert_label(test_label))

        if self.train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels
    
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        return img, target, index

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = AnomalyDataset(root='data/generated/118-bus', normal_class=0)

    print('0')
