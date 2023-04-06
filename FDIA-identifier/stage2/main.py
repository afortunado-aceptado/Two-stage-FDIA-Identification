import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from score_cal import ScoreCalculator, Trainer
from utils import seed_everything, dump_params, dump_scores

class myDataset(Dataset):
    def __init__(self, X, Y):
        assert X.shape == Y.shape, "Not matched {} vs. {}".format(X.shape, Y.shape)
        print(X.shape, Y.shape)
        self.data = []
        for i in range(len(X)):
            self.data.append((X[i], Y[i]))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

import scipy.io as scio
def data_load(data_dir, dryrun=False):

    data = scio.loadmat("data/data14_1.mat")
    train_x = data['x_train'].astype(np.float32)
    train_y = data['y_train'].astype(np.float32)
    # index = np.where(train_y.sum(axis=-1) > 0)
    # train_x = train_x[index]
    # train_y = train_y[index]

    with open(os.path.join(data_dir, "test_X"), "rb") as rf:
        test_x = np.load(rf)
    with open(os.path.join(data_dir, "test_Y"), "rb") as rf:
        test_y = np.load(rf)
    detection = pd.read_csv("data/SVDD_8.csv")
    assert len(detection) == len(test_y)
    
    test_index = detection[detection.pred > 0].idx.values
    normal_num = len(test_y) - len(test_index)
    print(normal_num, len(test_index))

    test_x = test_x[test_index]
    test_y = test_y[test_index]

    if dryrun:
        train_dataset = myDataset(train_x[:400, :], train_y[:400, :])
        test_dataset = myDataset(test_x[-200:, :], test_y[-200:, :])
    else:
        train_dataset = myDataset(train_x, train_y)
        test_dataset = myDataset(test_x, test_y)
    
    return train_dataset, test_dataset, normal_num * 101, test_x.shape[1]


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dryrun", action="store_true")
parser.add_argument("--batch_size", default=200, type=int)
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("--lr", default=1e-3, type=float)

parser.add_argument("--data_dir", default="data/100times", type=str)
parser.add_argument("--model_name", default="mine", type=str, choices=["mine", "conv"])

params = vars(parser.parse_args())


def main(dryrun=False, model_name="mine", data_dir="data/400/", batch_size=128, epoch=100, lr=1e-4, threshold=0.5):
    
    seed_everything(42)

    train_dataset, test_dataset, normal_num, node_num = data_load(data_dir, dryrun)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    #model = ScoreCalculator(node_num, hidden_size=hidden_size, channels=channels, kernel_size=kernel_size)
    model = ScoreCalculator(node_num, model_name=model_name)
    if dryrun: epoch = 2

    trainer = Trainer(model,
        epoch = epoch, 
        lr = lr, 
        threshold = threshold, 
        res_dir=f"./res/{dump_params(params)}",
    )
    test_scores = trainer.fit(train_dl, test_dl, normal_total = normal_num * node_num,)

    print(test_scores)

if __name__ == "__main__":
    main(**params)









