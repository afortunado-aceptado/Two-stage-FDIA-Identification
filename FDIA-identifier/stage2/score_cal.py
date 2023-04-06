from torch import nn
import torch
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.4),
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResBlock, in_channel=128):
        super(ResNet, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, in_channel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(0.4)
        )
        self.layer1 = self.make_layer(ResBlock, 256, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, stride=1)
        self.layer3 = self.make_layer(ResBlock, 64, stride=1)
        #self.layer4 = self.make_layer(ResBlock, 512, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 19, 19)
        self.score_output = nn.Sigmoid()
  
    def make_layer(self, block, channels, num_blocks=1, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x): 
        x = x.to(torch.float32).permute(0, 2, 1) # N, 1, V
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        #out = F.avg_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.score_output(out)

class ConvNet(nn.Module):
    def __init__(self, node_num, layer_num=4):
        super(ConvNet, self).__init__()
        self.layer_num = layer_num
        channels = [128, 256, 128, 64]
        kernels = [5, 3, 3, 3]
        layers = []
        for i in range(layer_num):
            in_channel = 1 if i == 0 else channels[i-1]
            out_channel = channels[i]
            layers += [nn.Sequential(
                    nn.Conv1d(in_channel, out_channel, kernel_size=kernels[i], stride=1, padding=kernels[i]//2),
                    nn.BatchNorm1d(out_channel),
                    nn.LeakyReLU(0.4))
                ]

        self.layers = layers
        self.fc = nn.Sequential(
            nn.Linear(64 * node_num, node_num),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.to(torch.float32).permute(0, 2, 1) # N, 1, V
        for i in range(self.layer_num):
            x = self.layers[i](x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ScoreCalculator(node_num, model_name="mine", *args):
    if model_name == "mine":
        return ResNet(ResBlock)
    elif model_name == "conv":
        return ConvNet(node_num)
        

import os
import time
import copy
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

class Trainer(object):
    def __init__(self, model, *, epoch, lr, threshold=0.5, device="cpu", res_dir="./res"):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epoch = epoch
        self.device = torch.device(device)
        self.threshold = threshold
        self.model_save_file = os.path.join(res_dir, "model.ckpt")
        self.loss_fn = nn.BCELoss()
    
    def load_model(self, model_save_file=""):
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def save_model(self, state):
        try:
            torch.save(state, self.model_save_file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, self.model_save_file)

    def evaluate(self, test_dl, normal_total=0):
        self.model.eval()
        pred, gdth = [0]*normal_total, [0]*normal_total
        test_loss = 0.0
        with torch.no_grad():
            for X, Y in test_dl:
                scores = self.model(X.unsqueeze(-1).to(self.device))
                pred.extend(np.where(scores.flatten().cpu().numpy() > self.threshold, 1, 0).tolist())
                gdth.extend(Y.flatten().cpu().numpy().tolist())
                test_loss += self.loss_fn(scores.to(self.device), Y.to(torch.float32).to(self.device)).item()
            print("Test loss", round(test_loss, 5))
                
        eval_res = {
            "f1": f1_score(gdth, pred),
            "rc": recall_score(gdth, pred),
            "pc": precision_score(gdth, pred),
        }
        print("{}".format(",".join([k+":"+str(f"{v:.4f}") for k, v in eval_res.items()])))
        return eval_res

    
    def fit(self, train_dl, test_dl, normal_total):
        best_f1 = -1
        best_state, best_test_scores = None, None

        for epoch in range(1, self.epoch+1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            batch_start = time.time()

            for X, Y in train_dl:
                self.optimizer.zero_grad()
                scores = self.model(X.unsqueeze(-1).to(self.device))
                loss = self.loss_fn(scores.to(self.device), Y.to(torch.float32).to(self.device))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            
            print("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoch, epoch_loss, time.time()-batch_start))

            if epoch % 1 == 0:
                test_results = self.evaluate(test_dl, normal_total)
                if test_results["f1"] > best_f1:
                    best_f1 = test_results["f1"]
                    best_test_scores = test_results
                    best_state = copy.deepcopy(self.model.state_dict())

        self.save_model(best_state)
        self.load_model(self.model_save_file)
        return best_test_scores

    
if __name__ == "__main__":
    node_num = 19
    batch_size = 128
    x = torch.randn(batch_size, node_num, 1)
    net = ScoreCalculator(node_num)
    scores = net(x)
    print(scores.shape)
