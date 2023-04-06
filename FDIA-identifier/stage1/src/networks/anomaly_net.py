import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

# class Anomaly_Net(BaseNet):
#     def __init__(self):
#         super().__init__()
#         self.rep_dim = 32
#         self.layer_num = 4
#         channels = [128, 256, 128, 64]
#         kernels = [5, 3, 3, 3]
#         layers = []
#         for i in range(self.layer_num):
#             in_channel = 1 if i == 0 else channels[i-1]
#             out_channel = channels[i]
#             layers += [nn.Sequential(
#                     nn.Conv1d(in_channel, out_channel, kernel_size=kernels[i], stride=1, padding=kernels[i]//2),
#                     nn.BatchNorm1d(out_channel),
#                     nn.LeakyReLU(0.4))
#                 ]

#         self.layers = layers
#         self.fc = nn.Sequential(
#             nn.Linear(64 * 19, self.rep_dim),
#             nn.Sigmoid()
#         )
#     def forward(self, x):
#         x = x.to(torch.float32).unsqueeze(1) # N, 1, V
#         for i in range(self.layer_num):
#             x = self.layers[i](x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# class Anomaly_Net_Autoencoder(BaseNet):
#     def __init__(self):
#         super().__init__()
#         self.rep_dim = 32
#         self.encoder = Anomaly_Net()
#         self.decoder = self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(self.rep_dim, 19),
#             torch.nn.ReLU(),
#         )
    
#     def forward(self, x):
#         x = self.encoder(x)
#         #print("Anomaly Net_AutoEncoder encoder", x.shape)
#         x = self.decoder(x)
#         #print("Anomaly Net_AutoEncoder decoder", x.shape)
#         return x


class Anomaly_Net(BaseNet):

    def __init__(self):
        super().__init__()
        self.rep_dim = 4
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(19, self.rep_dim),
            torch.nn.ReLU(),
            # torch.nn.Linear(16, 12),
            # torch.nn.ReLU(),
            # torch.nn.Linear(12, 8),
            # torch.nn.ReLU(),
            # torch.nn.Linear(8, self.rep_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Anomaly_Net_Autoencoder(BaseNet):
    def __init__(self):
        super().__init__()
        self.rep_dim = 4
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(19, self.rep_dim),
            # torch.nn.ReLU(),
            # torch.nn.Linear(16, 12),
            # torch.nn.ReLU(),
            # torch.nn.Linear(12, 8),
            # torch.nn.ReLU(),
            # torch.nn.Linear(8, self.rep_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.rep_dim, 19),
            torch.nn.ReLU(),
            # torch.nn.Linear(8, 12),
            # torch.nn.ReLU(),
            # torch.nn.Linear(12, 16),
            # torch.nn.ReLU(),
            # torch.nn.Linear(16, 19),
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
