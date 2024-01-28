from .st_att_layer import *
import torch.nn as nn
import torch

class DG_STA(nn.Module):
    def __init__(self, num_classes, dp_rate, time_len, coordinates_nums):
        super(DG_STA, self).__init__()
        h_dim = 32
        h_num = 8
        self.coordinates_nums = coordinates_nums
        self.input_map = nn.Sequential(
            nn.Linear(coordinates_nums, 128),
            nn.ReLU(),
            LayerNorm(128),
            nn.Dropout(dp_rate),
        )
        # time_len = 1
        # input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = ST_ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len = time_len, domain="spatial")
        self.t_att = ST_ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, time_len = time_len, domain="temporal")
        self.cls = nn.Linear(128, num_classes)

    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]
        # print(x.size())
        time_len = x.shape[1]
        joint_num = x.shape[2]

        # reshape x
        x = x.reshape(-1, time_len * joint_num, self.coordinates_nums)

        # input map
        x = self.input_map(x)

        # print(x.size())
        # exec()

        # spatial
        x = self.s_att(x)
        # temporal
        x = self.t_att(x)

        # print(x.size())
        # print( x.sum(1).size())

        x = x.sum(1) / x.shape[1]
        # print(x.size())
        pred = self.cls(x)
        
        return pred