from .st_att_layer import *  # 导入st_att_layer.py中的所有内容
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter # 使用tensorboard可视化网络架构图



class DG_STA(nn.Module): # 修改网络结构DG-STA以适应我们自己的数据集
    def __init__(self, num_classes, dp_rate, time_len, coordinates_nums):
        super(DG_STA, self).__init__()

        h_dim = 32 # the dimension d of the query\key\value vector
        h_num = 8 # head number of the spatial and temporal attention model

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
        # input shape: [batch_size, time_len, joint_num, coordinates_nums]
        # print(x.size())
        time_len = x.shape[1]
        joint_num = x.shape[2]

        # reshape x
        x = x.reshape(-1, time_len * joint_num, self.coordinates_nums)
        # print(x.shape) # [batch_size, time_len * joint_num, coordinates_nums]

        # input map
        x = self.input_map(x)

        # print(x.size()) # [batch_size, time_len * joint_num, 128]

        # exec()

        # spatial
        x = self.s_att(x) # [batch_size, time_len * joint_num, 128]
        # temporal
        x = self.t_att(x) # [batch_size, time_len * joint_num, 128]

        # print(x.size()) # [batch_size, time_len * joint_num, 128]
        # print(x.sum(1).size()) # [batch_size, 128]

        x = x.sum(1) / x.shape[1]
        # print(x.size()) # [batch_size, 128]
        pred = self.cls(x)
        # print(pred.size()) # [batch_size, num_classes]
        return pred


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DG_STA(7, 0.2, 41, 5)
    model.to(device)
    print(model)

    # 用来测试上面定义的模型是否是可用的
    input = torch.randn((1, 41, 2, 5))
    output = model(input.to(device))
    print("input size: {}".format(input.shape))
    print("output size: {}".format(output.shape))
    # 在tensorboard上可视化自己搭建的神经网络
    writer = SummaryWriter('../log_graph')
    writer.add_graph(model, input.to(device))         # "ctrl+p"快捷键可以查看括号里面应该写什么参数
    writer.close()