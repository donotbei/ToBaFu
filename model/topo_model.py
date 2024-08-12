import torch.nn as nn


# 定义线性特征输入的残差块
class LinearResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinearResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += x  # 跳跃连接
        out = self.relu(out)
        return out


# 定义线性特征输入的残差网络
class LinearResNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, num_classes, dropout_rate):
        super(LinearResNet, self).__init__()
        self.TopoNet = nn.Sequential()
        self.TopoNet.add_module('fc', nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
        ))

        self.TopoNet.add_module('res_blocks', self.make_blocks(hidden_size, num_blocks))
        self.TopoNet.add_module('fc_out', nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Dropout(dropout_rate),
            nn.Softmax(dim=1),
        ))

    def make_blocks(self, hidden_size, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(LinearResidualBlock(hidden_size, hidden_size))
        return nn.Sequential(*blocks)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        net = self.TopoNet
        out = net(x)
        return out


# 定义函数获取模型
def get_TOPO_model(
        input_size,
        num_classes,
        hidden_size,
        num_blocks,
        dropout_rate,
):
    TOPO_model = LinearResNet(input_size, hidden_size, num_blocks, num_classes, dropout_rate)
    TOPO_model.apply(TOPO_model.init_weights)
    return TOPO_model
