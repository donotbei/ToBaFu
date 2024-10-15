import torch.nn as nn


# Define the residual block for linear input
class LinearResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinearResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        out += x
        out = self.gelu(out)
        return out


# Define the linear ResNet model
class LinearResNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks, num_classes, dropout_rate):
        super(LinearResNet, self).__init__()
        self.TopoNet = nn.Sequential()
        self.TopoNet.add_module('fc', nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
        ))

        self.TopoNet.add_module('res_blocks', self.make_blocks(hidden_size, num_blocks))
        self.TopoNet.add_module('fc_out', nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes),
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
