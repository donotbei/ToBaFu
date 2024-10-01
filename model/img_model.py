import torch
import torch.nn as nn
import torchvision.models as models


# Define the ModifiedResNet50
class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(ModifiedResNet50, self).__init__()
        # self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features

        # Remove the original global average pooling layer and fully connected layer
        self.model.avgpool = nn.Identity()
        self.model.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=256, kernel_size=1),
            nn.ReLU()
        )
        self.model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 7 * 7, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def frozen_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.conv.parameters():
            param.requires_grad = True


# Define the function to get the model
def get_Image_model(
        num_classes,
        dropout_rate,
):
    model = ModifiedResNet50(num_classes, dropout_rate)
    model.frozen_layers()
    return model
