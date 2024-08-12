import torch
import torch.nn as nn
import torchvision.models as models


# Define the PretrainedModel class
class PretrainedModel(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(PretrainedModel, self).__init__()
        # self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Dropout(dropout_rate),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def frozen_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True


# Define the function to get the model
def get_Image_model(
        num_classes,
        dropout_rate,
):
    model = PretrainedModel(num_classes, dropout_rate)
    model.frozen_layers()
    return model
