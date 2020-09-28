from torchvision.models import resnet18
import torch

class Resnet18FeatureExtractor(torch.nn.Module):

    def __init__(self):
        base_model = resnet18(pretrained=True)
        self.features = torch.nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu,
            base_model.maxpool, base_model.layer1, base_model.layer2,
            base_model.layer3, base_model.layer4, base_model.avgpool,
            torch.nn.Flatten(start_dim=1))
    
    def forward(self, x):
        return self.features(x)