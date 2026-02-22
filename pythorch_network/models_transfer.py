import torch.nn as nn
from torchvision import models


class CartoonResNet(nn.Module):
    def __init__(self, config):
        super(CartoonResNet, self).__init__()
        # טעינת המודל עם משקולות ImageNet
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # הקפאה ראשונית (Warmup)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # החלפת הראש (Classifier) לפי ה-Config
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.resnet(x)

    def unfreeze_limited(self):
        """פתיחת שכבות מבוקרת למניעת Overfitting"""
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True