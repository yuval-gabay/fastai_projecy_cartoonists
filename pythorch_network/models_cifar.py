# models_cifar.py
import torch
import torch.nn as nn


class CIFARNet(nn.Module):
    def __init__(self, config):
        super(CIFARNet, self).__init__()
        self.use_bn = config.get('use_batch_norm', True)
        dr = config.get('dropout_rate', 0.4)
        final_filters = config.get('final_layer_filters', 512)

        # Deep & Slim architecture adapted for 32x32 input
        self.features = nn.Sequential(
            self._make_layer(3, 32),  # 32x32 -> 16x16
            self._make_layer(32, 64),  # 16x16 -> 8x8
            self._make_layer(64, 128),  # 8x8 -> 4x4
            self._make_layer(128, 256),  # 4x4 -> 2x2
            self._make_layer(256, final_filters)  # 2x2 -> 1x1
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dr),
            nn.Linear(final_filters, 256),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(256, 10)  # 10 classes for CIFAR-10
        )

    def _make_layer(self, in_c, out_c):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
        if self.use_bn: layers.append(nn.BatchNorm2d(out_c))
        layers.extend([nn.ReLU(), nn.MaxPool2d(2)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)