import torch
import torch.nn as nn


class DeepCartoonNet(nn.Module):
    def __init__(self, config):
        super(DeepCartoonNet, self).__init__()
        self.use_bn = config['use_batch_norm']
        dr = config['dropout_rate']
        # Dynamic filter number for the last feature layer
        final_filters = config.get('final_layer_filters', 512)

        self.features = nn.Sequential(
            self._make_layer(3, 16),
            self._make_layer(16, 32),
            self._make_layer(32, 64),
            self._make_layer(64, 128),
            self._make_layer(128, final_filters)  # Flexible last layer
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dr),
            nn.Linear(final_filters, 256),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(256, 3)
        )

    def _make_layer(self, in_c, out_c):
        layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
        if self.use_bn: layers.append(nn.BatchNorm2d(out_c))
        layers.extend([nn.ReLU(), nn.MaxPool2d(2)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

class WideCartoonNet(nn.Module):
    def __init__(self, config):
        super(WideCartoonNet, self).__init__()
        dr = config['dropout_rate']

        # שכבות רחבות מאוד אבל פחות עומק
        self.features = nn.Sequential(
            self._make_layer(3, 64),
            self._make_layer(64, 128),
            self._make_layer(128, 512)  # קפיצה ל-512 פילטרים מהר יותר
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dr),  # Dropout ראשון
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dr),  # Dropout שני למלחמה ב-Overfitting
            nn.Linear(256, 3)
        )

    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)