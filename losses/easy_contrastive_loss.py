import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class ContrastiveLoss(nn.Module):
    def __init__(self,perceptual_loss):
        super().__init__()
        self.vgg_features = perceptual_loss.vgg_features  # 共享VGG特征层
        self.extract_layer = perceptual_loss.extract_feature_layer  # 共享特征层索引
        self.mean = perceptual_loss.mean  # 共享标准化均值
        self.std = perceptual_loss.std  # 共享标准化标准差

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1)) #压缩为1x1，展平过后为特征向量，实质是取8个值的平均

    def forward(self, derained, truth):
        """预处理"""
        x = derained
        x = torch.clamp(x, 0, 1)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)  # 标准化
        y = truth
        y = torch.clamp(y, 0, 1)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)
        for idx, layer in enumerate(self.vgg_features):
            x = layer(x)
            y = layer(y)
            if idx in self.extract_layer:
                feature = x
                label = y
                break
        feature = self.avg_pool(feature).flatten(1)
        label = self.avg_pool(label).flatten(1)
        loss = (-1) * F.cosine_similarity(feature, label, dim=1).mean() #对批次取平均
        return loss

