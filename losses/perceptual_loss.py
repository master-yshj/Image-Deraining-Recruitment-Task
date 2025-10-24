import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
import matplotlib.pyplot as plt #因为要可视化就加了一个这个



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.net = vgg
        self.vgg_features = vgg.features
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
        self.extract_feature_layer = [35]
        # VGG输入的标准化参数（ImageNet数据集的均值和标准差，RGB通道）
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


    def forward(self, generated, ground_truth):
        """预处理"""
        x = generated
        x = torch.clamp(x, 0, 1)    #归一化
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)    #标准化
        y = ground_truth
        y = torch.clamp(y, 0, 1)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)
        loss = 0
        for idx, layer in enumerate(self.vgg_features):
            x = layer(x)
            y = layer(y)
            if idx in self.extract_feature_layer:
                feature = x
                label = y
                break
        loss += F.mse_loss(feature, label, reduction='mean')
        # """可视化（修复后）"""
        # feat_vis = x[0].cpu().detach()  #取批次中的第一个
        # feat_single = feat_vis[0]   #转为单通道
        #
        # feat_min = feat_single.min()
        # feat_max = feat_single.max()
        # feat_norm = (feat_single - feat_min) / (feat_max - feat_min + 1e-8)  # 加1e-8防除0
        #
        # # 3. 显示
        # plt.figure(figsize=(4, 4))
        # plt.imshow(feat_norm, cmap='viridis')
        # plt.show()
        return loss



