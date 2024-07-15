import torch
import torch.nn as nn
import torch.nn.functional as F
from Module import ASPP
import sys
sys.path.append('D:/Workplace/MNet/MNet/net/backbone')
from ResNet import ResNet101, ResNet18, ResNet34, ResNet50


INPUT_SIZE = 512

class ResNet_ASPP(nn.Module):
    def __init__(self, nInputChannels, os, backbone_type):
        super(ResNet_ASPP, self).__init__()

        self.os = os
        self.backbone_type = backbone_type

        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8 or os == 32:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if backbone_type == 'resnet18':
            self.backbone_features = ResNet18(nInputChannels, os, pretrained=True)
        elif backbone_type == 'resnet34':
            self.backbone_features = ResNet34(nInputChannels, os, pretrained=True)
        elif backbone_type == 'resnet50':
            self.backbone_features = ResNet50(nInputChannels, os, pretrained=True) # 16代表的一方面是ASPP的倍率,另一方面是ASPP输出的尺寸
        elif self.backbone_type == 'resnet101':
            self.backbone_features = ResNet101(nInputChannels, os, pretrained=True)
        else:
            raise NotImplementedError

        asppInputChannels = 512
        asppOutputChannels = 64  #适配统一的输出通道
        if backbone_type == 'resnet50' or backbone_type == 'resnet101': asppInputChannels = 2048

        self.aspp = ASPP(asppInputChannels, asppOutputChannels, rates)

    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, input):
        conv_feat, layer1_feat, layer2_feat, layer3_feat, layer4_feat = self.backbone_features(input)
        if self.os == 32:
            layer4_feat = F.interpolate(layer4_feat, scale_factor=4, mode='bilinear', align_corners=True)
        aspp_out = self.aspp(layer4_feat)

        layer = []
        layer.append(layer1_feat)
        layer.append(layer2_feat)
        layer.append(layer3_feat)
        layer.append(layer4_feat)

        return layer, aspp_out
    
class ResNet_NO_ASPP(nn.Module):
    def __init__(self, nInputChannels, os, backbone_type):
        super(ResNet_NO_ASPP, self).__init__()

        self.os = os
        self.backbone_type = backbone_type

        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8 or os == 32:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        if backbone_type == 'resnet18':
            self.backbone_features = ResNet18(nInputChannels, os, pretrained=True)
        elif backbone_type == 'resnet34':
            self.backbone_features = ResNet34(nInputChannels, os, pretrained=True)
        elif backbone_type == 'resnet50':
            self.backbone_features = ResNet50(nInputChannels, os, pretrained=True) # 16代表的一方面是ASPP的倍率,另一方面是ASPP输出的尺寸
        elif self.backbone_type == 'resnet101':
            self.backbone_features = ResNet101(nInputChannels, os, pretrained=True)
        else:
            raise NotImplementedError


    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, input):
        conv_feat, layer1_feat, layer2_feat, layer3_feat, layer4_feat = self.backbone_features(input)
        if self.os == 32:
            layer4_feat = F.interpolate(layer4_feat, scale_factor=4, mode='bilinear', align_corners=True)

        layer = []
        layer.append(layer1_feat)
        layer.append(layer2_feat)
        layer.append(layer3_feat)
        layer.append(layer4_feat)

        return layer