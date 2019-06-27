from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.utils import model_zoo

from basenet.senet import se_resnet50,se_resnext101_32x4d,se_resnext50_32x4d, se_resnext26_32x4d, se_resnet50,se_resnet152
#from oct_resnet import oct_resnet26,oct_resnet101
from basenet.nasnet import nasnetalarge
from basenet.multiscale_resnet import multiscale_resnet
from basenet.multiscale_se_resnext import multiscale_se_resnext
from basenet.multiscale_se_resnext_cat import multiscale_se_resnext_cat
from basenet.DPN import DPN92, DPN26
from basenet.SKNet import SKNet101
from basenet.multiscale_se_resnext_HR import multiscale_se_resnext_HR
import torch.nn.functional as F
import pretrainedmodels
class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MultiModalNet2(nn.Module):
    def __init__(self, backbone1, backbone2, drop, pretrained=True):
        super().__init__()
        if pretrained:
            img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained='imagenet') #seresnext101
        else:
            img_model = pretrainedmodels.__dict__[backbone1](num_classes=1000, pretrained=None)
       
        self.visit_model=DPN26()
        
        self.img_encoder = list(img_model.children())[:-1]
        self.img_encoder.append(nn.AdaptiveAvgPool2d(1))
        
        self.img_encoder = nn.Sequential(*self.img_encoder)
        if drop > 0:
            self.img_fc = nn.Sequential(FCViewer(),
                                    nn.Dropout(drop),
                                    nn.BatchNorm1d(img_model.last_linear.in_features),
                                    nn.Linear(img_model.last_linear.in_features, 512))
                                    
        else:
            self.img_fc = nn.Sequential(
                FCViewer(),
                nn.BatchNorm1d(img_model.last_linear.in_features),
                nn.Linear(img_model.last_linear.in_features, 512))
        self.bn=nn.BatchNorm1d(768)
        self.cls = nn.Linear(768,9) 

    def forward(self, x_img,x_vis):
        x_img = self.img_encoder(x_img)
        x_img = self.img_fc(x_img)
        x_vis=self.visit_model(x_vis)
        x_cat = torch.cat((x_img,x_vis),1)
        x_cat = F.relu(self.bn(x_cat))
        x_cat = self.cls(x_cat)
        return x_cat

'''
            print("load pretrained model from /home/zxw/2019BaiduXJTU/se_resnext50_32x4d-a260b3a4.pth")
            state_dict = torch.load('/home/zxw/2019BaiduXJTU/se_resnext50_32x4d-a260b3a4.pth')

            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
            self.img_encoder.load_state_dict(state_dict, strict = False)
'''
