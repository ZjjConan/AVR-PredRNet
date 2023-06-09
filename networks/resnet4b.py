import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_utils import (
    ResBlock, ConvNormAct, convert_to_rpm_matrix_v9, Classifier
)


class ResNet(nn.Module):

    def __init__(
        self, 
        num_filters=32, 
        block_drop=0.0, 
        classifier_drop=0.0, 
        classifier_hidreduce=1.0,
        in_channels=1,
        num_classes=1,
        num_extra_stages=1
    ):
        super().__init__()

        channels = [num_filters, num_filters*2, num_filters*3, num_filters*4]
        strides = [2, 2, 2, 2]

        # -------------------------------------------------------------------
        # frame encoder 

        self.inplanes = in_channels
        self.num_stages = len(strides)
        self.num_extra_stages = num_extra_stages

        for l in range(self.num_stages):
            setattr(
                    self, "res"+str(l), 
                    self._make_layer(
                        channels[l], 
                        stride = strides[l], 
                        block = ResBlock, 
                        dropout = block_drop
                    )
                )
        # -------------------------------------------------------------------

        for l in range(self.num_extra_stages):            
            setattr(
                self, "res"+str(4+l), 
                self._make_layer(
                    self.inplanes, 
                    stride=1,
                    block=ResBlock, 
                    dropout=block_drop
                )
            )

        self.featr_dims = 1024

        self.classifier = Classifier(
            self.featr_dims, num_classes, 
            norm_layer=nn.BatchNorm1d, 
            dropout=classifier_drop, 
            hidreduce=classifier_hidreduce
        )

        self.in_channels = in_channels
        self.ou_channels = 8


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, planes, stride, block, dropout):
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity(),
            ConvNormAct(self.inplanes, planes, 1, 0, activate=False),
        )

        stage = block(self.inplanes, planes, downsample, stride=stride, dropout=dropout)

        self.inplanes = planes

        return stage


    def forward(self, x):

        b, n, h, w = x.size()
        if self.in_channels == 1:
            x = x.reshape(b*n, 1, h, w)

        for l in range(self.num_stages+self.num_extra_stages):
            x = getattr(self, "res"+str(l))(x)

        _, _, h, w = x.size()
        # for raven
        if self.ou_channels == 8:
            x = convert_to_rpm_matrix_v9(x, b, h, w)

        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, self.featr_dims)    
        x = x.reshape(-1, self.featr_dims)
        
        out = self.classifier(x)

        if self.ou_channels == 1:
            return out
        else:
            return out.view(b, self.ou_channels)

def resnet4b(**kwargs):
    return ResNet(**kwargs)



