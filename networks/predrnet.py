from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_utils import Classifier, ResBlock, ConvNormAct, convert_to_rpm_matrix_v9, convert_to_rpm_matrix_v6

class PredictiveReasoningBlock(nn.Module):

    def __init__(self, in_planes, ou_planes, downsample, stride=1, dropout=0.0, num_contexts=8):

        super().__init__()

        md_planes = ou_planes*4
        self.pconv = ConvNormAct(in_planes, in_planes, (num_contexts, 1))
        self.conv1 = ConvNormAct(in_planes, md_planes, 3, 1)
        self.conv2 = ConvNormAct(md_planes, ou_planes, 3, 1)
        self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        
        b, c, t, l = x.size()
        contexts, choices = x[:,:,:t-1], x[:,:,t-1:]
        predictions = self.pconv(contexts)
        prediction_errors = F.relu(choices) - predictions
        
        out = torch.cat((contexts, prediction_errors), dim=2)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        identity = self.downsample(x)
        out = out + identity
        
        return out

class PredRNet(nn.Module):

    def __init__(self, num_filters=32, block_drop=0.0, classifier_drop=0.0, 
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8, 
                 num_extra_stages=1, reasoning_block=PredictiveReasoningBlock,
                 num_contexts=8, enable_rc=False):

        super().__init__()

        channels = [num_filters, num_filters*2, num_filters*3, num_filters*4]
        strides = [2, 2, 2, 2]

        # -------------------------------------------------------------------
        # frame encoder 
        self.in_planes = in_channels

        for l in range(len(strides)):
            setattr(
                self, "res"+str(l), 
                self._make_layer(
                    channels[l], stride=strides[l], 
                    block=ResBlock, dropout=block_drop,
                )
            )
        # -------------------------------------------------------------------


        # -------------------------------------------------------------------
        # predictive coding 
        self.num_extra_stages = num_extra_stages
        self.num_contexts = num_contexts
        self.in_planes = 32
        self.channel_reducer = ConvNormAct(channels[-1], self.in_planes, 1, 0, activate=False)    

        for l in range(num_extra_stages):
            setattr(
                self, "prb"+str(l), 
                self._make_layer(
                    self.in_planes, stride = 1, 
                    block = reasoning_block, 
                    dropout = block_drop
                )
            )
        # -------------------------------------------------------------------

        self.featr_dims = 1024

        self.classifier = Classifier(
            self.featr_dims, 1, 
            norm_layer = nn.BatchNorm1d, 
            dropout = classifier_drop, 
            hidreduce = classifier_hidreduce
        )

        self.in_channels = in_channels
        self.ou_channels = num_classes
        self.enable_rc = enable_rc

    def _make_layer(self, planes, stride, dropout, block, downsample=True):
        if downsample and block == ResBlock:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size = 2, stride = stride) if stride != 1 else nn.Identity(),
                ConvNormAct(self.in_planes, planes, 1, 0, activate = False, stride=1),
            )
        elif downsample and (block == PredictiveReasoningBlock or type(block) == partial):
            downsample = ConvNormAct(self.in_planes, planes, 1, 0, activate = False)
        else:
            downsample = nn.Identity()

        if block == PredictiveReasoningBlock or type(block) == partial:
            stage = block(self.in_planes, planes, downsample, stride = stride, 
                          dropout = dropout, num_contexts = self.num_contexts)
        elif block == ResBlock:
            stage = block(self.in_planes, planes, downsample, stride = stride, dropout = dropout)

        self.in_planes = planes

        return stage
    
    def extract_features(self, x):
        if self.in_channels == 1:
            b, n, h, w = x.size()
            x = x.reshape(b*n, 1, h, w)
        elif self.in_channels == 3:
            b, n, _, h, w = x.size()
            x = x.reshape(b*n, 3, h, w)

        for l in range(4):
            x = getattr(self, "res"+str(l))(x)

        x = self.channel_reducer(x)
        l = x.size(2)*x.size(3)
        return x.reshape(b, n, -1, l)
    
    def extract_relations(self, img_featrs):
        b, n, c, l = img_featrs.size()
        if self.num_contexts == 8:
            x = convert_to_rpm_matrix_v9(img_featrs, b, l, 1)
        else:
            x = convert_to_rpm_matrix_v6(img_featrs, b, l, 1)
        
        x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, l)
        # e.g. [b,9,c,l] -> [b,c,9,l] (l=h*w)
        x = x.permute(0,2,1,3)

        for l in range(0, self.num_extra_stages): 
            x = getattr(self, "prb"+str(l))(x)
        return x

    def forward(self, x):

        batches = x.size(0)

        img_featrs = self.extract_features(x)

        # normal row-wise relationships
        relations = self.extract_relations(img_featrs)     
        relations = relations.reshape(batches, self.ou_channels, -1)
        relations = F.adaptive_avg_pool1d(relations, self.featr_dims)    
        relations = relations.reshape(batches * self.ou_channels, self.featr_dims)

        scores = self.classifier(relations)

        if self.enable_rc and self.num_contexts == 8:
            col_featrs = img_featrs.clone()
            col_featrs[:,:8] = img_featrs[:,[0,3,6,1,4,7,2,5]]
            relations = self.extract_relations(col_featrs)
            relations = relations.reshape(batches, self.ou_channels, -1)
            relations = F.adaptive_avg_pool1d(relations, self.featr_dims)    
            relations = relations.reshape(batches * self.ou_channels, self.featr_dims)

            scores = scores + self.classifier(relations)

        return scores.view(batches, self.ou_channels)
    

def predrnet_raven(**kwargs):
    return PredRNet(**kwargs, num_contexts=8)


def predrnet_vad(**kwargs):
    return PredRNet(**kwargs, num_contexts=5, num_classes=4)