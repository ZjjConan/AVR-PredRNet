from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_utils import ResBlock, ConvNormAct

def construct_ul_matrix(ctx, cho, n_ctx):
    # b, n = ctx.shape[0:2]
    if n_ctx == 5:
        matrix = ctx[:, :6].clone().unsqueeze(1).repeat(1,9,1,1,1)
        # [b, 9, 6]
        matrix[:, 1:9, -1] = cho
    elif n_ctx == 2:
        matrix = ctx[:, :3].clone().unsqueeze(1).repeat(1,5,1,1,1)
        # [b, 4, 3]
        matrix[:, 1:5, -1] = cho
    return matrix


def construct_sl_matrix(ctx, cho, n_ctx):
    if n_ctx == 5:
        row13 = ctx[:, [0,1,2,6,7]]
        row23 = ctx[:, [3,4,5,6,7]]
        row13 = torch.stack([torch.cat([row13, cho[:,i].unsqueeze(1)], dim=1) for i in range(0, 8)], dim=1)
        row23 = torch.stack([torch.cat([row23, cho[:,i].unsqueeze(1)], dim=1) for i in range(0, 8)], dim=1)
        return row13, row23
    
    elif n_ctx == 2:
        row2 = ctx[:, [3,4]]
        row2 = torch.stack([torch.cat([row2, cho[:,i].unsqueeze(1)], dim=1) for i in range(0, 4)], dim=1)
        return row2, None

class SSPredictiveReasoningBlock(nn.Module):

    def __init__(self, in_planes, ou_planes, downsample, stride=1, dropout=0.0, num_contexts=5, last_block=False):

        super().__init__()

        self.stride = stride
        self.last_block = last_block

        md_planes = ou_planes*4
        self.pconv = ConvNormAct(in_planes, in_planes, (num_contexts, 1))
        if not last_block:
            self.conv1 = ConvNormAct(in_planes, md_planes, 3, 1)
            self.conv2 = ConvNormAct(md_planes, ou_planes, 3, 1)
            self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()
            self.downsample = downsample

    def forward(self, x):
        b, c, t, l = x.size()
        contexts, choices = x[:,:,:t-1], x[:,:,t-1:]
        predictions = self.pconv(contexts)
        prediction_errors = F.relu(choices) - predictions

        choices = F.relu(choices).mean(dim=[2,3])
        predictions = predictions.mean(dim=[2,3])
        output_errors = F.normalize(choices, dim=1) - F.normalize(predictions, dim=1)

        if self.last_block:
            return None, output_errors.pow(2).sum(dim=1).sqrt()
        else:
            out = torch.cat((contexts, prediction_errors), dim=2)
            out = self.conv1(out)
            out = self.conv2(out)
            out = self.drop(out)
            identity = self.downsample(x)
            out = out + identity
            return out, output_errors.pow(2).sum(dim=1).sqrt()

class UnPredRNet(nn.Module):

    def __init__(self, num_filters=32, block_drop=0.0, classifier_drop=0.0, 
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8, 
                 num_extra_stages=1, reasoning_block=SSPredictiveReasoningBlock,
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
            if l == num_extra_stages - 1:
                setattr(
                    self, "prb"+str(l), 
                    self._make_layer(
                        self.in_planes, stride = 1, 
                        block = reasoning_block, 
                        dropout = block_drop,
                        last_block=True
                    )
                )
            else:
                setattr(
                    self, "prb"+str(l), 
                    self._make_layer(
                        self.in_planes, stride = 1, 
                        block = reasoning_block, 
                        dropout = block_drop,
                        last_block=False
                    )
                )
        # -------------------------------------------------------------------
        self.enable_rc = enable_rc
        self.in_channels = in_channels
        self.ou_channels = num_classes

    def _make_layer(self, planes, stride, dropout, block, downsample=True, last_block=False):
        if downsample and block == ResBlock:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size = 2, stride = stride) if stride != 1 else nn.Identity(),
                ConvNormAct(self.in_planes, planes, 1, 0, activate = False, stride=1),
            )
        elif downsample and (block == SSPredictiveReasoningBlock or type(block) == partial):
            downsample = ConvNormAct(self.in_planes, planes, 1, 0, activate = False)
        else:
            downsample = nn.Identity()

        if block == SSPredictiveReasoningBlock or type(block) == partial:
            stage = block(self.in_planes, planes, downsample, stride = stride, 
                          dropout = dropout, num_contexts = self.num_contexts, 
                          last_block = last_block)
        elif block == ResBlock:
            stage = block(self.in_planes, planes, downsample, stride = stride, dropout = dropout)

        self.in_planes = planes

        return stage
    
    def _forward_and_return_errors(self, x, batches, num_matrix):

        # e.g. [b,n_ctx,c,l] -> [b,c,n_ctx,l] (l=h*w)
        x = x.permute(0,2,1,3)
        out_err = []
        for l in range(0, self.num_extra_stages): 
            x = getattr(self, "prb"+str(l))(x)
            if l > 0:
                out_err.append(x[1])
            x = x[0]
        out_err = [o.view(batches, num_matrix) for o in out_err]

        return out_err
    
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
        x = x.reshape(b, n, x.shape[1], l)
        return x

    def extract_relations_for_train(self, ctx, cho):
        b, n, c, l = ctx.size()
        rpms = construct_ul_matrix(ctx, cho, self.num_contexts)
        nums = rpms.shape[1]
        errs = self._forward_and_return_errors(
            rpms.reshape(-1, self.num_contexts+1, c, l), 
            b,
            nums
        )
        return errs
    
    def extract_relations_for_test(self, ctx, cho):
        b, n, c, l = ctx.size()
        x13, x23 = construct_sl_matrix(ctx, cho, self.num_contexts)
        nums = x13.shape[1]
        scores13 = self._forward_and_return_errors(
            x13.reshape(-1, self.num_contexts+1, c, l), 
            b,
            nums
        )
        if x23 is not None:
            scores23 = self._forward_and_return_errors(
                x23.reshape(-1, self.num_contexts+1, c, l), 
                b,
                nums
            )
            scores = [s13 + s23 for s13, s23 in zip(scores13, scores23)]
        else:
            scores = scores13
        return scores

    def forward(self, x):

        img_featrs = self.extract_features(x)
        b, n, c, l = img_featrs.size() 

        if self.num_contexts == 5:
            ctx_featrs, cho_featrs = img_featrs[:, :8], img_featrs[:, 8:]
        elif self.num_contexts == 2:
            ctx_featrs, cho_featrs = img_featrs[:, :5], img_featrs[:, 5:]
        
        if self.training:
            errors = self.extract_relations_for_train(ctx_featrs, cho_featrs)
            if self.enable_rc and self.num_contexts == 5:
                col_errors = self.extract_relations_for_train(
                    ctx_featrs[:, [0,3,6,1,4,7,2,5]], cho_featrs
                )
                errors = [r + c for r, c in zip(errors, col_errors)]
            labels = torch.zeros(b, dtype=torch.long).cuda()
            return errors, labels
        else:
            scores = self.extract_relations_for_test(ctx_featrs, cho_featrs)
            if self.enable_rc and self.num_contexts == 5:
                scores_col = self.extract_relations_for_test(
                    ctx_featrs[:, [0,3,6,1,4,7,2,5]], cho_featrs
                )
                scores = [sr + sc for sr, sc in zip(scores, scores_col)]
            return scores


def sspredrnet_raven(**kwargs):
    return UnPredRNet(**kwargs, num_contexts=5)

def sspredrnet_vad(**kwargs):
    return UnPredRNet(**kwargs, num_contexts=2)