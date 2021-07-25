import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

    def transmitting_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp

    def top_eigenvalue(self, K, n_power_iterations=5, dim=1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            v = torch.ones(K.shape[0], K.shape[1], 1).to(device)
            for _ in range(n_power_iterations):
                m = torch.bmm(K, v)
                n = torch.norm(m, dim=1).unsqueeze(1)
                v = m / n

            value = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
        return value

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x, preReLU=True)
        s_feats, s_out = self.s_net.extract_feature(x, preReLU=True)
        feat_num = len(t_feats)

        loss_distill = 0
        loss_largest_eigenvalue = 0
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                            / 2 ** (feat_num - i - 1)

            if i < feat_num - 1:
                # print(s_feats[i].shape, s_feats[i + 1].shape)
                TM_s = torch.bmm(self.transmitting_matrix(s_feats[i], s_feats[i + 1]),self.transmitting_matrix(s_feats[i], s_feats[i + 1]).transpose(2,1)) #for calculate the eigenvalue of fsp matrix, because fsp matrix is asymmetric
                TM_t = torch.bmm(self.transmitting_matrix(t_feats[i].detach(), t_feats[i + 1].detach()),self.transmitting_matrix(t_feats[i].detach(), t_feats[i + 1].detach()).transpose(2,1))

                loss_largest_eigenvalue += F.mse_loss(self.top_eigenvalue(K=TM_s), self.top_eigenvalue(K=TM_t)) / 2 ** (feat_num - i - 1)

        return s_out, loss_distill, loss_largest_eigenvalue
