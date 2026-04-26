import torch
from torch import nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model)

    def split_head(self, x):
        bs, seql, dm = x.shape
        return x.view(bs, seql, self.n_head, self.d_head).transpose(1, 2) # bs nh L dh

    def forward(self, x, atten_mask=None):
        bs, seql, dm = x.shape
        q = self.split_head(self.Q(x))
        kT = self.split_head(self.K(x)).transpose(2, 3) # bs bh dh L
        v = self.split_head(self.V(x))

        atten_score = torch.matmul(q, kT) / torch.sqrt(self.d_head)
        if atten_mask:
            atten_score.masked_fill(atten_mask==0, torch.float(-inf))
        atten_score = F.softmax(atten_score, dim=-1)
        o = torch.matmul(atten_score, v) # bs nh L dh
        o = o.transpose(1, 2)
        return o.view(bs, seql, dm)