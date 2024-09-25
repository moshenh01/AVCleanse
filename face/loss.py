'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

# The Additive Angular Margin Softmax (AAM-Softmax) loss is a variant of softmax used for face recognition or similar tasks.
# It introduces a margin between different classes in angular space to improve the discriminability of embeddings.
class AAMsoftmax(nn.Module):
    # initializes the AAMsoftmax loss function
    def __init__(self, n_class, m, s, c):   
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, c), requires_grad=True)        
        self.ce = nn.CrossEntropyLoss(reduction = 'mean')
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        
    # forward pass of the AAMsoftmax loss function
    # Cosine Similarity - Computes the cosine similarity between the input and weight tensors
    # Margin - Adds an angular margin to make the classifier more discriminative
    # Scale - Scales the output by a factor to make the classifier more robust
    # Cross-Entropy Loss - Computes the cross-entropy loss between the output and label tensors
    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))             # it normalizes the input and weight tensors and computes the cosine similarity between them
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))        # it computes the sine of the angle between the input and weight tensors
        phi = cosine * self.cos_m - sine * self.sin_m                           # it computes the cosine of the angle between the input and weight tensors with the margin
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)        # it applies the margin to the cosine of the angle between the input and weight tensors
        one_hot = torch.zeros_like(cosine)                                      # it creates a one-hot tensor with the same shape as the cosine tensor
        one_hot.scatter_(1, label.view(-1, 1), 1)                               # it scatters the one-hot tensor with the label                
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)                   # it computes the output by applying the margin to the cosine of the angle between the input and weight tensors
        output = output * self.s                                                # it scales the output by the scale parameter                       

        loss = self.ce(output, label)                                           # it computes the cross-entropy loss between the output and label tensors
        prec1, correct = accuracy(output.detach(), label.detach(), topk=(1,))   # it computes the top-1 accuracy between the output and label tensors

        return loss, prec1
