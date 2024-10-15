'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *


# Additive Angular Margin Softmax Loss
class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s, c):
        # n_class: number of classes
        # m: margin
        # s: scale
        # c: The size of the feature vector for each input sample.
        super(AAMsoftmax, self).__init__()
        self.m = m # margin
        self.s = s # scale

        # This is a parameter matrix of shape (n_class, c) which stores class weights vectors.
        # These weights represent the prototype (the weight vector that represent each class) vectors of each class.
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, c), requires_grad=True)        
        self.ce = nn.CrossEntropyLoss(reduction = 'mean')
        # Initialize the weights using the Xavier normal initialization.
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)  # cosine of the margin
        self.sin_m = math.sin(self.m)

        # This is the threshold value of the cosine value after which the margin is applied.
        # Essentially, this threshold is used to ensure that we only apply the additive margin adjustment (phi)
        # when the angle is within a certain range (i.e., when the cosine value is sufficiently positive).
        # If the cosine value is less than this threshold, it indicates that the feature and the weight vectors
        # are not sufficiently aligned, and applying the angular margin directly could lead to instability or incorrect gradient updates.
        self.th = math.cos(math.pi - self.m)

        # represents an offset value, This is used as a fallback adjustment to the cosine value when
        # the cosine value falls below self.th.
        # The purpose of self.mm is to provide a modified cosine value that is still meaningful for the
        # learning process but avoids the issues that arise from directly using phi when the cosine is too low.
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        # x: A batch of feature vectors with shape (batch_size, c)
        # label: A batch of labels with shape (batch_size)

        # Normalize the input feature vectors and the weight vectors.
        # and compute the cosine similarity between the input feature vectors and the weight vectors.
        # This results in a matrix of cosine similarities between each input and all the class weight vectors,
        # which has a shape of (batch_size, n_class).
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        # This calculates the sine value from the cosine using the trigonometric identity,
        # sin(a) = sqrt(1 - cos^2(a)).
        # The result is a matrix of sine values with the same shape as the cosine matrix.
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))

        # Apply Margin (Additive Angular Margin):
        # This calculates the new cosine value by adding an angular margin "m".
        # using the trigonometric identity, cos(a + m) = cos(a)cos(m) - sin(a)sin(m).
        # the result is a matrix of cosine values with the same shape as the cosine matrix.
        phi = cosine * self.cos_m - sine * self.sin_m

        # Conditionally Adjusting Phi:
        # This conditional adjustment ensures numerical stability.
        # When the cosine value is less than a threshold (self.th),
        # phi is adjusted to prevent numerical instability or ensure correct angular margin enforcement.
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # One-hot encoding of the labels
        one_hot = torch.zeros_like(cosine)  # create a tensor of zeros with the same shape as the cosine matrix
        one_hot.scatter_(1, label.view(-1, 1), 1)  # set the value at the index of the label to 1 in the one_hot tensor.

        # The phi value (with margin) is assigned only to the target classes,
        # while the cosine values for non-target classes are left unchanged.
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the output by a factor "s".
        # Suppose s = 30, which is a typical value used in practice.
        # The logits are now scaled up, so they lie in a higher range (e.g.,[−30,30]).
        # When these scaled values are passed through the softmax,
        # the resulting probabilities will be sharper, meaning the model will be more confident in its predictions.
        # This helps the model learn more discriminative features and improve classification performance.
        output = output * self.s

        # Compute the cross-entropy loss between the output and the labels.
        # by using the mathematical formula:
        # L = -log(exp(output[label]) / sum(exp(output)))
        loss = self.ce(output, label)

        # The accuracy is computed using a helper function named accuracy
        # (not defined here, but typically this function calculates the top-k accuracy).
        prec1, correct = accuracy(output.detach(), label.detach(), topk=(1,))

        return loss, prec1
