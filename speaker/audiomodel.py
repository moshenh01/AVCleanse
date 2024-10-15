import math, torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):  # A Squeeze-and-Excitation module that re-weights channel features to enhance useful ones.
    #  the number of channels is reduced to bottleneck, then expanded back to the original number of channels.
    # botteckneck is the number of channels after the first 1x1 convolution.
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()  # initialize the base class nn.Module
        self.se = nn.Sequential(  # Sequential container where the output of one layer is the input to the next layer.
            # squeeze
            nn.AdaptiveAvgPool1d(1),  # preform average pooling over the input, producing (batch - size, channels, 1)
            # This step stretch the spatial information into a single value per channel,
            # effectively "squeezing" the temporal context and providing a summary for each channel.

            # excitation
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            # out_channels, bottleneck = num of filters / output channels
            # channels, The number of input channels.
            # For example, if your input has multiple features, each feature is treated as a separate channel.
            # conv1d is a 1D convolutional layer,
            # which applies a 1D convolution over an input signal composed of several input planes.
            # Output: The output from nn.Conv1d has the shape [batch_size, out_channels, new_length],
            # where new_length depends on the kernel size, stride, padding, and dilation.
            nn.ReLU(),

            # restore the number of channels to the original number
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),  # The sigmoid function is used to scale the output of the SE block between 0 and 1.
            # when the output is 1, the channel is considered important,
            # and when the output is 0, the channel is considered unimportant.
        )

    def forward(self, input):  # input in shape [batch_size, channels, length]
        x = self.se(input)  # apply the SE block to the input,
        # where x is the output of the SE block,
        # in the same shape as the input.
        # but each channel is now weighted by a learned value between 0 and 1.
        return input * x  # same shape as the input tensor,
        # Element-wise multiplication of the input tensor with the weights x.
        # This means that each channel in the input tensor is scaled by the corresponding value from x,
        # which effectively emphasizes or de-emphasizes each channel.


class Bottle2neck(nn.Module):  # Res2Block

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        # inplanes: number of input channels, planes: number of output channels,
        # kernel_size: size of the convolutional kernel, dilation: dilation rate of the convolutional kernel,
        # scale: the number of parts into which the channels are divided.
        super(Bottle2neck, self).__init__()
        # This calculates the width of each channel group based on the total number of output
        # channels (planes) and the number of groups (scale).
        width = int(math.floor(planes / scale))
        # A 1x1 convolution to increase the number of channels from inplanes to width * scale.
        # This is like a channel expansion step, similar to the first convolution in the classic ResNet bottleneck.
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        # The number of subgroups that are processed with the convolution is scale - 1.
        # The remaining part is used as a skip connection. we can see it in Res2Net.
        self.nums = scale - 1

        # here we create "nums" convolutions each with input and output width of "width".
        # The convolutions use the given "kernel_size" and "dilation" for their **receptive field,
        # and "padding" ensures that the output length remains consistent.
        # The split feature maps are processed independently,
        # and the BatchNorm1d (bns) is applied after each convolution to improve stability during training.
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))

        self.convs = nn.ModuleList(convs)  # A ModuleList is a list of nn.Module objects.
        self.bns = nn.ModuleList(bns)

        # A 1x1 convolution is applied to reduce the number of channels back to planes.
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)  # BatchNorm1d is applied to the output of the last convolution.
        self.relu = nn.ReLU()
        self.width = width

        # The SEModule recalibrates the importance of each channel in the final output.
        # This module adds a self-attention mechanism to emphasize significant channels,
        # making it easier for the network to focus on useful features.
        self.se = SEModule(planes)

    def forward(self, x):  # x in shape [batch_size, inplanes, length]
        residual = x  # save the input for the skip connection

        # initial convolution
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        # split the output into "nums + 1" parts
        spx = torch.split(out, self.width, 1)
        # Each segment (spx[i]) is processed through the corresponding
        # convolution (convs[i]) and batch normalization (bns[i]).
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]  # add the previous segment to the current one

            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            # The processed segment is either stored as the first output (out = sp) or
            # concatenated with the previously processed segments (torch.cat((out, sp), 1)).
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        # The final split (spx[self.nums]) is concatenated at the end,
        # acting like a direct skip for that particular segment.
        out = torch.cat((out, spx[self.nums]), 1)

        # Final Convolution and SE
        out = self.conv3(out)  # The concatenated output goes through conv3 and bn3 layers, followed by ReLU.
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual  # add the input to the output
        return out


# PreEmphasis is a pre-processing filter commonly used in audio processing, especially in speech.
class PreEmphasis(torch.nn.Module):
    # It enhances higher frequencies by applying a simple filter defined by [-coef, 1].
    # This can make the important frequency content in speech more prominent
    # It is used at the beginning of the ECAPA_TDNN pipeline to enhance the input signal.
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


# FbankAug is used for data augmentation during training to improve
# generalization by randomly masking parts of the input.
class FbankAug(nn.Module):

    # It randomly masks frequencies (freq_mask_width) and time frames (time_mask_width) of the input spectrogram.
    # It is used in the main ECAPA_TDNN model before feature extraction layers,
    # to simulate noise and variability in data, which helps prevent overfitting.
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


# Main Model, Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network (ECAPA_TDNN)
class ECAPA_TDNN(nn.Module):

    # The constructor initializes various components and layers that form the building blocks of the model.
    def __init__(self, model='ecapa1024'):

        super(ECAPA_TDNN, self).__init__()
        # given a model name, the constructor sets the number of channels (C) accordingly.
        if model == 'ecapa1024':
            C = 1024
        elif model == 'ecapa512':
            C = 512
        else:
            quit()
        # Feature Extraction Layer
        self.torchfbank = torch.nn.Sequential(
            #  Applies a pre-emphasis filter to boost higher frequencies.
            #  This is a typical preprocessing step in speech analysis to highlight important features.
            PreEmphasis(),
            # MelSpectrogram: Converts the raw audio waveform into a Mel spectrogram,
            # which is a more suitable representation for audio tasks like speaker recognition.
            # Together, these transformations convert the audio input into a 2D time-frequency representation.
            # one axis represents time, and the other represents frequency.
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min=20, f_max=7600, window_fn=torch.hamming_window, n_mels=80),
        )
        # FbankAug: This is used for spec augmentation,
        # a type of data augmentation that involves masking parts of the spectrogram to simulate noise and variability,
        # improving the robustness of the model during training.
        self.specaug = FbankAug()  # Spec augmentation

        # initial convolutional layer that processes the input spectrogram.
        # conv1: The first convolutional layer, which has 80 input channels
        # (from the n_mels parameter of the Mel spectrogram) and C output channels.
        # It uses a kernel size of 5, with padding to ensure the output length matches the input length.
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)

        # Residual Bottleneck Layers
        # These layers help extract temporal features from the input data,
        # and the dilation allows for an expanded receptive field,
        # meaning that each layer can consider more context from the audio.
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)

        # Feature Aggregation Layer
        # layer4 aggregates the output of the previous three bottleneck layers (x1, x2, x3),
        # which are concatenated along the channel dimension
        # It uses a 1x1 convolution to transform the concatenated output into
        # a unified feature representation of 1536 channels.
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)

        # Attention Mechanism
        # This mechanism allows the model to determine which parts of the feature map are more important.

        self.attention = nn.Sequential(
            # The input has 4608 (3 Res2Block) channels, which is a concatenation of the original features (1536),
            # their mean, and standard deviation (each repeated across the sequence length).
            # The attention block processes these features to create a set of weights (w) that
            # represent the importance of different parts of the input over time.
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)  # projects the output to the final 192-dimensional speaker embedding.
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug=False):
        # Feature Extraction Part.

        # The input waveform x is first transformed into a Mel spectrogram and pre-emphasized using torchfbank.
        # A small value (1e-6) is added to prevent numerical issues when taking the log.
        # The mean is subtracted from the spectrogram, which is standard normalization,
        # ensuring that the features are centered around zero.
        with torch.no_grad():
            # The feature extraction part,
            # which involves creating a Mel spectrogram and applying some pre-processing transformations,
            # is not part of the model's trainable parameters.
            # Therefore, there is no need to compute or store gradients for this part of the computation,
            # which can save memory and speed up computations.
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x) + 1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                # if aug is True, the input is passed through the SpecAugment module to simulate noise and variability.
                if aug == True:
                    x = self.mask(x, max_fmask=0.1, max_tmask=0.1)

        # Initial Convolution:
        # The spectrogram is passed through conv1, followed by ReLU activation and BatchNorm1d.
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        # Residual Bottleneck Layers:
        # The output passes through three Bottle2neck layers (layer1, layer2, layer3),
        # each receiving the sum of the previous layers' outputs as input.
        # This combination helps capture features at different scales and build progressively richer representations.
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        # Feature Aggregation and Attention:
        # The features from x1, x2, and x3 are concatenated and then transformed by layer4.
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]  # get the length of the sequence, this is the time dimension.

        # The "mean" and "standard deviation" are also concatenated with the output, creating a global feature map.
        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        # The attention mechanism (self.attention) generates attention weights (w) for different time steps.
        w = self.attention(global_x)

        # Pooling and Final Output:
        # The attention weights (w) are used to pool the features across time.
        # he mean (mu) and standard deviation (sg) are computed for each channel using the weighted features.
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

        # The pooled features (mu and sg) are concatenated and passed through a final set of fully connected
        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

    # The mask function is a method that applies time masking and frequency masking to the input tensor x.
    # This technique is used to augment training data in audio models, inspired by SpecAugment,
    # which is a data augmentation strategy that involves masking certain regions in the
    # spectrogram to make the model more robust to variations in the input.
    def mask(self, x, max_fmask, max_tmask):
        b, f, t = x.shape
        orig_shape = x.shape
        if max_tmask != 0:
            max_tmask = int(max_tmask * t)
            tmask_len = torch.randint(0, max_tmask, (b, 1)).unsqueeze(2).cuda()
            tmask_pos = torch.randint(0, max(1, t - tmask_len.max()), (b, 1)).unsqueeze(2).cuda()
            arange = torch.arange(t).view(1, 1, -1).cuda()
            mask = (tmask_pos <= arange) * (arange < (tmask_pos + tmask_len))
            mask = mask.any(dim=1)
            mask = mask.unsqueeze(1)
            x = x.masked_fill_(mask, 0.0)
            x = x.view(*orig_shape)
        if max_fmask != 0:
            max_fmask = int(max_fmask * f)
            fmask_len = torch.randint(0, max_fmask, (b, 1)).unsqueeze(2).cuda()
            fmask_pos = torch.randint(0, max(1, f - fmask_len.max()), (b, 1)).unsqueeze(2).cuda()
            arange = torch.arange(f).view(1, 1, -1).cuda()
            mask = (fmask_pos <= arange) * (arange < (fmask_pos + fmask_len))
            mask = mask.any(dim=1)
            mask = mask.unsqueeze(2)
            x = x.masked_fill_(mask, 0.0)
            x = x.view(*orig_shape)
        return x
