# %% [markdown]
# # Neural network models

# %%
import torch
import torch.nn as nn
import math
from torchvision import models
from collections import OrderedDict
from copy import deepcopy

from pvtv2 import *
import torch.nn.functional as F

# %% [markdown]
# ## U-Net Network
# num_classes = Number of classes expecteed from output

# %%
def conv2d_layer(channels_in, channels_out, kernel_size, padding, stride, dilation):
    layer = nn.ModuleList()

    conv2d = nn.Conv2d(
        in_channels = channels_in, 
        out_channels = channels_out, 
        kernel_size = kernel_size, 
        stride = stride, 
        padding = padding, 
        dilation = dilation
    )
    torch.nn.init.kaiming_normal_(conv2d.weight)
    
    layer.append(conv2d)
    layer.append(nn.ReLU())

    return layer

# %%
class Encoder_Block(nn.Module):
    def __init__(self, channels_in, channels_out, params: dict):
        super().__init__()

        # Config parameters
        try:
            block_width = params['block_width']
            kernel_size = params['kernel_size']
            padding = params['padding']
            stride = params['stride']
            dilation = params['dilation']
            pool_type = params['pool_type']
            pool_size = params['pool_kernel_size']
        except KeyError as e:
            raise Exception(f'Parameter "{e.args[0]}" NOT found!')
        
        # Add first Conv2D layer
        self.layers = nn.ModuleList()
        self.layers.extend(conv2d_layer(channels_in, channels_out, kernel_size, padding, stride, dilation))

        # Add other Conv2D layers
        for _ in range(block_width - 1):
            self.layers.extend(conv2d_layer(channels_out, channels_out, kernel_size, padding, stride, dilation))

        # Add pooling layer
        match pool_type:
            case 'max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
            case 'avg':
                self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)


    def forward_noSkip(self, x):
        # Convolution
        for layer in self.layers:
            x = layer(x)

        # Pooling
        pooled = self.pooling(x)
        
        return pooled
    
    def forward_skip(self, x):
        # Convolution
        for layer in self.layers:
            x = layer(x)

        # Pooling
        pooled = self.pooling(x)

        return (pooled, x)

# %%
class Decoder_Block(nn.Module):
    def __init__(self, channels_in, channels_out, padding_t, params: dict):
        super().__init__()

        # Config parameters
        try:
            block_width = params['block_width']
            kernel_size = params['kernel_size']
            padding = params['padding']
            stride = params['stride']
            dilation = params['dilation']
            pool_size = params['pool_kernel_size']
            skip_features = params['skip_features']
        except KeyError as e:
            raise Exception(f'Parameter "{e.args[0]}" NOT found!')

        # Add transpose convolution
        channel_out_convT = math.floor(channels_in / 2)

        self.convT = nn.ConvTranspose2d (
            in_channels = channels_in, 
            out_channels = channel_out_convT, 
            kernel_size = pool_size, 
            stride = pool_size, 
            output_padding = padding_t
        )
        
        match skip_features:
            case "none":
                channels_in = channel_out_convT
                
            case "concat":
                # Half of channels are from convT, other half from enc features
                # channels_in = 2 * channel_out_convT
                pass

        # Add first Conv2D layer
        self.layers = nn.ModuleList()
        self.layers.extend(conv2d_layer(channels_in, channels_out, kernel_size, padding, stride, dilation))

        # Add other Conv2D layers
        for _ in range(block_width - 1):
            self.layers.extend(conv2d_layer(channels_out, channels_out, kernel_size, padding, stride, dilation))

    def forward_noSkip(self, x):
        # Transposed convolution
        x = self.convT(x)

        # Convolution
        for layer in self.layers:
            x = layer(x)

        return x

    def forward_skip(self, x, enc_feature):
        # Transposed convolution
        x = self.convT(x)

        # Copy features
        x = torch.cat([enc_feature, x], dim = 1)

        # Convolution
        for layer in self.layers:
            x = layer(x)

        return x

# %%
class U_Net(nn.Module):
    def __init__(self, channels_in, num_classes, params: dict):
        super().__init__()

        torch.set_default_dtype(torch.float32)

        # Config parameters
        try:
            channel_mul = params['channel_mul']
            depth = params['network_depth']
            channels_out = params['channels_out_init']
            padding_convT = deepcopy(params['padding_convT'])
            self.skip_features = params['skip_features']
            if (len(padding_convT) != depth):
                raise Exception(f"Padding of Transposed convolution has length {len(padding_convT)}, but should be {depth}!")
            
        except KeyError as e:
            raise Exception(f'Parameter "{e.args[0]}" NOT found!')
        
        # Create encoder
        self.encoders = nn.ModuleList()

        for _ in range(depth):
            self.encoders.append(Encoder_Block(channels_in, channels_out, params))
            channels_in = channels_out
            channels_out = math.floor(channels_out * channel_mul)

        # Create bridge as Conv2D layer
        last_encoder = Encoder_Block(channels_in, channels_out, params)
        self.bridge = last_encoder.layers[0]

        channels_in = channels_out
        channels_out = math.floor(channels_in / channel_mul)

        # Create decoder
        self.decoders = nn.ModuleList()
        
        for _ in range(depth) :
            paddingT = padding_convT.pop(0)
            self.decoders.append(Decoder_Block(channels_in, channels_out, paddingT, params))
            channels_in = channels_out
            channels_out = math.floor(channels_out / channel_mul)

        # Create output layer (1x1 convolution, return logits)
        self.output_layer = nn.Conv2d(
            in_channels = channels_in, 
            out_channels = num_classes, 
            kernel_size = 1
        )

        match self.skip_features:
            case "none":
                print("This U-Net does NOT skip any connections.")
            case "concat":
                print("This U-Net skips connections using concatenation.")

    def forward(self, x):
        match self.skip_features:
            # No skipping
            case "none":
                # Encoding
                for enc in self.encoders:
                    x = enc.forward_noSkip(x)

                # Bridge between encoder and decoder
                x = self.bridge(x)

                # Decoding
                for dec in self.decoders:
                    x = dec.forward_noSkip(x)

            # Cat tensors => features + pooled output
            case "concat":
                # Features created during enconding
                features = []

                # Encoding
                for enc in self.encoders:
                    x, enc_feature = enc.forward_skip(x)
                    features.append(enc_feature)

                # Bridge between encoder and decoder
                x = self.bridge(x)

                # Decoding
                for dec in self.decoders:
                    enc_output = features.pop()
                    x = dec.forward_skip(x, enc_output)  

        # Output as logits
        x = self.output_layer(x)

        return x

# %%
def conv2d_output_shape(height, width, conv2d: nn.Module):
    height = ((height + (2 * conv2d.padding[0]) - (conv2d.dilation[0] * (conv2d.kernel_size[0] - 1)) - 1) / conv2d.stride[0]) + 1
    height = math.floor(height)
    
    width = ((width + (2 * conv2d.padding[1]) - (conv2d.dilation[1] * (conv2d.kernel_size[1] - 1)) - 1) / conv2d.stride[1]) + 1
    width = math.floor(width)

    return (height, width, conv2d.out_channels)

def pool_output_shape(height, width, conv2d: nn.Module):
    height = ((height + (2 * conv2d.padding) - (conv2d.dilation * (conv2d.kernel_size - 1)) - 1) / conv2d.stride) + 1
    height = math.floor(height)
    
    width = ((width + (2 * conv2d.padding) - (conv2d.dilation * (conv2d.kernel_size - 1)) - 1) / conv2d.stride) + 1
    width = math.floor(width)

    return (height, width)

def convT_output_shape(height, width, convT: nn.Module):
    height = (height - 1) * convT.stride[0] - (2 * convT.padding[0]) + (convT.dilation[0] * (convT.kernel_size[0] - 1)) + convT.output_padding[0] + 1
    
    width = (width - 1) * convT.stride[1] - (2 * convT.padding[1]) + (convT.dilation[1] * (convT.kernel_size[1] - 1)) + convT.output_padding[1] + 1
    
    return (height, width, convT.out_channels)


def output_shapes(net : U_Net, height, width):
    h = height
    w = width
    depth = len(net.encoders) + 1

    print("----------------------------------------------")
    print(f"CNN with depth = {depth}")
    print("----------------------------------------------")

    i = 1
    for enc in net.encoders:
        print(f"||Layer {i} - encoders||")

        for l in enc.layers:
            if not isinstance(l, nn.ReLU):
                (h,w,c) = conv2d_output_shape(h, w, l)
                print (f"Conv2D -- H = {h} | W = {w} | C = {c}")

        (h,w) = pool_output_shape(h, w, enc.pooling)
        print (f"Pooling -- H = {h} | W = {w} | C = {c}") 

        print()
        i += 1
            
    print(f"||Layer {i} - bridge||")
    (h,w,c) = conv2d_output_shape(h, w, net.bridge)
    print (f"H = {h} | W = {w} | C = {c}")
    print("----------------------------------------------")

    for dec in net.decoders:
        print(f"||Layer {i} - decoders||")

        (h,w,c) = convT_output_shape(h, w, dec.convT)
        print (f"Transpose -- H = {h} | W = {w} | C = {c}")   

        for l in dec.layers:
            if not isinstance(l, nn.ReLU):
                (h,w,c) = conv2d_output_shape(h, w, l)
                print (f"Conv2D -- H = {h} | W = {w} | C = {c}")   

        print()
        i -= 1


    print(f"||Layer {i} - output||")
    (h,w,c) = conv2d_output_shape(h, w, net.output_layer)
    print (f"H = {h} | W = {w} | C = {c}")
    print("----------------------------------------------")

# %%
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        middle_channels = math.floor(in_planes // ratio)

        self.fc1 = nn.Conv2d(in_planes, middle_channels, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(middle_channels, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans
    
def pad_dimensions(desired: torch.tensor, wrong: torch.tensor):
    if desired.ndimension() != wrong.ndimension():
        return wrong
    
    padding = ()
    
    for i in reversed(range(desired.ndimension())):
        diff = desired.size(i) - wrong.size(i)
        left = math.floor(diff / 2)
        right = math.ceil(diff / 2)
        padding += (left, right)

    return F.pad(wrong, padding, "constant", 0)
    

class UNetV2(nn.Module):
    def __init__(self, n_classes, params: dict):
        super().__init__()
        # Config parameters
        try:
            channels = params['SDI_channels']
            ratio_attention = params['channel_att_ratio']
        except KeyError as e:
            raise Exception(f'Parameter "{e.args[0]}" NOT found!')
        
        self.deep_supervision = False

        self.encoder = pvt_v2_b2()

        self.ca_1 = ChannelAttention(64, ratio_attention)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(128, ratio_attention)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(320, ratio_attention)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(512, ratio_attention)
        self.sa_4 = SpatialAttention()

        self.Translayer_1 = BasicConv2d(64, channels, 1)
        self.Translayer_2 = BasicConv2d(128, channels, 1)
        self.Translayer_3 = BasicConv2d(320, channels, 1)
        self.Translayer_4 = BasicConv2d(512, channels, 1)

        self.sdi_1 = SDI(channels)
        self.sdi_2 = SDI(channels)
        self.sdi_3 = SDI(channels)
        self.sdi_4 = SDI(channels)

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(channels, n_classes, 1, 1) for _ in range(4)])

        self.deconv2 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1,
                                          bias=False)
        self.deconv3 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2,
                                          padding=1, bias=False)

    def forward(self, x):
        seg_outs = []
        f1, f2, f3, f4 = self.encoder(x) 

        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)

        f41 = self.sdi_4([f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f1, f2, f3, f4], f2)
        f11 = self.sdi_1([f1, f2, f3, f4], f1)

        seg_outs.append(self.seg_outs[0](f41))

        y = self.deconv2(f41)
        y = pad_dimensions(f31, y)
        y += f31
        seg_outs.append(self.seg_outs[1](y))


        y = self.deconv3(y)
        y = pad_dimensions(f21, y)
        y += f21
        seg_outs.append(self.seg_outs[2](y))


        y = self.deconv4(y)
        y = pad_dimensions(f11, y)
        y += f11
        seg_outs.append(self.seg_outs[3](y))

        for i, o in enumerate(seg_outs):
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear')

        if self.deep_supervision:
            return seg_outs[::-1]
        else:
            return seg_outs[-1]


