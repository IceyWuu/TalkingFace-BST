from torch.nn import functional as F
import torch
import torch.nn as nn
import torchvision
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math


## 同landmark generator
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# class AdaINLayer(nn.Module):# 256 or 64,   512
#     def __init__(self, input_nc, modulation_nc,
#                  T=1,d_model=256,nlayers=3,nhead=1,dim_feedforward=512,dropout=0.1):
#         super().__init__()

#         # self.InstanceNorm2d = nn.InstanceNorm2d(input_nc, affine=False)

#         #nhidden = 128
#         nhidden = input_nc
#         use_bias=True

#         self.mlp_shared = nn.Sequential(
#             nn.Linear(modulation_nc, nhidden, bias=use_bias),
#             nn.ReLU()
#         )
#         # self.mlp_gamma = nn.Linear(nhidden, input_nc, bias=use_bias)
#         # self.mlp_beta = nn.Linear(nhidden, input_nc, bias=use_bias)

#         encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, int(nlayers)) #!!说是float不能看作是整数
#         decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True) 
#         self.transformer_decoder = TransformerDecoder(decoder_layers, int(nlayers))
#         self.Norm_1=nn.LayerNorm(d_model) # normalization layer
#         self.Norm_2=nn.LayerNorm(d_model) # normalization layer
#         self.position_in = PositionalEmbedding(d_model=d_model,max_len=d_model)  #for input image
#         self.position_au = PositionalEmbedding(d_model=d_model,max_len=d_model)  #for input sketch
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, input, modulation_input):
#         # (B*1,256,64,64), (B*1,1,512)
#         #  (B*1,64,128,128),(B*1,1,512)

#         # Part 1. generate parameter-free normalized activations
#         # normalized = self.InstanceNorm2d(input)

#         # Part 2. produce scaling and bias conditioned on feature
#         modulation_input = modulation_input.view(modulation_input.size(0), -1) # (B*1,512)
#         # actv = self.mlp_shared(modulation_input) # (B*1,128)
#         audio_feature = self.mlp_shared(modulation_input) # (B*1,256)
#         # gamma = self.mlp_gamma(actv) # gamma (B, 256)
#         # beta = self.mlp_beta(actv)
# # 
#         # apply scale and bias
#         # gamma = gamma.view(*gamma.size()[:2], 1,1) # *gamma.size()[:2]返回的是gamma前两个维度的大小
#         # beta = beta.view(*beta.size()[:2], 1,1) # 得到的是(B*1, 256, 1, 1)，为了后面能进行广播用的
#         # out = normalized * (1 + gamma) + beta   # 用广播机制进行运算
#         input = input.flatten(2).permute(0,2,1) # (B*1, 4096, 256)
#         # !!!!audio_feature 应该手动广播成(B*1, 4096, 256)，这里还没换！！
#         audio_feature = audio_feature.expand(-1, -1, input.size(-1)) # (B*1, 256, 4096) 手动广播

#         #normalization
#         input_embedding = self.Norm_1(input) # (B*1, 4096, 256) 
#         audio_embedding = self.Norm_2(audio_feature) # (B*1, 256, 4096)  

#         # (1).  positional(temporal) encoding
#         position_in_encoding = self.position_in(input_embedding)
#         position_au_encoding = self.position_au(audio_embedding)

#         #(2)  modality encoding

#         input_tokens = input_embedding + position_in_encoding # + modality_v    
#         audio_tokens = audio_embedding + position_au_encoding # + modality_a   

#         input_tokens = self.dropout(input_tokens)
#         audio_tokens = self.dropout(audio_tokens) 

#         #(4) input to transformer # 编、解码器都是nlayers=3层,nhead=1
#         output_encoder = self.transformer_encoder(input_tokens)
#         output_decoder = self.transformer_decoder(audio_tokens, output_encoder) # output_encoder作为kv
#         # q, kv
#         # 原来的形状 # (B*1, 256, 64 * 64) 
#         output_decoder = output_decoder.reshape(-1, self.input_channel, output_decoder.shape(-1)** 0.5, output_decoder.shape(-1) ** 0.5)
#         return output_decoder

#         # return out

# class AdaIN(torch.nn.Module):

#     def __init__(self, input_channel, modulation_channel,kernel_size=3, stride=1, padding=1,
#                  T=1,d_model=256,nlayers=3,nhead=1,dim_feedforward=512,dropout=0.1):
#         super(AdaIN, self).__init__()
#         self.conv_1 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.conv_2 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.leaky_relu = torch.nn.LeakyReLU(0.2)
#         self.adain_layer_1 = AdaINLayer(input_channel, modulation_channel,
#                                         T=1,d_model=256,nlayers=3,nhead=1,dim_feedforward=512,dropout=0.1) #AdaIN(256,512)
#         self.adain_layer_2 = AdaINLayer(input_channel, modulation_channel,
#                                         T=1,d_model=64,nlayers=3,nhead=1,dim_feedforward=128,dropout=0.1) #AdaIN(64,512)

#     def forward(self, x, modulation): 
#         #  (B*1,256,64,64), (B*1,1,512)
#         #  (B*1,64,128,128),(B*1,1,512)

#         x = self.adain_layer_1(x, modulation)
#         x = self.leaky_relu(x)
#         x = self.conv_1(x)
#         x = self.adain_layer_2(x, modulation)
#         x = self.leaky_relu(x)
#         x = self.conv_2(x)

#         return x

class AdaINLayer(nn.Module):# 256 or 64,   512
    def __init__(self, input_nc, modulation_nc):
        super().__init__()

        self.InstanceNorm2d = nn.InstanceNorm2d(input_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(modulation_nc, nhidden, bias=use_bias),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, input_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(nhidden, input_nc, bias=use_bias)

    def forward(self, input, modulation_input):
        # (B*1,256,64,64), (B*1,1,512)
        #  (B*1,64,128,128),(B*1,1,512)

        # Part 1. generate parameter-free normalized activations
        normalized = self.InstanceNorm2d(input)

        # Part 2. produce scaling and bias conditioned on feature
        modulation_input = modulation_input.view(modulation_input.size(0), -1) # (B*1,512)
        actv = self.mlp_shared(modulation_input) # (B*1,128)
        gamma = self.mlp_gamma(actv) # gamma (B, 256)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1) # *gamma.size()[:2]返回的是gamma前两个维度的大小
        beta = beta.view(*beta.size()[:2], 1,1) # 得到的是(B*1, 256, 1, 1)，为了后面能进行广播用的
        out = normalized * (1 + gamma) + beta # 用广播机制进行运算
        return out

class AdaIN(torch.nn.Module):

    def __init__(self, input_channel, modulation_channel,kernel_size=3, stride=1, padding=1):
        super(AdaIN, self).__init__()
        self.conv_1 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(input_channel, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.adain_layer_1 = AdaINLayer(input_channel, modulation_channel) #AdaIN(256,512)
        self.adain_layer_2 = AdaINLayer(input_channel, modulation_channel) #AdaIN(64,512)

    def forward(self, x, modulation): 
        #  (B*1,256,64,64), (B*1,1,512)
        #  (B*1,64,128,128),(B*1,1,512)

        x = self.adain_layer_1(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_1(x)
        x = self.adain_layer_2(x, modulation)
        x = self.leaky_relu(x)
        x = self.conv_2(x)

        return x


class SPADELayer_ORI(torch.nn.Module):
    def __init__(self, input_channel, modulation_channel, hidden_size=128, kernel_size=3, stride=1, padding=1):
        super(SPADELayer_ORI, self).__init__()
        self.instance_norm = torch.nn.InstanceNorm2d(input_channel)

        self.conv1 = torch.nn.Conv2d(modulation_channel, hidden_size, kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        self.gamma = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.beta = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
# encoder decoder
    def forward(self, input, modulation):
        # (128,64,64)  (256,64,64)
        norm = self.instance_norm(input)
        conv_out = self.conv1(modulation) # (256,64,64)
        gamma = self.gamma(conv_out)
        beta = self.beta(conv_out)

        return norm + norm * gamma + beta


class SPADE_ORI(torch.nn.Module):
    def __init__(self, num_channel, num_channel_modulation, hidden_size=128, kernel_size=3, stride=1, padding=1):
        super(SPADE_ORI, self).__init__()
        self.conv_1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.spade_layer_1 = SPADELayer_ORI(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)
        self.spade_layer_2 = SPADELayer_ORI(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)

    def forward(self, input, modulations): # stride=1，在forward里输入输出向量大小不变
        input = self.spade_layer_1(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_1(input)
        input = self.spade_layer_2(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_2(input)
        return input


class SPADELayer(torch.nn.Module): # input_channel:256 or 64; modulation_channel=3*5; hidden_size=512
    def __init__(self, input_channel, modulation_channel, hidden_size=512, kernel_size=3, stride=1, padding=1,
                 T=1,d_model=512,nlayers=3,nhead=1,dim_feedforward=1024,dropout=0.1):
        super(SPADELayer, self).__init__()
        # self.instance_norm = torch.nn.InstanceNorm2d(input_channel)

        self.conv1 = torch.nn.Conv2d(modulation_channel, hidden_size, kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        # self.gamma = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        # self.beta = torch.nn.Conv2d(hidden_size, input_channel, kernel_size=kernel_size, stride=stride, padding=padding)

        self.input_channel = input_channel
        # encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, int(nlayers)) #!!说是float不能看作是整数
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True) 
        self.transformer_decoder = TransformerDecoder(decoder_layers, int(nlayers)) #!!说是float不能看作是整数
        self.Norm_1=nn.LayerNorm(d_model) # normalization layer
        self.Norm_2=nn.LayerNorm(d_model) # 两个tokens的normalization是不一样的，不能只用一个
 
        self.position_in = PositionalEmbedding(d_model=d_model,max_len=4096)  #for input image
        self.position_sk = PositionalEmbedding(d_model=d_model,max_len=4096)  #for input sketch
        self.dropout = nn.Dropout(p=dropout)
        
        self.d_model=d_model
        print("d_model_init",d_model)


#定义 encoder decoder                     
    def forward(self, input, modulation): #icey input (B*1, 512, 32, 32) #modulation(B*1, 3*5, 32, 32)
        # norm = self.instance_norm(input)
        input = input.flatten(2).permute(0,2,1) # (B*1, 1024, 512)  
        conv_out = self.conv1(modulation) # (B*1, 512, 32， 32) 
        conv_out = conv_out.flatten(2).permute(0,2,1) # (B*1, 1024, 512)      

# input b,256,4096 key
# modulation→conv_out-b,256,64,64 -> b,256,4096

        # gamma = self.gamma(conv_out) # or(B*1, 64, 128, 128)
        # beta = self.beta(conv_out)

        #normalization
        input_embedding = self.Norm_1(input) # (B*1, 1024, 512)  
        conv_out_embedding = self.Norm_2(conv_out)

        # 输入到encoder decoder
        # (1).  positional(temporal) encoding
        position_in_encoding = self.position_in(input_embedding) # (1, 1024, 512)  
        position_sk_encoding = self.position_sk(conv_out_embedding)
        #(2)  modality encoding
        input_tokens = input_embedding + position_in_encoding # + modality_v   # (B*1, 1024, 512)  
        sketch_tokens = conv_out_embedding + position_sk_encoding # + modality_a   

        input_tokens = self.dropout(input_tokens) # (B*1, 1024, 512)  
        sketch_tokens = self.dropout(sketch_tokens) 

        # #(4) input to transformer # nhead=1
        # encoder + decoder
        # output_encoder = self.transformer_encoder(input_tokens)
        # output_decoder = self.transformer_decoder(sketch_tokens, output_encoder)
        
        # 只有decoder
        output_decoder = self.transformer_decoder(sketch_tokens, input_tokens) # output_encoder作为kv
        
        # 原来的形状
        output_decoder = output_decoder.permute(0, 2, 1) # (B*1, 512, 1024)  
        #print("int(output_decoder.size(-1)** 0.5)",int(output_decoder.size(-1)** 0.5))
        output_decoder = output_decoder.reshape(-1, self.input_channel, int(output_decoder.size(-1)** 0.5), int(output_decoder.size(-1) ** 0.5))
        
        return output_decoder

        # return norm + norm * gamma + beta# 直接返回decoder的结果


class SPADE(torch.nn.Module):
    def __init__(self, num_channel, num_channel_modulation, hidden_size=512, kernel_size=3, stride=1, padding=1,
                 T=1,d_model=512,nlayers=3,nhead=1,dim_feedforward=1024,dropout=0.1):
        super(SPADE, self).__init__()
        print("Transformer layer",nlayers)
        self.conv_1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.spade_layer_1 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding,
                                        T=T,d_model=d_model,nlayers=nlayers,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout)
        self.spade_layer_2 = SPADELayer(num_channel, num_channel_modulation, hidden_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding,
                                        T=T,d_model=d_model,nlayers=nlayers,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout)

    def forward(self, input, modulations): # stride=1，在forward里输入输出向量大小不变
        input = self.spade_layer_1(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_1(input)
        input = self.spade_layer_2(input, modulations)
        input = self.leaky_relu(input)
        input = self.conv_2(input)
        return input

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, act='ReLU', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'Tanh':
            self.act = nn.Tanh()

        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

def downsample(x, size): # Icey: driving_sketch:(B, 3*5, H, W) size:(64, 64)
    if len(x.size()) == 5:
        size = (x.size(2), size[0], size[1])
        return torch.nn.functional.interpolate(x, size=size, mode='nearest')
    return  torch.nn.functional.interpolate(x, size=size, mode='nearest') # Icey Mode mode='nearest' matches buggy OpenCV’s INTER_NEAREST interpolation algorithm


def convert_flow_to_deformation(flow):
    r"""convert flow fields to deformations.
    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        deformation (tensor): The deformation used for warpping
    """
    b, c, h, w = flow.shape
    flow_norm = 2 * torch.cat([flow[:, :1, ...] / (w - 1), flow[:, 1:, ...] / (h - 1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.permute(0, 2, 3, 1)
    return deformation


def make_coordinate_grid(flow):
    r"""obtain coordinate grid with the same size as the flow filed.
    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        grid (tensor): The grid with the same size as the input flow
    """
    b, c, h, w = flow.shape

    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.expand(b, -1, -1, -1)
    return meshed


def warping(source_image, deformation):
    r"""warp the input image according to the deformation
    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear') # 将形变张量进行插值（interpolation），使其大小与输入图像相同
        deformation = deformation.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(source_image, deformation) # 对输入图像进行变形，利用形变张量中的信息进行像素插值，得到输出的变形后的图像


class DenseFlowNetwork(torch.nn.Module):
    def __init__(self, num_channel=6, num_channel_modulation=3*5, hidden_size=512,
                 T=1,d_model=512,nlayers=2,nhead=1,dim_feedforward=1024,dropout=0.1): # 初始化
        super(DenseFlowNetwork, self).__init__()

        # Convolutional Layers                                                            #(6,128,128)
        self.conv1 = torch.nn.Conv2d(num_channel, 32, kernel_size=7, stride=1, padding=3) #(32,128,128)
        self.conv1_bn = torch.nn.BatchNorm2d(num_features=32, affine=True) # Icey: C from an expected input of size (N,C,H,W), output shape = input
        self.conv1_relu = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(32, 256, kernel_size=3, stride=2, padding=1)         #(256,64,64)
        self.conv2_bn = torch.nn.BatchNorm2d(num_features=256, affine=True)
        self.conv2_relu = torch.nn.ReLU()

        # add conv for 32 * 32
        self.conv3 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)        #(512,32,32)
        self.conv3_bn = torch.nn.BatchNorm2d(num_features=512, affine=True)
        self.conv3_relu = torch.nn.ReLU()

        self.spade_layer_1 = SPADE(512, num_channel_modulation, hidden_size,
                                   T=T,d_model=d_model,nlayers=nlayers,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout)
        # SPADE 第2层和第4层都去掉
        # self.spade_layer_2 = SPADE(512, num_channel_modulation, hidden_size,
        #                            T=T,d_model=d_model,nlayers=nlayers,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout)
        self.pixel_shuffle_1 = torch.nn.PixelShuffle(2) # Icey r = 2, (∗,C × r^2 ,H,W) to (∗,C,H × r,W × r)
        self.pixel_shuffle_2 = torch.nn.PixelShuffle(2)
        self.spade_layer_4 = SPADE(128, num_channel_modulation, hidden_size=128,
                                   T=T,d_model=128,nlayers=nlayers,nhead=nhead,dim_feedforward=256,dropout=dropout)

        # Final Convolutional Layer
        self.conv_4 = torch.nn.Conv2d(32, 2, kernel_size=7, stride=1, padding=3)
        self.conv_5= nn.Sequential(torch.nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(16, 1, kernel_size=7, stride=1, padding=3),
                                   torch.nn.Sigmoid(),
                                   )#predict weight

    def forward(self, ref_N_frame_img, ref_N_frame_sketch, T_driving_sketch): #to output: (B*T,3,H,W)
                   #   (B, N, 3, H, W)(B, N, 3, H, W)    (B, 5, 3, H, W)  #
        ref_N = ref_N_frame_img.size(1)

        # 因为渲染时，T = 1, 故此处让 5 放到通道数这，作为 T = 1
        driving_sketch=torch.cat([T_driving_sketch[:,i] for i in range(T_driving_sketch.size(1))], dim=1)  #(B, 3*5, H, W)

        wrapped_h1_sum, wrapped_h2_sum, wrapped_h3_sum, wrapped_ref_sum=0.,0.,0.,0.
        softmax_denominator=0.
        T = 1  # during rendering, generate T=1 image  at a time
        for ref_idx in range(ref_N): # each ref img provide information for each B*T frame # 选择N帧中的1帧
            ref_img= ref_N_frame_img[:, ref_idx]  #(B, 3, H, W)
            ref_img = ref_img.unsqueeze(1).expand(-1, T, -1, -1, -1)  # (B,T, 3, H, W) # -1 表示保持该维度的原始大小
            ref_img = torch.cat([ref_img[i] for i in range(ref_img.size(0))], dim=0)  # (B*T, 3, H, W)

            ref_sketch = ref_N_frame_sketch[:, ref_idx] #(B, 3, H, W)
            ref_sketch = ref_sketch.unsqueeze(1).expand(-1, T, -1, -1, -1)  # (B,T, 3, H, W)
            ref_sketch = torch.cat([ref_sketch[i] for i in range(ref_sketch.size(0))], dim=0)  # (B*T, 3, H, W)

            #predict flow and weight
            flow_module_input = torch.cat((ref_img, ref_sketch), dim=1)  #(B*T, 3+3, H, W)
            # Convolutional Layers
            h1 = self.conv1_relu(self.conv1_bn(self.conv1(flow_module_input)))   #(32,128,128)
            h2 = self.conv2_relu(self.conv2_bn(self.conv2(h1)))    #(256,64,64)
            h3 = self.conv3_relu(self.conv3_bn(self.conv3(h2)))    #(512,32,32)
            # h3 (B, 512, 32, 32), 因为 T = 1
            
            # SPADE Blocks↓-------------------------------------------------------
            downsample_64 = downsample(driving_sketch, (64, 64))
            downsample_32 = downsample(driving_sketch, (32, 32))
            # driving_sketch:(B, T*3, 128, 128), downsample_64(B, 5*3, 64, 64), downsample_32(B, 5*3, 32, 32) T=5

            # Icey 大小前面还有 B*T
            # spade有两层，的两个输入是 h2 和 driving_sketch
            spade_layer = self.spade_layer_1(h3, downsample_32)             #(512,32,32)
            # spade_layer = self.spade_layer_2(spade_layer, downsample_32)   #(512,32,32)

            spade_layer = self.pixel_shuffle_1(spade_layer)   #(128,64,64)

            spade_layer = self.spade_layer_4(spade_layer, downsample_64)#(128,64,64)
            
            ##!!!!!卷积前再进行一个上采样，暂时用 pixel_shuffle 代替
            spade_layer = self.pixel_shuffle_2(spade_layer)   #(32,128,128)
            # SPADE Blocks↑-------------------------------------------------------

            # Final Convolutional Layer # conv4 (in 64, out 2) # conv5 (in 64, out 1)
            output_flow = self.conv_4(spade_layer)      #   (B*T,2,128,128)
            output_weight=self.conv_5(spade_layer)       #  (B*T,1,128,128)

            deformation=convert_flow_to_deformation(output_flow)
            wrapped_h1 = warping(h1, deformation)  #(32,128,128)
            wrapped_h2 = warping(h2, deformation)  #(256,64,64)
            wrapped_h3 = warping(h3, deformation)  #(512,32,32)
            wrapped_ref = warping(ref_img, deformation)  #(3,128,128)

            softmax_denominator+=output_weight
            wrapped_h1_sum+=wrapped_h1*output_weight
            wrapped_h2_sum+=wrapped_h2*downsample(output_weight, (64,64))
            wrapped_h3_sum+=wrapped_h3*downsample(output_weight, (32,32))
            wrapped_ref_sum+=wrapped_ref*output_weight
        #return weighted warped feataure and images
        softmax_denominator+=0.00001
        wrapped_h1_sum=wrapped_h1_sum/softmax_denominator
        wrapped_h2_sum = wrapped_h2_sum / downsample(softmax_denominator, (64,64))
        wrapped_h3_sum = wrapped_h3_sum / downsample(softmax_denominator, (32,32))
        wrapped_ref_sum = wrapped_ref_sum / softmax_denominator
        return wrapped_h1_sum, wrapped_h2_sum, wrapped_h3_sum, wrapped_ref_sum


class TranslationNetwork(torch.nn.Module):
    def __init__(self, T=1,d_model=512,nlayers=3,nhead=1,dim_feedforward=1024,dropout=0.1):
        super(TranslationNetwork, self).__init__()
        self.audio_encoder = nn.Sequential( # 这个encoder和Landmark_generator一样，但act不同
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ) # 另一个act='Tanh'

        # Encoder
        self.conv1 = torch.nn.Conv2d(in_channels=3+3*5, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1_bn = torch.nn.BatchNorm2d(num_features=32, affine=True) # affine=True，给该层添加可学习的仿射变换参数
        self.conv1_relu = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2_bn = torch.nn.BatchNorm2d(num_features=256, affine=True)
        self.conv2_relu = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3_bn = torch.nn.BatchNorm2d(num_features=512, affine=True)
        self.conv3_relu = torch.nn.ReLU()

        # Decoder
        self.spade_1 = SPADE(num_channel=512, num_channel_modulation=512,
                             T=T,d_model=d_model,nlayers=nlayers,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout)
        self.adain_1 = AdaIN(input_channel=512,modulation_channel=512)
        self.pixel_suffle_1 = nn.PixelShuffle(upscale_factor=2) # Icey r = 2, (∗,C × r^2 ,H,W) to (∗,C,H × r,W × r)

        # self.spade_2 = SPADE_ORI(num_channel=128, num_channel_modulation=256)
        self.spade_2 = SPADE(num_channel=128, num_channel_modulation=256,hidden_size=128,
                             T=T,d_model=128,nlayers=nlayers,nhead=nhead,dim_feedforward=256,dropout=dropout)
        self.adain_2 = AdaIN(input_channel=128,modulation_channel=512)
        self.spade_4 = SPADE_ORI(num_channel=128, num_channel_modulation=32)

        self.pixel_suffle_2 = nn.PixelShuffle(upscale_factor=2) 

        # Final layer
        self.leaky_relu = torch.nn.LeakyReLU()
        self.conv_last = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, stride=1, padding=3, bias=False)
        self.Sigmoid=torch.nn.Sigmoid()
    def forward(self, translation_input, wrapped_ref, wrapped_h1, wrapped_h2, wrapped_h3, T_mels): # Icey: h=w=128
        #              (B,3+3*5,H,W)   (B,3,128,128)  (B,32,128,128) (B,256,64,64) (B,512,32,32) (B,T,1,h,w)  #T=1
        # Encoder
        T_mels=torch.cat([T_mels[i] for i in range(T_mels.size(0))],dim=0)# B*T,1,h,w # h=80,w=16
        x = self.conv1_relu(self.conv1_bn(self.conv1(translation_input)))    #B,32,128,128
        x = self.conv2_relu(self.conv2_bn(self.conv2(x)))  #B,256,64,64
        x = self.conv3_relu(self.conv3_bn(self.conv3(x)))  #B,512,32,32 # add 32*32

        #print("T_mels",T_mels.size()) #1, 1, 80, 16 error, 原来的是20, 1, 80, 16 
        audio_feature = self.audio_encoder(T_mels).squeeze(-1).permute(0,2,1) #(B*T,1,512) #T=1

        # Decoder
        x = self.spade_1(x, wrapped_h3) # (C=512,32,32)
        x = self.adain_1(x, audio_feature)  # (C=512,32,32)
        x = self.pixel_suffle_1(x)   # (C=128,64,64)


        x = self.spade_2(x, wrapped_h2)   # (C=128,64,64)
        x = self.adain_2(x, audio_feature)  # (C=128,64,64)
        # 不要 wrapped_ref 了，换成h1，并且下采样了
        # x = self.spade_4(x, wrapped_ref)    # (64,128,128)
        wrapped_h1 = downsample(wrapped_h1, (64, 64)) # (B,32,64,64)
        x = self.spade_4(x, wrapped_h1)    # (C=128,64,64)


        x = self.pixel_suffle_2(x) # (32,128,128)

        # output layer
        x = self.leaky_relu(x)
        x = self.conv_last(x)
        x = self.Sigmoid(x)
        return x

class Renderer(torch.nn.Module):
    
    def __init__(self):
        super(Renderer, self).__init__()

        # 1.flow Network
        self.flow_module = DenseFlowNetwork() 
        #2. translation Network
        self.translation = TranslationNetwork()
        #3.return loss
        self.perceptual = PerceptualLoss(network='vgg19',
                                         layers=['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1'],
                                         num_scales=2)

    def forward(self, face_frame_img, target_sketches, ref_N_frame_img, ref_N_frame_sketch, audio_mels): #T=1
        #            (B,1,3,H,W)   (B,5,3,H,W)       (B,N,3,H,W)   (B,N,3,H,W)  (B,T,1,hv,wv)T=1 Icey:audio(B,1,1,80,16)
        # (1)warping reference images and their feature
        wrapped_h1, wrapped_h2, wrapped_h3, wrapped_ref = self.flow_module(ref_N_frame_img, ref_N_frame_sketch, target_sketches)
        #(B,C,H,W)

        # (2)translation module
        target_sketches = torch.cat([target_sketches[:, i] for i in range(target_sketches.size(1))], dim=1)
        # (B,3*T,H,W)
        gt_face = torch.cat([face_frame_img[i] for i in range(face_frame_img.size(0))], dim=0)
        # (B,3,H,W)
        gt_mask_face = gt_face.clone()
        gt_mask_face[:, :, gt_mask_face.size(2) // 2:, :] = 0  # (B,3,H,W) # Icey 高度设为原来的一半了，(B, C, H, W) → 批量数、通道数、高、宽
        #
        translation_input=torch.cat([gt_mask_face, target_sketches], dim=1)  #Icey (B,3+3*5,H,W)
        generated_face = self.translation(translation_input, wrapped_ref, wrapped_h1, wrapped_h2, wrapped_h3, audio_mels) #translation_input

        perceptual_gen_loss = self.perceptual(generated_face, gt_face, use_style_loss=True,
                                              weight_style_to_perceptual=250).mean()
        perceptual_warp_loss = self.perceptual(wrapped_ref, gt_face, use_style_loss=False,
                                               weight_style_to_perceptual=0.).mean()
        return generated_face, wrapped_ref, torch.unsqueeze(perceptual_warp_loss, 0), torch.unsqueeze(
            perceptual_gen_loss, 0)
        # (B,3,H,W) and losses

#the following is the code for Perceptual(VGG) loss

def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input back to [0, 1]
    normalized_input = (input + 1) / 2
    # normalize the input using the ImageNet mean and std
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output

class _PerceptualNetwork(nn.Module):
    r"""The network that extracts features to compute the perceptual loss.

    Args:
        network (nn.Sequential) : The network that extracts features.
        layer_name_mapping (dict) : The dictionary that
            maps a layer's index to its name.
        layers (list of str): The list of layer names that we are using.
    """

    def __init__(self, network, layer_name_mapping, layers):
        super().__init__()
        assert isinstance(network, nn.Sequential), \
            'The network needs to be of type "nn.Sequential".'
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        r"""Extract perceptual features."""
        output = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                # If the current layer is used by the perceptual loss.
                output[layer_name] = x
        return output

def _vgg19(layers):
    r"""Get vgg19 layers"""
    network = torchvision.models.vgg19(pretrained=True).features
    layer_name_mapping = {1: 'relu_1_1',
                          3: 'relu_1_2',
                          6: 'relu_2_1',
                          8: 'relu_2_2',
                          11: 'relu_3_1',
                          13: 'relu_3_2',
                          15: 'relu_3_3',
                          17: 'relu_3_4',
                          20: 'relu_4_1',
                          22: 'relu_4_2',
                          24: 'relu_4_3',
                          26: 'relu_4_4',
                          29: 'relu_5_1'}
    return _PerceptualNetwork(network, layer_name_mapping, layers)

class PerceptualLoss(nn.Module):
    r"""Perceptual loss initialization.

    Args:
        network (str) : The name of the loss network: 'vgg16' | 'vgg19'.
        layers (str or list of str) : The layers used to compute the loss.
        weights (float or list of float : The loss weights of each layer.
        criterion (str): The type of distance function: 'l1' | 'l2'.
        resize (bool) : If ``True``, resize the input images to 224x224.
        resize_mode (str): Algorithm used for resizing.
        instance_normalized (bool): If ``True``, applies instance normalization
            to the feature maps before computing the distance.
        num_scales (int): The loss will be evaluated at original size and
            this many times downsampled sizes.
    """

    def __init__(self, network='vgg19', layers='relu_4_1', weights=None,
                 criterion='l1', resize=False, resize_mode='bilinear',
                 instance_normalized=False, num_scales=1,):
        super().__init__()
        if isinstance(layers, str):
            layers = [layers]
        if weights is None:
            weights = [1.] * len(layers)
        elif isinstance(layers, float) or isinstance(layers, int):
            weights = [weights]

        assert len(layers) == len(weights), \
            'The number of layers (%s) must be equal to ' \
            'the number of weights (%s).' % (len(layers), len(weights))
        if network == 'vgg19':
            self.model = _vgg19(layers)
        else:
            raise ValueError('Network %s is not recognized' % network)

        self.num_scales = num_scales
        self.layers = layers
        self.weights = weights
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized


        print('Perceptual loss:')
        print('\tMode: {}'.format(network))

    def forward(self, inp, target, mask=None,use_style_loss=False,weight_style_to_perceptual=0.):
        r"""Perceptual loss forward.

        Args:
           inp (4D tensor) : Input tensor.
           target (4D tensor) : Ground truth tensor, same shape as the input.

        Returns:
           (scalar tensor) : The perceptual loss.
        """
        # Perceptual loss should operate in eval mode by default.
        self.model.eval()
        inp, target = \
            apply_imagenet_normalization(inp), \
            apply_imagenet_normalization(target)
        if self.resize:
            inp = F.interpolate(
                inp, mode=self.resize_mode, size=(256, 256),
                align_corners=False)
            target = F.interpolate(
                target, mode=self.resize_mode, size=(256, 256),
                align_corners=False)

        # Evaluate perceptual loss at each scale.
        loss = 0
        style_loss=0
        for scale in range(self.num_scales):
            input_features, target_features = \
                self.model(inp), self.model(target)
            for layer, weight in zip(self.layers, self.weights):
                # Example per-layer VGG19 loss values after applying
                # [0.03125, 0.0625, 0.125, 0.25, 1.0] weighting.
                # relu_1_1, 0.014698
                # relu_2_1, 0.085817
                # relu_3_1, 0.349977
                # relu_4_1, 0.544188
                # relu_5_1, 0.906261
                input_feature = input_features[layer]
                target_feature = target_features[layer].detach()
                if self.instance_normalized:
                    input_feature = F.instance_norm(input_feature)
                    target_feature = F.instance_norm(target_feature)

                if mask is not None:
                    mask_ = F.interpolate(mask, input_feature.shape[2:],
                                          mode='bilinear',
                                          align_corners=False)
                    input_feature = input_feature * mask_
                    target_feature = target_feature * mask_
                    # print('mask',mask_.shape)


                loss += weight * self.criterion(input_feature,
                                                target_feature)
                if use_style_loss and scale==0:
                    style_loss += self.criterion(self.compute_gram(input_feature),
                                                 self.compute_gram(target_feature))

            # Downsample the input and target.
            if scale != self.num_scales - 1:
                inp = F.interpolate(
                    inp, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)
                target = F.interpolate(
                    target, mode=self.resize_mode, scale_factor=0.5,
                    align_corners=False, recompute_scale_factor=True)

        if use_style_loss:
            return loss + style_loss*weight_style_to_perceptual
        else:
            return loss


    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

