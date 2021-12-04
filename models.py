import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

from buildingblocks import Encoder, Decoder, FinalConv, DoubleConv, ExtResNetBlock, SingleConv, DecoderAttention

# from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SegNet(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet, self).__init__()
        batchNorm_momentum = 0.1
        self.batchNorm_momentum = batchNorm_momentum
        
        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        
        self.sigm = nn.Sigmoid()

    def forward(self, x):
# Each encoder layer consists of one (actually two)
# convolutional layer with batch normalization and a ReLu non-linearity layer, which is followed by
# a maxpooling layer for downsampling.
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)
        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)
        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)
        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)
        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)
# Each decoder consists of one (three) convolutional layer
# with batch normalization and a ReLu non-linearity layer, followed by a upsampling layer.
        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        return self.sigm(x11d)

    
class SegNet_3D(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(SegNet_3D, self).__init__()
        batchNorm_momentum = 0.1
        self.batchNorm_momentum = batchNorm_momentum
        
        self.conv11 = nn.Conv3d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm3d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm3d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm3d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm3d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm3d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm3d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm3d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm3d(512, momentum= batchNorm_momentum)

        self.conv53d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm3d(512, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm3d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm3d(256, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm3d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm3d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv3d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm3d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm3d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm3d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm3d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv3d(64, label_nbr, kernel_size=3, padding=1)
        
        self.sigm = nn.Sigmoid()

    def forward(self, x):
# Each encoder layer consists of one (actually two)
# convolutional layer with batch normalization and a ReLu non-linearity layer, which is followed by
# a maxpooling layer for downsampling.
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool3d(x12,kernel_size=2, stride=2,return_indices=True)
        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool3d(x22,kernel_size=2, stride=2,return_indices=True)
        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool3d(x33,kernel_size=2, stride=2,return_indices=True)
        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool3d(x43,kernel_size=2, stride=2,return_indices=True)
        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool3d(x53,kernel_size=2, stride=2,return_indices=True)
# Each decoder consists of one (three) convolutional layer
# with batch normalization and a ReLu non-linearity layer, followed by a upsampling layer.
        # Stage 5d
        x5d = F.max_unpool3d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool3d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool3d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool3d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool3d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        return self.sigm(x11d)

    

class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8, 
                 pool_kernel_size=(2,2,2),
                 **kwargs):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            number_of_fmaps=4
            f_maps = [f_maps * 2 ** k for k in range(number_of_fmaps)]

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
            
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, pool_kernel_size=pool_kernel_size,
                                  basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, pool_kernel_size=pool_kernel_size,
                                  basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            
                
            decoder = Decoder(in_feature_num, out_feature_num, scale_factor=pool_kernel_size, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)
#             self.final_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # encoder part
#         print (x.shape)
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x
    

    
class UNet3D_attention(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8, device=0,
                 **kwargs):
        super(UNet3D_attention, self).__init__()
        self.device = device

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            number_of_fmaps=4
            f_maps = [f_maps * 2 ** k for k in range(number_of_fmaps)]
#             f_maps = create_feature_maps(f_maps, number_of_fmaps=4)
#           f_maps = [64, 128, 256, 512]  # filters

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            
            if i == len(reversed_f_maps) - 2:
                inter_feature_num = reversed_f_maps[i + 1]//2
            else:
                inter_feature_num = reversed_f_maps[i + 2]
                
            decoder = DecoderAttention(in_feature_num, out_feature_num, inter_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups,
                                        device = self.device
                                       )
            decoders.append(decoder)


        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)
    

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x
    


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.outChans = outChans
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        x_out = torch.cat(self.outChans*[x], 1)
        out = self.relu1(torch.add(out, x_out))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out
    
# 2D Attention UNet: https://towardsdatascience.com/biomedical-image-segmentation-attention-u-net-29b6f0827405        
class Attention(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super(Attention, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int))
#         self.W_g = SingleConv(F_l, F_int, kernel_size=3, order='cb',padding=0)
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.BatchNorm3d(F_int))
#         self.W_x = SingleConv(F_g, F_int, kernel_size=3, order='cb',padding=0)

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
#         print('g',g.shape)
#         print (g.device)
#         print (self.W_g)
        g1 = self.W_g(g)
#         print('g1',g1.shape)
#         print('x',x.shape)
        x1 = self.W_x(x)
#         print('x1',x1.shape)
        psi = self.relu(g1 + x1)
#         print ('psi',psi.shape)
        psi = self.psi(psi)
#         print ('psi',psi.shape)
        out = x * psi
        return out

class UpTransitionAtt(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransitionAtt, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
#         self.up_conv = nn.ConvTranspose3d(inChans, inChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
#         self.bn1 = ContBatchNorm3d(inChans)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
#         self.relu1 = ELUCons(elu, inChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)
        
        inter_feature_num = outChans//2
        f_g = inter_feature_num
        f_x = inter_feature_num
#         print (f_g, f_x, inter_feature_num)
        self.attention = Attention(f_g, f_x, inter_feature_num) #f_g, f_x, f_int

    def forward(self, x, skipx):
        x = self.do1(x) # passthrough
        skipxdo = self.do2(skipx) # nn.Dropout3d()
        x_up = self.relu1(self.bn1(self.up_conv(x)))
        x_attention = self.attention(x_up, skipxdo)
#         print ('x_att',x_attention.shape)
#         print ('enc', skipxdo.shape)
#         print ('x_up', x_up.shape)
        xcat = torch.cat((x_up, x_attention), 1)
#         print ('xcat', xcat.shape)
        xcat_old = torch.cat((x_up, skipxdo), 1)
#         print ('xcat_old', xcat_old.shape)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
#         print ('out',out.shape)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = nn.LogSoftmax(dim=-1)
        else:     
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = self.softmax(out)
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
#     def __init__(self, elu=True, nll=False):
#         super(VNet, self).__init__()
#         self.in_tr = InputTransition(16, elu)
#         self.down_tr32 = DownTransition(16, 1, elu)
#         self.down_tr64 = DownTransition(32, 2, elu)
#         self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
#         self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
#         self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
#         self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
#         self.up_tr64 = UpTransition(128, 64, 1, elu)
#         self.up_tr32 = UpTransition(64, 32, 1, elu)
#         self.out_tr = OutputTransition(32, elu, nll)
        
    # The network topology as described in the VNet paper
    def __init__(self, elu=False, nll=False):
        super(VNet, self).__init__()
        self.in_tr =  InputTransition(16, elu)
        # the number of convolutions in each layer corresponds
        # to what is in the actual prototxt, not the intent
        self.down_tr32 = DownTransition(16, 2, elu)
        self.down_tr64 = DownTransition(32, 3, elu)
        self.down_tr128 = DownTransition(64, 3, elu)
        self.down_tr256 = DownTransition(128, 3, elu)
        self.up_tr256 = UpTransition(256, 256, 3, elu)
        self.up_tr128 = UpTransition(256, 128, 3, elu)
        self.up_tr64 = UpTransition(128, 64, 2, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32,elu, nll)
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        out = out.permute(0,4,1,2,3)
        return out  
    
class VNet_attention(nn.Module):
    
#     def __init__(self, elu=False, nll=False):
#         super(VNet_attention, self).__init__()
#         self.in_tr = InputTransition(16, elu)
#         self.down_tr32 = DownTransition(16, 1, elu)
#         self.down_tr64 = DownTransition(32, 2, elu)
#         self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
#         self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
#         self.up_tr256 = UpTransitionAtt(256, 256, 2, elu, dropout=True)
#         self.up_tr128 = UpTransitionAtt(256, 128, 2, elu, dropout=True)
#         self.up_tr64 = UpTransitionAtt(128, 64, 1, elu)
#         self.up_tr32 = UpTransitionAtt(64, 32, 1, elu)
#         self.out_tr = OutputTransition(32, elu, nll)

    def __init__(self, elu=False, nll=False):
        super(VNet, self).__init__()
        self.in_tr =  InputTransition(16, elu)
        # the number of convolutions in each layer corresponds
        # to what is in the actual prototxt, not the intent
        self.down_tr32 = DownTransition(16, 2, elu)
        self.down_tr64 = DownTransition(32, 3, elu)
        self.down_tr128 = DownTransition(64, 3, elu)
        self.down_tr256 = DownTransition(128, 3, elu)
        self.up_tr256 = UpTransition(256, 256, 3, elu)
        self.up_tr128 = UpTransition(256, 128, 3, elu)
        self.up_tr64 = UpTransition(128, 64, 2, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32,elu, nll)
        


    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
#         print ('1', out.shape)
        out = self.up_tr128(out, out64)
#         print ('2', out.shape)
        out = self.up_tr64(out, out32)
#         print ('3', out.shape)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        out = out.permute(0,4,1,2,3)
        return out

   
class LSTM(nn.Module):
    def __init__(self,input_dim= 37*48, hidden_dim=1000, num_layers=1, output_dim=37*48,
                 batch = 256, drop_prob=0.2):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch = batch
        self.output_dim = output_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim) 
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, x, hidden):
        """ param image: torch tensor containing inception vectors. shape: [batch, cnn_feature_size]""" 
        seq_len = x.size(2)
        lstm_out, hidden = self.lstm(x[0].view(self.batch, seq_len, self.input_dim), hidden)
        lstm_out = lstm_out.view(self.batch,-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out.view(self.batch, seq_len, self.input_dim)
        return out, hidden

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, self.batch, self.hidden_dim).zero_(),#.to(device),
                      weight.new(self.num_layers, self.batch, self.hidden_dim).zero_())#.to(device))
        return hidden
    

class UNet3D_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8, 
                 pool_kernel_size=(2,2,2),
                 input_dim = 37*48,hidden_dim=1024,n_layers=2,seq_len = 8,batch_size=256,drop_prob=0.2,
                 **kwargs):
        super(UNet3D_LSTM, self).__init__()
        self.batch_size=batch_size
        self.seq_len = seq_len

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            number_of_fmaps=4
            f_maps = [f_maps * 2 ** k for k in range(number_of_fmaps)]

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
            
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, pool_kernel_size=pool_kernel_size,
                                  basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, pool_kernel_size=pool_kernel_size,
                                  basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
#                 print (f_maps[i - 1], out_feature_num, pool_kernel_size,num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            
                
            decoder = Decoder(in_feature_num, out_feature_num, scale_factor=pool_kernel_size, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)
#             self.final_activation = nn.LogSoftmax(dim=1)

        output_dim=input_dim
        self.lstm_layer = LSTM(input_dim, hidden_dim, n_layers, output_dim, batch_size, drop_prob)#.to(x.device)

    def forward(self, x):
        # encoder part
#         print (x.shape)
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            
        encoder_x = x
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
# LSTM part:
# batch size, sequence length, input dimension

        hidden = self.lstm_layer.init_hidden()
        lstm_out, hidden = self.lstm_layer(x, hidden)
#             lstm_out_resh = lstm_out.view([batch_size, seq_len, x.shape[3],x.shape[4]])

        lstm_x = lstm_out.view(1,self.batch_size,self.seq_len,encoder_x.shape[-2],encoder_x.shape[-1])
        x = lstm_x
#         print("LSTM output shape: ", x.shape)

# Decoder part:
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)
            
#         print("Final output shape: ", x.shape)
        return x, encoder_x, lstm_x
    