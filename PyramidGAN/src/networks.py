import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from src.transfomer import SelfAttention
# from torchsummary import summary
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x

def make_layers(in_channel, out_channel, kernel_size, stride, padding, dilation=1, bias=True, norm=True, activation=True, is_relu=False):
    layer = []
    layer.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
    if norm:
        layer.append(nn.InstanceNorm2d(out_channel, affine=True))
    if activation:
        if is_relu:
            layer.append(nn.ReLU())
        else:
            layer.append(nn.LeakyReLU(negative_slope=0.2))
    return nn.Sequential(*layer)

class Abstract_Feature_Extraction(nn.Module):
    def __init__(self, size=[256, 4, 4], split=4,  device=0):
        # --PARAMS--
        # size: size for each input feature map [channel, height, width]
        # split: # of portions each input feature map should be split into
        # pred_step: # of steps for the prediction of the intermediate region
        #            (the width of the resultant intermediate region will be the width of input portion * pred_step)

        super(Abstract_Feature_Extraction, self).__init__()

        self.channel, self.height, self.width = size


        self.lstm_size = int(self.channel * 4 * int(4 / split))
        self.LSTM_direct = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        self.LSTM_reverse = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        self.LSTM_up = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        self.LSTM_down = nn.LSTM(self.lstm_size, self.lstm_size, num_layers=2, batch_first=True)
        layer = []
        layer.append(nn.Conv2d(4*self.channel, self.channel, kernel_size=1, stride=1, padding=0,
                               dilation=1, bias=True))
        layer.append(nn.LeakyReLU(negative_slope=0.2))
        self.output = nn.Sequential(*layer)
        self.split = split
        step = int(math.log(self.height / 4 / 4,2))
        self.encoder = []
        self.decoder = []

        for i in range(step):
            self.encoder.append(nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=4, stride=2, padding=1))
            self.encoder.append(nn.InstanceNorm2d(self.channel, track_running_stats=False))
            self.encoder.append(nn.ReLU(True))
        for i in range(step):
            self.decoder.append(nn.ConvTranspose2d(in_channels=self.channel, out_channels=self.channel, kernel_size=4, stride=2, padding=1),)
            self.decoder.append(nn.InstanceNorm2d(self.channel, track_running_stats=False) )
            self.decoder.append(nn.ReLU(True))
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        self.device = device
        self.core_width = 4
        self.core_height = 4
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_out = x
        x_out = self.encoder(x_out)

        init_hidden = (Variable(torch.zeros(2, x_out.shape[0], self.lstm_size)).cuda(self.device),
                       Variable(torch.zeros(2, x_out.shape[0], self.lstm_size)).cuda(self.device))  # h0和c0


        x_split = torch.stack(torch.split(x_out, int(self.core_width / self.split), dim=3)).view(self.split, -1, 1,
                                                                                          self.lstm_size)
        # print('x_out',x_out.shape)
        # print('split',torch.split(x_out, int(self.core_width / self.split), dim=3)[0].shape)
        # print('stack',torch.stack(torch.split(x_out, int(self.core_width / self.split), dim=3)).shape)
        x_split_reversed = torch.stack(torch.split(x_out, int(self.core_width / self.split), dim=3)).flip(dims=[4]).view(
            self.split, -1, 1, self.lstm_size)
        x_split_up=torch.stack(torch.split(x_out, int(self.core_width / self.split), dim=2)).view(self.split, -1, 1,
                                                                                          self.lstm_size)
        x_split_down = torch.stack(torch.split(x_out, int(self.core_width / self.split), dim=2)).flip(dims=[3]).view(self.split, -1, 1,
                                                                                          self.lstm_size)

        for i in range(self.split):
            de_out, init_hidden = self.LSTM_direct(x_split[i], init_hidden)  # f_1^2 ~ f_1^5
            if i == 0:
                out_direct  = de_out.view(-1, self.channel, self.core_height, int(self.core_width / self.split))
            else:
                out_direct = torch.cat(( out_direct,de_out.view(-1, self.channel, self.core_height, int(self.core_width / self.split))), 3)
        init_hidden = (Variable(torch.zeros(2, x_out.shape[0], self.lstm_size)).cuda(self.device),
                       Variable(torch.zeros(2, x_out.shape[0], self.lstm_size)).cuda(self.device))  # h0和c0
        for i in range(self.split):
            de_out, init_hidden = self.LSTM_direct(x_split_reversed[i], init_hidden)  # f_1^2 ~ f_1^5
            if i == 0:
                out_reversed  = de_out.view(-1, self.channel, self.core_height, int(self.core_width / self.split))
            else:
                out_reversed = torch.cat(( out_reversed,de_out.view(-1, self.channel, self.core_height, int(self.core_width / self.split))), 3)

        init_hidden = (Variable(torch.zeros(2, x_out.shape[0], self.lstm_size)).cuda(self.device),
                       Variable(torch.zeros(2, x_out.shape[0], self.lstm_size)).cuda(self.device))  # h0和c0
        for i in range(self.split):
            de_out, init_hidden = self.LSTM_up(x_split_up[i], init_hidden)  # f_1^2 ~ f_1^5
            if i == 0:
                out_up  = de_out.view(-1, self.channel, self.core_height, int(self.core_width / self.split))
            else:
                out_up = torch.cat(( out_up,de_out.view(-1, self.channel, self.core_height, int(self.core_width / self.split))), 3)
        init_hidden = (Variable(torch.zeros(2, x_out.shape[0], self.lstm_size)).cuda(self.device),
                       Variable(torch.zeros(2, x_out.shape[0], self.lstm_size)).cuda(self.device))  # h0和c0
        for i in range(self.split):
            de_out, init_hidden = self.LSTM_down(x_split_down[i], init_hidden)  # f_1^2 ~ f_1^5
            if i == 0:
                out_down  = de_out.view(-1, self.channel, self.core_height, int(self.core_width / self.split))
            else:
                out_down = torch.cat(( out_down,de_out.view(-1, self.channel, self.core_height, int(self.core_width / self.split))), 3)


        x_out = self.output(torch.cat((out_direct, out_reversed,out_up,out_down), 1))
        x_out = self.decoder(x_out)
        out = self.gamma * x_out + x
        print('gama:',self.gamma)

        return out

class TransferGenerator(BaseNetwork):
    def __init__(self, img_size_min, num_scale, scale_factor,size_list,init_weights=True):
        super(TransferGenerator, self).__init__()
        self.img_size_min = img_size_min
        self.scale_factor = scale_factor
        self.num_scale = num_scale
        self.nf = 32
        self.current_scale = 0
        self.residual_blocks = 4
        self.size_list = size_list
        self.conv_out = nn.ModuleList()
        for i in range(0,len(size_list)):
            self.conv_out.append(nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,stride= 1, padding=1))
        self.merge = nn.ModuleList()
        for i in range(0,len(size_list)-1):
            self.merge.append(BasicConv(256, 256, kernel_size=3, stride=1, relu=False))
        self.x_in_deconv = nn.ModuleList()
        for i in range(0,len(size_list)-1):
            self.x_in_deconv.append(BasicConv(3,3, kernel_size=4, relu=True, stride=2, transpose=True))
        self.middle_in_deconv = nn.ModuleList()
        for i  in range(0,len(size_list)-1):
            self.middle_in_deconv.append(BasicConv(256,256, kernel_size=4, relu=True, stride=2, transpose=True))


        #浅层特征
        encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

        )
        #抽象特征提取
        self.AFE = Abstract_Feature_Extraction(size=[256,self.size_list[self.current_scale], self.size_list[self.current_scale]], split=4, device=0)
        blocks = []
        for _ in range(self.residual_blocks):
            if _ == self.residual_blocks//2:
                blocks.append(SelfAttention(256))
            block = ResnetBlock(256, 2)
            blocks.append(block)

        middle = nn.Sequential(*blocks)

        decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

        self.sub_generators = nn.ModuleList()
        first_generator = nn.ModuleList()

        first_generator.append(encoder)
        first_generator.append(middle)
        first_generator.append(decoder)

        # first_generator = nn.Sequential(*first_generator)

        self.sub_generators.append(first_generator)

    def forward(self, z, img=None):
        x_list = []
        x_middle_list = []
        encoder = self.sub_generators[0][0](z[0])
        AFE  = self.AFE(encoder)
        middle = self.sub_generators[0][1](AFE)
        docoder = self.sub_generators[0][2](middle)
        x_first = (torch.tanh(docoder) + 1) / 2
        x_list.append(x_first)

        x_middle_list.append(middle)

        if img is not None:
            x_inter = img
        else:
            x_inter = x_first

        for i in range(1, self.current_scale + 1):
            #x_inter = F.interpolate(x_inter, (self.size_list[i], self.size_list[i]), mode='bilinear', align_corners=True)
            x_inter = self.x_in_deconv[i-1](x_inter)
            #x_prev = x_inter
            x_inter = x_inter + z[i]#草图3通道
            # print(self.conv_out[i](z[i]*x_inter).shape)
            x_inter = z[i] + self.conv_out[i](x_inter)#草图3通道

            encoder = self.sub_generators[i][0](x_inter)
            AFE = self.sub_generators[i][1](encoder)#AFE模块
            #merge = encoder*x_middle_list[-1]
            merge = AFE*self.middle_in_deconv[i-1](x_middle_list[-1])
            middle = self.sub_generators[i][2](AFE+merge)
            # middle = self.sub_generators[i][2](AFE)
            docoder = self.sub_generators[i][3](middle)
            x_middle_list.append(middle)

            x_inter = (torch.tanh(docoder) + 1) / 2 # + x_prev
            x_list.append(x_inter)

        return x_list

    def progress(self):
        self.current_scale += 1

        if self.current_scale % 1 == 0:
            self.residual_blocks = self.residual_blocks + 1
        encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )
        AFE = Abstract_Feature_Extraction(size=[256,self.size_list[self.current_scale], self.size_list[self.current_scale]], split=4, device=0)
        blocks = []
        for _ in range(self.residual_blocks):
            if _ == self.residual_blocks//2:
                blocks.append(SelfAttention(256))
            block = ResnetBlock(256, 2)
            blocks.append(block)

        middle = nn.Sequential(*blocks)

        decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        tmp_generator = nn.ModuleList()

        tmp_generator.append(encoder)
        tmp_generator.append(AFE)
        tmp_generator.append(middle)
        tmp_generator.append(decoder)

        # tmp_generator = nn.Sequential(*tmp_generator)

        # if self.current_scale % 2 != 0:
        #     prev_generator = self.sub_generators[-1]
        #
        #     # Initialize layers via copy
        #     if self.current_scale >= 1:
        #         tmp_generator.load_state_dict(prev_generator.state_dict())

        self.sub_generators.append(tmp_generator)
        print("GENERATOR PROGRESSION DONE")

'''
class TransferGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(TransferGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x

'''

class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.nf = 32
        self.current_scale = 0

        self.sub_discriminators = nn.ModuleList()

        first_discriminator = nn.ModuleList()

        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=self.nf, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=self.nf, out_channels=self.nf*2, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=self.nf*2, out_channels=self.nf*4, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=self.nf*4, out_channels=self.nf*8, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=self.nf*8, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        first_discriminator.append(self.conv1)
        first_discriminator.append(self.conv2)
        first_discriminator.append(self.conv3)
        first_discriminator.append(self.conv4)
        first_discriminator.append(self.conv5)


        first_discriminator = nn.Sequential(*first_discriminator)

        self.sub_discriminators.append(first_discriminator)
        if init_weights:
            self.init_weights()

    def forward(self, x):
        out = self.sub_discriminators[self.current_scale](x)
        if self.use_sigmoid:
            outputs = torch.sigmoid(out)
        return outputs

    def progress(self):
        use_spectral_norm = True
        in_channels = 3
        self.current_scale += 1
        # Lower scale discriminators are not used in later ... replace append to assign?
        if self.current_scale % 4 == 0:
            self.nf *= 2

        tmp_discriminator = nn.ModuleList()
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=self.nf, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=self.nf, out_channels=self.nf * 2, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=self.nf * 2, out_channels=self.nf * 4, kernel_size=4, stride=2, padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=self.nf * 4, out_channels=self.nf * 8, kernel_size=4, stride=1, padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=self.nf * 8, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
        )

        tmp_discriminator.append(self.conv1)
        tmp_discriminator.append(self.conv2)
        tmp_discriminator.append(self.conv3)
        tmp_discriminator.append(self.conv4)
        tmp_discriminator.append(self.conv5)

        tmp_discriminator = nn.Sequential(*tmp_discriminator)

        if self.current_scale % 4 != 0:
            prev_discriminator = self.sub_discriminators[-1]

            # Initialize layers via copy
            if self.current_scale >= 1:
                tmp_discriminator.load_state_dict(prev_discriminator.state_dict())

        self.sub_discriminators.append(tmp_discriminator)
        print("DISCRIMINATOR PROGRESSION DONE")

'''
class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

'''
class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)