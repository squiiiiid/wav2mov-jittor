import jittor as jt
from jittor import nn, Module

from core.models.base_model import BaseModel
from models.layers.conv_layers import Conv2dBlock
from models.utils import get_same_padding,squeeze_batch_frames


class SequenceDiscriminator(BaseModel):
    def __init__(self):
        super(SequenceDiscriminator, self).__init__()

        #param
        self.lr = 0.0001
        relu_neg_slope = 0.01
        in_size, h_size, num_layers = 32, 256, 1
        in_channels = 3
        chs = [in_channels] + [64, 128, 256, 512, 1]

        self.gru = nn.GRU(input_size=in_size,hidden_size=h_size,num_layers=num_layers,batch_first = True)
        kernel,stride = 4,2
        padding = get_same_padding(kernel,stride)
        cnn = nn.ModuleList([Conv2dBlock(chs[i],chs[i+1],kernel,stride,padding,
                                         use_norm=True,use_act=True,
                                         act=nn.LeakyReLU(relu_neg_slope)) for i in range(len(chs)-2)])
        cnn.append(Conv2dBlock(chs[-2],chs[-1],kernel,stride,padding,
                               use_norm=False,use_act=True,
                               act=nn.Tanh()))
        self.cnn = nn.Sequential(*cnn)
        ############################################
        # channels : 3  => 64 => 128 => 256 => 512          
        # frame sz : 256=> 128 => 64  =>  32 => 16 =>8 = Width and height =4 (since only upper half)        
        # height is half so : final height is 4
        # thus out of self.cnn : 512x4x8       
        ############################################

    def execute(self, frames):
        """frames : B,T,C,H,W"""
        img_height = frames.shape[-2]
        frames = frames[...,0:img_height//2,:]#consider upper half
        batch_size, num_frames, *img_size = frames.shape
        frames = squeeze_batch_frames(frames)
        frames = self.cnn(frames)
        frames = frames.reshape(batch_size,num_frames,-1)
        out,_ = self.gru(frames)#out is of shape (batch_size,seq_len,num_dir*hidden_dim)
        return out[:,-1,:]#batch_size,hidden_size

    def get_optimizer(self):
        return jt.optim.Adam(self.parameters(), lr=self.lr, betas=(0.5,0.999))
