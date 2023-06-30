import jittor as jt
from jittor import nn, Module

from models.utils import squeeze_batch_frames,get_same_padding
from models.layers.conv_layers import Conv2dBlock

class IdentityDiscriminator(Module):
    def __init__(self):
        super(IdentityDiscriminator, self).__init__()

        #param
        self.lr = 0.0001
        relu_neg_slope = 0.01
        in_channels = 6
        chs = [64, 128, 256, 512, 1024, 1]
        chs = [in_channels] + chs

        padding = get_same_padding(kernel_size=4,stride=2)
        self.conv_blocks = nn.ModuleList(Conv2dBlock(chs[i],chs[i+1],
                                                     kernel_size=(4,4),
                                                     stride=2,
                                                     padding=padding,
                                                     use_norm=True,
                                                     use_act=True,
                                                     act=nn.LeakyReLU(relu_neg_slope)
                                                     ) for i in range(len(chs)-2)
                                         )

        self.conv_blocks.append(Conv2dBlock(chs[-2],chs[-1],
                                            kernel_size=(4,4),
                                            stride=2,
                                            padding=padding,
                                            use_norm=False,
                                            use_act=False
                                            )
                                )

    def execute(self,x,y):
        """
        x : frame image (B,F,H,W)
        y : still image
        """
        assert x.shape==y.shape

        if len(x.shape)>4:#frame dim present
            x = squeeze_batch_frames(x)
            y = squeeze_batch_frames(y)
        
        x = jt.concat([x,y],dim=1)#along channels
        for block in self.conv_blocks:
          x = block(x)
        return x
    
    def get_optimizer(self):
        return jt.optim.Adam(self.parameters(), lr=self.lr, betas=(0.5,0.999))
