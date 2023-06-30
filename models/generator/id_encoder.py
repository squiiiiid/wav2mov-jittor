import jittor as jt
from jittor import nn, Module

from models.layers.conv_layers import Conv2dBlock
from models.utils import get_same_padding


class IdEncoder(Module):
    def __init__(self):
        super(IdEncoder, self).__init__()

        #param
        in_channels = 3
        chs = [64, 128, 256, 512, 1024]
        chs = [in_channels] + chs + [1]# 1 is added here not in params because see how channels are being used in  id_decoder

        padding = get_same_padding(kernel_size=4,stride=2)
        self.conv_blocks = nn.ModuleList(Conv2dBlock(chs[i],chs[i+1],
                                                     kernel_size=(4,4),
                                                     stride=2,
                                                     padding=padding,
                                                     use_norm=True,
                                                     use_act=True,
                                                     act=nn.ReLU()
                                                     ) for i in range(len(chs)-2)
                                         )

        self.conv_blocks.append(Conv2dBlock(chs[-2],chs[-1],
                                            kernel_size=(4,4),
                                            stride=2,
                                            padding=padding,
                                            use_norm=False,
                                            use_act=True,
                                            act=nn.Tanh()))
    def execute(self,images):
        intermediates = []
        for block in self.conv_blocks[:-1]:
            images = block(images)
            intermediates.append(images)
        encoded = self.conv_blocks[-1](images)
        return encoded,intermediates
        