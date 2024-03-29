import jittor as jt
from jittor import nn, Module

from models.layers.conv_layers import Conv2dBlock
from models.utils import squeeze_batch_frames,get_same_padding


class AudioEncoder(Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        #param
        self.hidden_size = 256

        padding_31 = get_same_padding(3,1)
        padding_32 = get_same_padding(3,2)
        padding_42 = get_same_padding(4,2)
        # each frame has mfcc of shape (7,13)
        self.conv_encoder = nn.Sequential(
            Conv2dBlock(1,32,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU()),#7,13
            Conv2dBlock(32,32,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv2dBlock(32,32,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),  
              
            Conv2dBlock(32,64,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU()),    #7,13
            Conv2dBlock(64,64,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv2dBlock(64,64,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),  
              
            Conv2dBlock(64,128,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU()),    #7,13
            Conv2dBlock(128,128,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv2dBlock(128,128,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            
            Conv2dBlock(128,256,3,(2,1),padding=padding_32,use_norm=True,use_act=True,act=nn.ReLU()),  #4,13
            Conv2dBlock(256,256,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv2dBlock(256,256,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
           
            Conv2dBlock(256,512,(4,3),(2,1),padding=1,use_norm=True,use_act=True,act=nn.ReLU()),  #2,13
            Conv2dBlock(512,512,3,1,padding=padding_31,use_norm=True,use_act=True,act=nn.ReLU(),residual=True),    
            Conv2dBlock(512,512,(2,5),(1,2),padding=(0,1),use_norm=True,use_act=True,act=nn.ReLU()),#1,6,
            )
        
        self.features_len = 6*512#out of conv layers
        self.num_layers = 1
        self.gru = nn.GRU(input_size=self.features_len,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True)            
        self.final_act = nn.Tanh()
        
    def execute(self, x):
        """ x : audio frames of shape B,T,t,13"""
        batch_size,num_frames,*_ = x.shape
        x = self.conv_encoder(squeeze_batch_frames(x).unsqueeze(1))#add channel dimension
        x = x.reshape(batch_size,num_frames,self.features_len)
        x,_ = self.gru(x)
        return self.final_act(x)  # shape (batch_size,num_frames,hidden_size=latent_dim_audio)
