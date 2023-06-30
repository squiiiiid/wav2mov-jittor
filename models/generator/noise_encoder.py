import jittor as jt
from jittor import nn, Module


class NoiseEncoder(Module):
    def __init__(self,hparams):
        super(NoiseEncoder, self).__init__()
        self.hparams = hparams
        self.features_len = 10
        self.hidden_size = self.hparams['latent_dim_noise']
        self.gru = nn.GRU(input_size=self.features_len,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=True)
        #input should be of shape batch_size,seq_len,input_size
    def execute(self,batch_size,num_frames):
        noise = jt.randn(batch_size,num_frames,self.features_len)
        out,_ = self.gru(noise)
        return out#(batch_size,seq_len,hidden_size)
