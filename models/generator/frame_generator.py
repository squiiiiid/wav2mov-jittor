import jittor as jt
from jittor import nn, Module

from models.generator.audio_encoder import AudioEncoder
from models.generator.noise_encoder import NoiseEncoder
from models.generator.id_encoder import IdEncoder
from models.generator.id_decoder import IdDecoder

from models.utils import squeeze_batch_frames


class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        #param
        self.lr = 0.0002

        self.id_encoder = IdEncoder()
        self.id_decoder = IdDecoder()
        self.audio_encoder = AudioEncoder()
        # self.noise_encoder = NoiseEncoder(self.hparams)

    def execute(self,audio_frames,ref_frames):
        batch_size,num_frames,*_ = ref_frames.shape
        assert num_frames == audio_frames.shape[1]
        encoded_id , intermediates = self.id_encoder(squeeze_batch_frames(ref_frames))
        encoded_id = encoded_id.reshape(batch_size*num_frames,-1,1,1)
        encoded_audio = self.audio_encoder(audio_frames).reshape(batch_size*num_frames,-1,1,1)
        # encoded_noise = self.noise_encoder(batch_size,num_frames).reshape(batch_size*num_frames,-1,1,1)
        # logger.debug(f'encoded_id {encoded_id.shape} encoded_audio {encoded_audio.shape} encoded_noise {encoded_noise.shape}')
        encoded = jt.concat([encoded_id,encoded_audio],dim=1)#along channel dimension
        gen_frames =  self.id_decoder(encoded,intermediates)
        _,*img_shape = gen_frames.shape
        return gen_frames.reshape(batch_size,num_frames,*img_shape)

    def get_optimizer(self):
        return jt.optim.Adam(self.parameters(), lr=self.lr, betas=(0.5, 0.999))
