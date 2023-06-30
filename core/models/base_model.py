from abc import abstractmethod
import jittor as jt
from jittor import nn, Module

import logging 
logger = logging.getLogger(__name__)

class BaseModel(Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    
    def save_to(self,checkpoint_fullpath):
        jt.save(self,checkpoint_fullpath)
        logger.log(f'Model saved at {checkpoint_fullpath}','INFO')
    
    def load_from(self,checkpoint_fullpath):
        try:
             self.load_statedict(jt.load(checkpoint_fullpath))
        except:
            logger.log(f'Cannot load checkpoint from {checkpoint_fullpath}',type="ERROR")
        
    @abstractmethod
    def execute(self,*args):
        raise NotImplementedError(f'Forward method is not defined in {self.__class__.__name__}')
    
    def freeze_learning(self):
        for p in self.parameters():
          p.require_grad = False



