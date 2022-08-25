from types import LambdaType
import torch.nn as nn
from torch import Tensor

class Model(nn.Module):
    def __init__(self,name):
        super().__init__()
        self.name = name    
        
        
class TrainedModel(Model):
    def __init__(self,name,model_init:LambdaType,transformation=None,acc1=None,acc5=None):
        super().__init__(name)
        self._net = None
        self.model_init = model_init
        self.transformation = transformation
        self.acc1 = acc1
        self.acc5 = acc5
        
    @property
    def net(self):
        if self._net is None:
            self._net = self.model_init()
        return self._net
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.net(x)
    
    
    
""" model_pool """     
from models.torchvision_models import model_pool