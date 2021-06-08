import os
import torch
from torch import nn
import torchvision
from src.model import PlainResnet50, CustomResnet50, PlainEfficientnetB4, PlainEfficientnetB5, PlainEfficientnetB7
import IPython

class CallModel():
    def __init__(self, model_type=None, pretrained=True, logger=None, path='./pretrained_model'):
        
        # MODEL TYPE
        if model_type == 'plain_resnet50':
            base_model = PlainResnet50()
            weight_path = os.path.join(path, 'plain_resnet50_ckpt.pth')
            
        elif model_type == 'custom_resnet50':
            base_model = CustomResnet50()
            weight_path = os.path.join(path, 'custom_resnet50_ckpt.pth')
            
        elif model_type == 'plain_efficientnetb4':
            base_model = PlainEfficientnetB4()
            weight_path = os.path.join(path, 'plain_efficientnetb4_ckpt.pth')
            
        elif model_type == 'plain_efficientnetb5':
            base_model = PlainEfficientnetB5()
            weight_path = os.path.join(path, 'plain_efficientnetb5_ckpt.pth')
            
        elif model_type == 'plain_efficientnetb7':
            base_model = PlainEfficientnetB7()
            weight_path = os.path.join(path, 'plain_efficientnetb7_ckpt.pth')
            
        else:
            raise Exception(f"No such model type: {model_type}")
        
        
        # LOAD PRETRAINED WEIGHTS
        if pretrained:
            logger.info(f"Using pretrained model. Loading weights from {weight_path}")
            base_model = CallModel._load_weights(base_model, weight_path)
            
            # b5 model
            nn.init.xavier_normal_(base_model.block[0]._fc.weight)
            #IPython.embed(); exit(1)
           
        else:
            logger.info(f"Not using pretrained model.")
        
        self.model = base_model
            
    def model_return(self):
        return self.model
        
    @staticmethod
    def _load_weights(model, path):
        model.load_state_dict(torch.load(path))
        return model