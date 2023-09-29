import torchvision
import torch
import torch.nn as nn

from ..base_model import BaseModel


class VGG11_bn(BaseModel):
    def __init__(self, layer_config, num_classes, activation, norm_layer, fine_tune, pretrained=True):
        super(VGG11_bn, self).__init__()

        # TODO: Initialize the different model parameters from the config file  #
        # for activation and norm_layer refer to cnn/model.py
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._layer_config = layer_config
        self._num_classes = num_classes

        self._activation = getattr(nn, activation["type"])(**activation["args"])
        self._norm_layer = getattr(nn, norm_layer["type"])
        self._fine_tune = fine_tune

        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._pretrained = pretrained
        self._build_model()

    def _build_model(self):
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the weights variable. Set it to None if  #
        # you want to train from scratch, 'DEFAULT'/'IMAGENET1K_V1' if you want to use  #
        # pretrained weights. You can either write it here manually or in config file   #
        # You can enable and disable training the feature extraction layers based on    # 
        # the fine_tune flag.                                                           #
        #################################################################################
        self.vgg11_bn = torchvision.models.vgg11_bn(pretrained=self._pretrained)
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)***** 

        if self._fine_tune:
            for param in self.vgg11_bn.parameters():
                param.requires_grad = False


        self.vgg11_bn.avgpool = nn.Identity()
        self.vgg11_bn.classifier = nn.Identity()


        self.features = self.vgg11_bn.features
        self.input_size = 512  #hardcoded for vgg11_bn

        layers = []
        layers.append(nn.Linear(self.input_size, self._layer_config[0]))
        layers.append(self._norm_layer(self._layer_config[0]))
        layers.append(self._activation)
        

        for i in range(1, len(self._layer_config)):
            layers.append(nn.Linear(self._layer_config[i-1], self._layer_config[i]))
            layers.append(self._norm_layer(self._layer_config[i]))
            layers.append(self._activation)
            
        
        layers.append(nn.Linear(self._layer_config[-1], self._num_classes))

        self.classifier = nn.Sequential(*layers)

        
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x