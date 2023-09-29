import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super(ConvNet, self).__init__()

        ######################################################################################################
        # TODO: Initialize the different model parameters from the config file                               #    
        # You can use the arguments given in the constructor. For activation and norm_layer                  #
        # to make it easier, you can use the following two lines                                             #                              
        #   self._activation = getattr(nn, activation["type"])(**activation["args"])                         #        
        #   self._norm_layer = getattr(nn, norm_layer["type"])                                               #
        # Or you can just hard-code using nn.Batchnorm2d and nn.ReLU as they remain fixed for this exercise. #
        ###################################################################################################### 
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.drop_prob = drop_prob
        self._activation = getattr(nn, activation["type"])(**activation["args"])
        self._norm_layer = getattr(nn, norm_layer["type"])
        self._dropout = nn.Dropout(p = self.drop_prob)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layers.append(nn.Conv2d(in_channels = self.input_size, out_channels = self.hidden_layers[0], kernel_size=3, stride=1, padding=1))
        layers.append(self._norm_layer(self.hidden_layers[0]))
        layers.append(nn.MaxPool2d(kernel_size=2, stride = 2))
        layers.append(self._activation)
        layers.append(self._dropout)
        
        for i in range(1, len(self.hidden_layers)-1):
            layers.append(nn.Conv2d(in_channels = self.hidden_layers[i-1], out_channels = self.hidden_layers[i], kernel_size=3, stride=1, padding=1))
            layers.append(self._norm_layer(self.hidden_layers[i]))
            layers.append(nn.MaxPool2d(kernel_size=2, stride = 2))
            layers.append(self._activation)
            layers.append(self._dropout)
            


        # layers.append(nn.Linear(self.hidden_layers[-1], self.num_classes))
        # layers.append(self._activation)

        self.layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(self.hidden_layers[-1], 10)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        conv1_weights = self.layers[0].weight.data
        filters = conv1_weights.view(128, 3, 3, 3).cpu()

        grid_image = np.zeros((40, 80, 3))
        pad_width = ((1, 1), (1, 1), (0, 0))
        for i in range(8):
            for j in range(16):
                filter_image = filters[i * 3 + j]
                padded_matrix = np.pad(self._normalize(filter_image.numpy()), pad_width, mode='constant')
                grid_image[i * 5: (i + 1) * 5, j * 5: (j + 1) * 5, :] = padded_matrix
                
        plt.figure(figsize=(10, 20))
        plt.imshow(grid_image)
        plt.axis('off')
        plt.show()


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x
