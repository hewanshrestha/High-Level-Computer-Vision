import torch
import torchvision
import torch.nn as nn


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """
        Load a pretrained ResNet-152 and modify top layers to extract features
        """
        super(EncoderCNN, self).__init__()
        
        resnet = torchvision.models.resnet152(pretrained=True)
        #########################
        # TODO 
        # Create a sequential model (named `self.resnet`) with all the layers of resnet except the last fc layer.
        # Add a linear layer (named `self.linear`) to bring resnet features down to embed_size. Don't put the self.linear into the Sequential module.
        #########################
        # Create a sequential model with all the layers of resnet except the last fc layer.
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Add a linear layer to bring resnet features down to embed_size. Don't put the self.linear into the Sequential module.
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

        self.bn = nn.BatchNorm1d(embed_size)

        # Freeze the weights of the resnet layers
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        """Extract feature vectors from input images."""
        #########################
        # TODO 
        # Run your input images through the modules you created above (input -> Sequential -> final linear -> self.bn)
        # Make sure to freeze the weights of the resnet layers
        # finally return the normalized features
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        features = features / torch.norm(features, p=2, dim=1, keepdim=True)
        return features
        #########################