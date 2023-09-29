import torch
import numpy as np

class TopKAccuracy:
    def __init__(self, k=1):
        self.k = k

    def compute(self, output, target):
        correct = 0
        ####################################################
        # TODO 
        # given an output of shape (N, C) and target labels of shape (N),
        # Compute the topK accuracy, where a "correct" classification is considered
        # when the target can be found in top-K (e.g. Top-1 or Top-5) classes.
        # Top-1 would be what's often referred to as "Accuracy".
        ####################################################
        
        #Numpy Implementation
        top_k_preds = np.fliplr(np.argsort(output.detach().cpu().numpy(), axis=1))[:, :self.k]

        for i in range(len(top_k_preds)):
            if np.isin(target.detach().cpu().numpy()[i], top_k_preds[i]):
                correct +=1

        return correct / len(target)

    def __str__(self):
        return f"top{self.k}"