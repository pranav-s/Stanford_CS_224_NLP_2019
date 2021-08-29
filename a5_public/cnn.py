#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch

class CNN(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, max_word_size):
        super(CNN, self).__init__()
        self.kernel_size = 5
        self.relu = nn.ReLU()
        self.conv1d = nn.Conv1d(in_channels=embed_size, out_channels= embed_size, kernel_size=self.kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=max_word_size-self.kernel_size+1)

    def forward(self, x_reshaped : torch.tensor):
        # Will have to split all batches in a for loop and operate on each batch separately and then collate the output
        # and return it
        x_out_final = []
        for x_reshaped_batch in torch.split(x_reshaped, 1, dim=0):
            x_reshaped_batch = x_reshaped_batch.squeeze(0)
            x_conv = self.conv1d(x_reshaped_batch)
            x_conv_nonlinearity = self.relu(x_conv)
            x_out = self.maxpool(x_conv_nonlinearity)
            x_out_final.append(x_out.squeeze(2))
        x_out_final = torch.stack(x_out_final, dim=0)
        return x_out_final
### END YOUR CODE

