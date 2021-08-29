#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn
import torch


class Highway(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size):
        super(Highway, self).__init__()
        self.embed_size = embed_size
        self.linear_layer1 = nn.Linear(embed_size, embed_size)
        self.linear_layer2 = nn.Linear(embed_size, embed_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x_conv_out):
        x_proj = self.relu(self.linear_layer1(x_conv_out))
        x_gate = self.softmax(self.linear_layer2(x_conv_out))
        x_highway = x_proj*x_gate + (torch.ones(self.embed_size)-x_gate)*x_proj
        return x_highway

### END YOUR CODE 

