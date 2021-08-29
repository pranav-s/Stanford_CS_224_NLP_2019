#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 


class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab['<pad>']
        self.embed_size = embed_size
        self.vocab = vocab
        self.embeddings = nn.Embedding(len(vocab.char2id.keys()), embed_size, padding_idx=pad_token_idx)
        self.dropout_rate = 0.3
        # self.target = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=pad_token_idx)
        ### END YOUR CODE

    def forward(self, input: torch.tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        #batch_size = input.shape[1]
        x_emb = self.embeddings(input.permute([1,0,2]))
        max_word_size = x_emb.shape[2]
        CNN_network = CNN(self.embed_size, max_word_size)
        x_conv_out = CNN_network.forward(x_emb.permute([0,1,3,2]))
        highway_network = Highway(self.embed_size)
        x_highway = highway_network.forward(x_conv_out)
        dropout = nn.Dropout(p=self.dropout_rate)
        x_wordemb = dropout(x_highway)
        return x_wordemb.permute([1,0,2])





        ### END YOUR CODE

