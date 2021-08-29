#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = len(target_vocab.char2id)
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, self.vocab_size)
        self.decoderCharEmb = nn.Embedding(self.vocab_size, char_embedding_size)
        self.target_vocab = target_vocab
        

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.

        # Split batch
        # For each elem of batch, lookup char embedding
        # Pass each vector through LSTM and linear layer to generate output
        # Concerns - What does length correspond to here ?
        batch_size = input.shape[1]
        if dec_hidden is None:
            dec_hidden = (torch.rand((1, batch_size, self.hidden_size)), torch.rand((1, batch_size, self.hidden_size)))
        # dec_hidden, dec_cell = dec_hidden
        # dec_hidden = dec_hidden.squeeze(0)
        # dec_cell = dec_cell.squeeze(0)
        # dec_hidden_final = []
        # dec_cell_final = []
        # s_t = []
        # for i, X_b in enumerate(torch.split(input, 1, dim=1)):
        #     X_emb= self.decoderCharEmb(X_b.squeeze(0))
        #     dec_hidden_b = (dec_hidden[:,i,:].unsqueeze(0), dec_cell[:,i,:].unsqueeze(0))
        #     # s_bt = []
        #     dec_out, (dec_hid_b, dec_cell_b) = self.charDecoder(X_emb, dec_hidden_b)
        #     # dec_hidden = (dec_hidden_b, dec_cell_b)
        #     # for x_t in torch.split(X_emb, 1,  dim=0): # Need to confirm dimension
        #     #     dec_state_t = self.charDecoder(x_t, dec_state_t)
        #     #     dec_hidden_t, dec_cell_t = dec_state_t
        #     #     s_bt.append(self.char_output_projection(dec_hidden_t).view(self.vocab_size))
        #     # s_t.append(torch.stack(s_bt, dim=0))
        #     s_t.append(self.char_output_projection(dec_out).squeeze(1))
        #     dec_hidden_final.append(dec_hid_b.view(self.hidden_size))
        #     dec_cell_final.append(dec_cell_b.view(self.hidden_size))

        X_emb = self.decoderCharEmb(input)
        dec_out, dec_hidden = self.charDecoder(X_emb, dec_hidden)
        scores = self.char_output_projection(dec_out)

        # scores = torch.stack(s_t, dim=0).permute([1, 0, 2])
        # dec_hidden_final = torch.stack(dec_hidden_final, dim=0).unsqueeze(0)
        # dec_cell_final = torch.stack(dec_cell_final, dim=0).unsqueeze(0)
        # dec_hidden = (dec_hidden_final, dec_cell_final)
        return scores, dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        # Input to forward in :-1
        batch_size = char_sequence.shape[1]
        length = char_sequence.shape[0]
        scores, _ = self.forward(char_sequence[:-1,:], dec_hidden)
        target = char_sequence[1:,:].view((length-1)*batch_size)
        scores_flatten = scores.view((length-1)*batch_size, self.vocab_size)
        # Compute cross entropy loss
        loss = nn.CrossEntropyLoss()
        ce_loss = loss(scores_flatten, target)

        return ce_loss


        ### END YOUR CODE


    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        dec_hidden, dec_cell = initialStates
        batch_size = dec_hidden.shape[1]
        output_words = []
        for i in range(batch_size):
            output_word = ''
            X_emb = self.decoderCharEmb(torch.tensor([[self.target_vocab.start_of_word]], device=device))
            dec_hidden_b = (dec_hidden[:,i,:].unsqueeze(0), dec_cell[:,i,:].unsqueeze(0))

            for j in range(max_length):
                dec_out, (dec_hid_b, dec_cell_b) = self.charDecoder(X_emb, dec_hidden_b)
                dec_hidden_b = dec_hid_b, dec_cell_b
                s_t = self.char_output_projection(dec_hid_b).view(self.vocab_size)
                next_index = int(torch.argmax(s_t))
                next_char = self.target_vocab.id2char[next_index]
                output_word += next_char
                X_emb = self.decoderCharEmb(torch.tensor([[next_index]], device=device))
                # find next char and its embedding, construct word
            if '}' in output_word:
                end_index = output_word.index('}')
                output_word = output_word[:end_index]
            output_words.append(output_word)
            # print(output_words)

        return output_words


        
        ### END YOUR CODE

