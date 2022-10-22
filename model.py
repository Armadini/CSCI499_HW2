# IMPLEMENT YOUR MODEL CLASS HERE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class ArmansSuperDuperSkipGram(nn.Module):

    def __init__(self, embedding_dim, vocab_size):
        super(ArmansSuperDuperSkipGram, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        # The linear layer that maps from embedding to the possible context tokens
        self.embedding2context = nn.Linear(embedding_dim, vocab_size)


    def forward(self, word):
        embeds = self.embed(word)
        logits = self.embedding2context(embeds)
        return logits