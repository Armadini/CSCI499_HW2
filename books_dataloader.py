import json
from black import out
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import preprocess_string, build_output_tables, build_tokenizer_table


def flatten_and_clean(data):
    # Flatten into one long list of shape: (instruction, (action, object))
    return [(preprocess_string(instruction), a_o)
            for sublist in data for (instruction, a_o) in sublist]


class EncodedSentenceDataset(Dataset):

    def __init__(self, encoded_sentences, sentence_lens, context_window_size: int):
        """Initializes a dataset from encoded sentences"""
        self.context_window_size = context_window_size
        self.sentences = encoded_sentences
        self.sentence_lens = sentence_lens
        self.num_sentences = encoded_sentences.shape[0]
        self.length = np.sum(sentence_lens) - context_window_size*2*self.num_sentences
        self.adj_sentence_len = encoded_sentences.shape[1] - context_window_size*2

    def __getitem__(self, index):
        # Get an index of the dataset
        sentence_num, count = 0, 0
        while True:
            count += self.sentence_lens[sentence_num, 0] - self.context_window_size*2
            if count >= index: 
                count -= self.sentence_lens[sentence_num, 0] - self.context_window_size*2
                break
            sentence_num += 1
        adj_index = index - count + self.context_window_size
        x = self.sentences[sentence_num, adj_index]
        yl = self.sentences[sentence_num, adj_index-self.context_window_size:adj_index]
        yr = self.sentences[sentence_num, adj_index+1:adj_index+self.context_window_size+1]
        # print("YL", yl)
        # print("YR", yr)
        # print("YL+YR", np.concatenate([yl, yr]))
        return x, np.concatenate([yl, yr])
        

    def __len__(self):
        # Number of instances
        return self.length

def get_loaders(encoded_sentences, lens, context_window_size, batch_size=4, shuffle=False):
    train_dataset = EncodedSentenceDataset(encoded_sentences, lens, context_window_size)
    valid_dataset = EncodedSentenceDataset(encoded_sentences, lens, context_window_size)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, valid_dataloader
