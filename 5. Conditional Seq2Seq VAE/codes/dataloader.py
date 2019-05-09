# Package
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from os import system

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import torch.utils.data as data
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0 # Start Of Sentence token
EOS_token = 1 # End Of Sentence token
MAX_LENGTH = 16
vocab_size = 28

# Prepare Data
def prepare_data():
    words = []
    tenses = []
    with open("data/train.txt") as f:
        for line in f:
            words.append(line.split('\n')[0].split(' '))
            tenses.append([label2tense(l) for l in LABELS])
    return words, tenses

def tensorsFromWord(word, tense=None):
    indexes = [char2num(char) for char in word]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensorFromTense(tense):
    index = tense2label(tense)
    return torch.tensor([index], dtype=torch.long).view(-1, 1)

# Data conversion
# simple present(sp), third person(tp), present progressive(pg), simple past(p).
TENSES = ['sp', 'tp', 'pg', 'p']
LABELS = [i for i in range(len(TENSES ))]
def tense2label(tense):
    t2l = dict(zip(TENSES , LABELS))
    return t2l[tense]
    
def label2tense(label):
    l2t = dict(zip(LABELS, TENSES ))
    return l2t[label]

CHARS = ['SOS', 'EOS'] + [chr(i) for i in range(ord('a'), ord('z')+1)]
NUMS = [i for i in range(len(CHARS))]
def char2num(char):
    c2n = dict(zip(CHARS, NUMS))
    if char == 'SOS':
        return 0
    elif char == 'EOS':
        return 1
    else:
        return c2n[char.lower()]
    
def num2char(num):
    n2c = dict(zip(NUMS, CHARS))
    return n2c[num]

# Batch Training
class MyData(data.Dataset):
    def __init__(self):
        words, tenses = prepare_data()
        self.words = words
        self.tenses = tenses
        
    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        tense_index_input = random.randint(0, 4-1)
        input       = self.words[idx][tense_index_input]
        input_cond  = self.tenses[idx][tense_index_input]
        input_tensor = tensorsFromWord(input)
        input_cond_tensor = tensorFromTense(input_cond)
        return input_tensor, input_cond_tensor

def collate_fn(data):
    batch_size = len(data)
    input_tensor = [data[i][0] for i in range(batch_size)]
    input_cond_tensor = torch.LongTensor([data[i][1] for i in range(batch_size)])
    input_tensor = torch.LongTensor(rnn_utils.pad_sequence(input_tensor, batch_first=True, padding_value=EOS_token))
    return input_tensor, input_cond_tensor
