# Package
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import random
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from dataloader import *

# Test Model
def test_model(vae, word='', input_tense='', target_tense=''):
    output = []
    batch_size = 1
    input = tensorsFromWord(word).view(batch_size, -1, 1).to(device)
    input_condition = tensorFromTense(input_tense).view(batch_size, 1, 1).to(device)
    target_condition = tensorFromTense(target_tense).view(batch_size, 1, 1).to(device)
    vae.eval()
    with torch.no_grad():
        outputs, mean, logvar = vae.inference(input, input_condition, target_condition)
    vae.train()

    predictions = []
    for i in range(outputs.size(1)):
        output = outputs[0][i].data.cpu()
        topv, topi = output.topk(1)
        prediction = topi.item() 
        if prediction == EOS_token:
            break
        predictions.append(num2char(prediction))
    return ''.join(predictions)

# Generation
def generation(vae, z, tense):
    target_condition = tensorFromTense(tense).view(1, 1, 1).to(device)
    vae.eval()
    with torch.no_grad():
        outputs, mean, logvar = vae.latent_cond2hidden(z, condition)
    vae.train()

    predictions = []
    for i in range(outputs.size(1)):
        output = outputs[0][i].data.cpu()
        topv, topi = output.topk(1)
        prediction = topi.item() 
        if prediction == EOS_token:
            break
        predictions.append(num2char(prediction))
    return ''.join(predictions)

# Save / Load Model
def safe_make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_model(model, model_name='test.pth'):
    if '.' not in model_name:
        model_name += '.pth'
    safe_make_dir('weights')
    path = os.path.join('weights', model_name)
    torch.save(model, path)
    
def load_model(model_name='test.pth'):
    if '.' not in model_name:
        model_name += '.pth'
    path = os.path.join('weights', model_name)
    model = torch.load(path)
    return model

# Draw History
def draw_history(history):
    kl_loss = history["KL_loss"]
    ce_loss = history["CE_loss"]
    bleu = history["BLEU"]
    xs = [i for i in range(1, len(kl_loss)+1)]
    plt.plot(xs, ce_loss, label='CE Loss')
    plt.plot(xs, kl_loss, label='KL Loss')
    plt.plot(xs, bleu, label='BLEU')
    plt.legend(loc='best')
    
def draw_ratio(n_iters=50000):
    xs = [i for i in range(1, n_iters+1)]
    tf_ratio = []
    kl_weight = []
    for x in xs:
        tf_ratio.append(teacher_force_rate_schedule(x, n_iters))
        kl_weight.append(KL_weight_schedule(x, n_iters))
    plt.plot(xs, tf_ratio, label='tf_ratio')
    plt.plot(xs, kl_weight, label='kl_weight')
    plt.legend(loc='best')

def compute_bleu(output, reference):
    cc = SmoothingFunction()
    return sentence_bleu([reference], output,weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=cc.method1)

def eval_model(vae, verbose=False):
    input_tenses = ['sp', 'sp', 'sp', 'sp','p', 'sp', 'p', 'pg', 'pg', 'pg']
    target_tenses = ['p', 'pg', 'tp', 'tp', 'tp', 'pg', 'sp', 'sp', 'p', 'tp']
    inputs = []
    targets = []
    with open('data/test.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split('\n')[0].split(' ')
            inputs.append(data[0])
            targets.append(data[1])
            
    score_sum = 0
    if verbose:
        print("{:^6} | {:^10} | {:^10}".format("BLEU", "Prediction", "Target"))
        print('-'*30)
    for i in range(len(inputs)):
        prediction = test_model(vae, word=inputs[i], input_tense=input_tenses[i], target_tense=target_tenses[i])
        score = compute_bleu(prediction, targets[i])
        if verbose:
            print("{:6} | {:10} | {:10}".format('%.4f'%score, prediction, targets[i]))
        score_sum += score
    if verbose:
        print('-'*30)
        print("Avg:", score_sum / 10)
    return score_sum / 10