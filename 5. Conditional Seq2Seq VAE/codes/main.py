import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import *
from utils import *
from train import *
from models.vae import VAE 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0 # Start Of Sentence token
EOS_token = 1 # End Of Sentence token
MAX_LENGTH = 16
max_length = 16

if __name__ =='__main__':
    #----------Hyper Parameters----------#
    hidden_size = 256
    cond_size = 4
    latent_size = 32
    vocab_size = 28 #The number of vocabulary

    vae = VAE(vocab_size, hidden_size, latent_size, cond_size, vocab_size).to(device)

    words, tenses = prepare_data()
    data = MyData()
    data_loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=collate_fn)

    history = trainEpochs(vae, data_loader, n_epochs=5000, learning_rate=0.001, verbose=False)
    save_model(vae, model_name='vae_5000')