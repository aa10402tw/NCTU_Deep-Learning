import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .encoder import *
from .decoder import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0 # Start Of Sentence token
EOS_token = 1 # End Of Sentence token
MAX_LENGTH = 16
max_length = 16

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, cond_size, output_size):
        super(VAE, self).__init__()
        
        # Encoder & Decoder
        self.encoder = EncoderRNN(input_size, hidden_size, cond_size)
        self.decoder = DecoderRNN(hidden_size, output_size)
        
        # Hidden to Latent (REPARAMETERIZATION)
        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2logvar = nn.Linear(hidden_size, latent_size)
        
        # Latent to Hidden
        self.latent2hidden = nn.Linear(latent_size+8, hidden_size)
        self.cond_embedding = nn.Embedding(cond_size, 8)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.cond_size = cond_size
        self.output_size = output_size
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        latent = mean + eps*std
        return latent
    
    def generation(self, latent, condition):
        batch_size = 1
        hidden = self.latent_cond2hidden(latent, condition)
        #----------sequence to sequence part for decoder----------#
        outputs = Variable(torch.zeros(batch_size, MAX_LENGTH, self.output_size)).to(device)
        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).view(batch_size, 1, 1)
        decoder_hidden = hidden
        # Without teacher forcing: use its own predictions as the next input
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, di, :] = decoder_output.view(batch_size, self.output_size)
            # next input
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view(batch_size, 1, 1)  # detach from history as input
        return outputs
    
    def inference(self, input, input_condition, target_condition):
        #----------sequence to sequence part for encoder----------#
        batch_size, seq_length = input.size(0), input.size(1)
        encoder_output, encoder_hidden = self.encoder(input, input_condition, None)
        
        #---------- sequence to sequence part for VAE----------#
        mean = self.hidden2mean(encoder_hidden)
        logvar = self.hidden2logvar(encoder_hidden)
        latent = self.reparameterize(mean, logvar)
        hidden = self.latent_cond2hidden(latent, target_condition)
        
        #----------sequence to sequence part for decoder----------#
        outputs = Variable(torch.zeros(batch_size, MAX_LENGTH, self.output_size)).to(device)
        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).view(batch_size, 1, 1)
        decoder_hidden = hidden
        # Without teacher forcing: use its own predictions as the next input
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, di, :] = decoder_output.view(batch_size, self.output_size)
            # next input
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view(batch_size, 1, 1)  # detach from history as input
        return outputs, mean, logvar

    def forward(self, input, condition, use_teacher_forcing=True):
        #----------sequence to sequence part for encoder----------#
        batch_size, seq_length = input.size(0), input.size(1)
        encoder_output, encoder_hidden = self.encoder(input, condition, None)
        
        #---------- sequence to sequence part for VAE----------#
        mean = self.hidden2mean(encoder_hidden)
        logvar = self.hidden2logvar(encoder_hidden)
        latent = self.reparameterize(mean, logvar)
        hidden = self.latent_cond2hidden(latent, condition)
        
        #----------sequence to sequence part for decoder----------#
        batch_size, seq_length = input.size(0), input.size(1)
        outputs = Variable(torch.zeros(batch_size, seq_length, self.output_size)).to(device)
        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device).view(batch_size, 1, 1)
        decoder_hidden = hidden
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(seq_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, di, :] = decoder_output.view(batch_size, self.output_size)
                # next input
                decoder_input = input[:, di].view(batch_size, 1, 1)
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(seq_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, di, :] = decoder_output.view(batch_size, self.output_size)
                # next input
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().view(batch_size, 1, 1)  # detach from history as input
        return outputs, mean, logvar
    
    def latent_cond2hidden(self, latent, cond_tensor):
        batch_size = cond_tensor.size(0)
        cond_tensor = cond_tensor.to(device)
        latent = latent.view(batch_size, self.latent_size).to(device)
        cond = self.cond_embedding(cond_tensor).view(batch_size, 8).to(device)
        latent_cond = torch.cat((latent, cond), dim=1).view(1, batch_size, self.latent_size+8) 
        hidden_out = self.latent2hidden(latent_cond)
        return hidden_out


def test_vae():
    input_size = 28
    cond_size = 4
    hidden_size = 32
    latent_size = 4
    output_size = 28

    batch_size = 32
    seq_length = 10

    vae = VAE(input_size, hidden_size, latent_size, cond_size, output_size).to(device)
    input = torch.randint(0, input_size, (batch_size, seq_length), dtype=torch.long).to(device)
    condition = torch.randint(0, cond_size, (batch_size, ), dtype=torch.long).to(device)
    outputs, mean, logvar = vae(input, condition)

    print(outputs.shape)
    print('output: (batch_size, sequence_length, output_size)')
    print(mean.shape, logvar.shape) 
    print('mean/logvar: (num_layers, batch_size, latent_size)')

if __name__ == '__main__':
    test_vae()
