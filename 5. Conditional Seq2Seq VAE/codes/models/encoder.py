import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, cond_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.cond_size = cond_size
        self.cond_embedding = nn.Embedding(cond_size, 8)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
    
    def forward(self, input, input_cond, hidden=None):
        if hidden is None:
            hidden = self.initHidden(input_cond)
        batch_size, seq_len = input.size(0), input.size(1)
        embedded = self.embedding(input).view(batch_size, seq_len, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, cond_tensor):
        batch_size = cond_tensor.size(0)
        h0 = torch.zeros(batch_size, self.hidden_size-8).to(device)
        cond_embedding = self.cond_embedding(cond_tensor).view(batch_size, 8).to(device)
        hidden = torch.cat((h0, cond_embedding), dim=1).view(1, batch_size, self.hidden_size).to(device)
        return hidden


def test_encoder():
    input_size = 28
    cond_size = 4
    hidden_size = 32

    batch_size = 32
    seq_length = 10

    encoder = EncoderRNN(input_size, hidden_size, cond_size).to(device)
    input = torch.randint(0, input_size, (batch_size, seq_length), dtype=torch.long).to(device)
    condition = torch.randint(0, cond_size, (batch_size, ), dtype=torch.long).to(device)
    output, hidden = encoder(input, condition)

    print(output.shape)
    print('output: (batch_size, sequence_length, hidden_size)')
    print(hidden.shape) 
    print('hidden: (num_layers, batch_size, hidden_size)')

if __name__ == '__main__':
    test_encoder()