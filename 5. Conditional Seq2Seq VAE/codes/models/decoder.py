import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.initHidden(input.size(0))
        batch_size, seq_len = input.size(0), input.size(1)
        embedded = self.embedding(input).view(batch_size, seq_len, -1)
        output = F.relu(embedded)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


def test_decoder():
    output_size = 28
    hidden_size = 256

    batch_size = 32
    seq_length = 10

    decoder = DecoderRNN(hidden_size, output_size).to(device)
    input = torch.randint(0, output_size, (batch_size, seq_length), dtype=torch.long).to(device)
    output, hidden = decoder(input)

    print(output.shape)
    print('output: (batch_size, sequence_length, output_size)')
    print(hidden.shape) 
    print('hidden: (num_layers, batch_size, hidden_size)')

if __name__ == '__main__':
    test_decoder()