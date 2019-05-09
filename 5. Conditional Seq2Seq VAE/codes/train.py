import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import *
from utils import *
from dataloader import *
from models.vae import VAE 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def teacher_force_rate_schedule(epoch, num_epochs):
    teacher_forcing_ratio = 1 - sigmoid( (epoch - num_epochs//2) / (num_epochs//15))
    return teacher_forcing_ratio

# Ratio Scheduling
def KL_weight_schedule(epoch, num_epochs):
    period = num_epochs // 5
    epoch %= period
    KL_weight = sigmoid( (epoch - period//2) / (period//10))
    #KL_weight = (iter-10000) * 0.001
    KL_weight = max(0.01, KL_weight)
    KL_weight = min(1, KL_weight)
    return KL_weight 

def loss_fn(output, target, mean, logvar):
    num_classes = vocab_size
    # Cross Entropy Loss
    loss_fn_ce = nn.CrossEntropyLoss(reduction='mean')
    output = output.view(-1, num_classes)
    target = target.view(-1)
    CE_loss = loss_fn_ce(output, target)
    # KL Divergence
    KL_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return CE_loss, KL_loss


def train(input, condition, vae, vae_optimizer, teacher_forcing_ratio=0.5, KL_weight=0.001, max_length=MAX_LENGTH):
    vae_optimizer.zero_grad()
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    output, mean, logvar = vae(input, condition, use_teacher_forcing)
    CE_loss, KL_loss = loss_fn(output, input, mean, logvar)
    loss = CE_loss + KL_weight*KL_loss 
    
    loss.backward()
    vae_optimizer.step()
    return KL_loss.item(), CE_loss.item()

def trainEpochs(vae, data_loader, n_epochs=100, learning_rate=0.01, verbose=True):
    
    start = time.time()
    history = {"KL_loss":[], "CE_loss":[], "BLEU":[]}
    losses_total = {"KL_loss":0, "CE_loss":0} # Reset every print_every
    
    vae_optimizer     = optim.Adam(vae.parameters(), lr=learning_rate) # VAE
    
    # Pbar
    pbar = tqdm(total=n_epochs, unit=' epochs', ascii=True)
    
    for epoch in range(1, n_epochs+1):
        # Pbar
        pbar.set_description("({}/{})".format(epoch, n_epochs))
        
        teacher_forcing_ratio = teacher_force_rate_schedule(epoch, n_epochs)
        KL_weight = KL_weight_schedule(epoch, n_epochs)
        teacher_forcing_ratio = 0
        for i, (input, condition) in enumerate(data_loader):
            input = input.to(device)
            condition = condition.to(device)
            
            kl_loss, ce_loss = train(input, condition, vae, vae_optimizer, teacher_forcing_ratio=teacher_forcing_ratio, KL_weight=KL_weight)

            losses_total['KL_loss'] += kl_loss
            losses_total['CE_loss'] += ce_loss
        # Record Epoch Loss
        kl_loss_avg = losses_total['KL_loss'] / len(data_loader)
        ce_loss_avg = losses_total['CE_loss'] / len(data_loader)
        bleu = eval_model(vae)
        history['KL_loss'] += [kl_loss_avg]
        history['CE_loss'] += [ce_loss_avg]
        history['BLEU'] += [bleu]
        losses_total = {"KL_loss":0, "CE_loss":0}
        pbar.set_postfix({'CE_loss':history['CE_loss'][-1], 
                          'KL_loss':history['KL_loss'][-1], 'BLUE':history['BLEU'][-1]})
        pbar.update()
        if verbose and epoch % (num_epochs//10) == 0:
            word_idx = np.random.randint(0, len(words)-1, size=1).item()
            cond_idx = np.random.randint(0, 4-1, size=1).item()
            word = words[word_idx][cond_idx]
            input_tense = label2tense(cond_idx)
            predicts = []
            labels = []
            for i, target_tense in enumerate(['sp', 'tp', 'pg', 'p']):
                predict = test_model(vae, word, input_tense, target_tense)
                predicts.append(predict)
                labels.append(words[word_idx][i])
            print("| {:^10} | {:^10} | {:^10} | {:^10} | {:^10} |".format('', 'sp', 'tp', 'pg', 'p'))
            print("| {:^10} | {:^10} | {:^10} | {:^10} | {:^10} |".format(
                'Label', labels[0], labels[1], labels[2], labels[3]))
            print("| {:^10} | {:^10} | {:^10} | {:^10} | {:^10} |".format(
                'Predict', predicts[0], predicts[1], predicts[2], predicts[3]))
            print()
    pbar.close()
    return history
