# Package
import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm as tqdm

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def to_categorical(y, num_classes):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_classes))
    y_cat[range(y.shape[0]), y] = 1.0
    return Variable(FloatTensor(y_cat))

class Generator(nn.Module):
    def __init__(self, img_size=64, img_channels=1, latent_dim=52, cat_dim=10, cont_dim=2):
        super(Generator, self).__init__()
        
        def deconv_block(in_channels, out_channels, kernel_size=(4,4), stride=(1,1), padding=0):
            block = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
            return block
            
        self.deconv_blocks = nn.Sequential(
            *deconv_block(64 , 512, kernel_size=(4, 4), stride=(1, 1), padding=0), 
            *deconv_block(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=1), 
            *deconv_block(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=1), 
            *deconv_block(128, 64 , kernel_size=(4, 4), stride=(2, 2), padding=1), 
            nn.ConvTranspose2d( 64,   1, kernel_size=(4, 4), stride=(2, 2), padding=(1,1), bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, cat_code, cont_code):
        batch_size = noise.size(0)
        gen_input = torch.cat((noise, cat_code, cont_code), -1).view(batch_size, -1, 1, 1)
        img = self.deconv_blocks(gen_input)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size=64, latent_dim=52, cat_dim=10, cont_dim=2):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, bn=True):
            block = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        self.conv_blocks = nn.Sequential(
            *discriminator_block(1, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )
        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.discriminator = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=(4,4), stride=(1, 1), bias=False),
            nn.Sigmoid()
        )
        self.Q = nn.Sequential(nn.Linear(512 * ds_size ** 2, cat_dim+cont_dim))

    def forward(self, img):
        batch_size = img.size(0)
        out = self.conv_blocks(img)
        validity = self.discriminator(out).view(batch_size, -1)
        out = out.view(out.shape[0], -1)
        code = self.Q(out)
        cat_code = code[:, :cat_dim]  # Discrete Code (Category)
        cont_code = code[:, cat_dim:] # Continuous Code
        return validity, cat_code, cont_code


IMG_SIZE = 64
batch_size = 64
n_classes =10
latent_dim = 52
cat_dim = n_classes
cont_dim = 2

os.makedirs("images/fixed/", exist_ok=True)
os.makedirs("images/varying_c1/", exist_ok=True)
os.makedirs("images/varying_c2/", exist_ok=True)
cuda = True if torch.cuda.is_available() else False

# Loss functions
adversarial_loss = torch.nn.BCELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()
    
# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

lr_G = 1e-3
lr_D = 2e-4

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=lr_G, betas=(0.5, 0.999)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

zero_noise = Variable(FloatTensor(np.zeros((n_classes ** 2, latent_dim))))
fixed_noise = Variable(FloatTensor(np.random.normal(0, 1, (n_classes ** 2, latent_dim))))
fixed_cat_code = to_categorical(
    np.array([num for _ in range(n_classes) for num in range(n_classes)]), num_classes=n_classes
)
zero_cont_code = Variable(FloatTensor(np.zeros((n_classes ** 2, cont_dim))))

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Fixed sample
    noise = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    fixed_sample = generator(fixed_noise, fixed_cat_code, zero_cont_code)
    save_image(fixed_sample.data, "images/fixed/%d.png" % batches_done, nrow=n_row, normalize=True)

    # Varied c1 and c2
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
    c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
    sample1 = generator(zero_noise, fixed_cat_code, c1)
    sample2 = generator(zero_noise, fixed_cat_code, c2)
    save_image(sample1.data, "images/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
    save_image(sample2.data, "images/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)

# Loss weights
lambda_cat = 1
lambda_con = 0.1

def prob_of_real(netD, images):
    with torch.no_grad():
        output, _, _ = discriminator(images.detach())
    return output.mean().item()

# Training
n_epochs = 80
sample_interval = 500

history = {"loss_G":[], "loss_D":[], "loss_info":[], 
           "D(x)_real_before":[], "D(x)_real_after":[],
           "D(x)_fake_before":[], "D(x)_fake_after":[]}

pbar = tqdm(total=len(dataloader)) 
for epoch in range(1, n_epochs+1):
    # Reset the pbar
    pbar.set_description("(%03d/%03d)"%(epoch, n_epochs))
    pbar.n = 0
    pbar.last_print_n = 0 
    pbar.refresh()
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Real image
        real_imgs = Variable(imgs.type(FloatTensor))
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        history["D(x)_real_before"] += [prob_of_real(discriminator, real_imgs)]
        
        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)
        
        # Sample Latent code
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        label_input = to_categorical(np.random.randint(0, n_classes, batch_size), n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, cont_dim))))
        # Fake image
        gen_imgs = generator(z, label_input, code_input)
        
        # Loss for fake images
        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        history['loss_D'] += [d_loss.item()]
        
        d_loss.backward()
        optimizer_D.step()
        
        history["D(x)_real_after"] += [prob_of_real(discriminator, real_imgs)]
        # -----------------
        #  Train Generator
        # -----------------
        history["D(x)_fake_before"] += [prob_of_real(discriminator, gen_imgs)]
        optimizer_G.zero_grad()

        # Generate a batch of images
        # gen_imgs = generator(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator
        fake_pred, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(fake_pred, valid)
        history['loss_G'] += [g_loss.item()]
        
        g_loss.backward()
        optimizer_G.step()
        
        with torch.no_grad():
            gen_imgs = generator(z, label_input, code_input)
        
        history["D(x)_fake_after"] += [prob_of_real(discriminator, gen_imgs)]
        
        # ------------------
        # Information Loss
        # ------------------
        optimizer_info.zero_grad()

        # Sample labels
        sampled_labels = np.random.randint(0, n_classes, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        label_input = to_categorical(sampled_labels, n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, cont_dim))))

        gen_imgs = generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_imgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            pred_code, code_input
        )
        history['loss_info'] += [info_loss.item()]
        
        info_loss.backward()
        optimizer_info.step()

        # --------------
        # Log Progress
        # --------------
        pbar.set_postfix({"D loss":d_loss.item(), "G loss":g_loss.item(), "Info Loss":info_loss.item()})
        pbar.update()
        batches_done = (epoch-1) * len(dataloader) + i
        if batches_done % sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)

# Save Weights
WEIGHT_DIR = 'weights'
os.makedirs(WEIGHT_DIR, exist_ok=True)
def save_model(net, name='model'):
    if '.pth' not in name:
        name += '.pth'
    torch.save(net, os.path.join(WEIGHT_DIR, name))

save_model(generator, "NetG")
save_model(discriminator, "NetD")

import matplotlib.pyplot as plt
def plot_history_loss(history, interval=100):
    N = len(history['loss_G'])
    xs = [i for i in range(1, N+1, interval)]
    loss_G = [np.mean(history['loss_G'][low:low+interval]) for low in range(0, N, interval)]
    loss_D = [np.mean(history['loss_D'][low:low+interval]) for low in range(0, N, interval)]
    loss_info = [np.mean(history['loss_info'][low:low+interval]) for low in range(0, N, interval)]
    plt.plot(xs, loss_G, label='Loss_G')
    plt.plot(xs, loss_D, label='Loss_D')
    plt.plot(xs, loss_info, label='Loss_Info')
    plt.legend(loc='best')
    plt.show()
    
def plot_history_prob(history, interval=100):
    N = len(history['loss_G'])
    xs = [i for i in range(1, N+1, interval)]
    
    prob_real_before = [np.mean(history['D(x)_real_before'][low:low+interval]) for low in range(0, N, interval)]
    prob_real_after  = [np.mean(history['D(x)_real_after'][low:low+interval]) for low in range(0, N, interval)]
    prob_fake_before = [np.mean(history['D(x)_fake_before'][low:low+interval]) for low in range(0, N, interval)]
    prob_fake_after  = [np.mean(history['D(x)_fake_after'][low:low+interval]) for low in range(0, N, interval)]
    
    plt.plot(xs, prob_real_before, label='D(x)_real_before')
    plt.plot(xs, prob_real_after, label='D(x)_real_after')
    plt.plot(xs, prob_fake_before, label='D(x)_fake_before')
    plt.plot(xs, prob_fake_after, label='D(x)_fake_after')
    plt.legend(loc='best')
    plt.show()

plot_history_loss(history)
plot_history_prob(history)