import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import numpy as np
import sys
import random

from nets.utils import ReplayBuffer, weights_init, set_requires_grad
from nets.discriminator import PatchGAN as Discriminator
from nets.noise_generator import OrigGenerator as Generator
from data.single_dataset import CustomDataset

lrG = 0.0002
lrD = 0.0002
num_epochs = 2000
batch_size = 26
ngpu = 2  
num_workers = ngpu*2
size = 256
nz = 400

writer = SummaryWriter("/edward-slow-vol/Sketch2Model/sketch2model/noisy_labels")

datarealname = "/edward-slow-vol/Sketch2Model/Sketch2Model/data/airplane.csv"
savedir = "./airplane6/"

dataset = CustomDataset(datarealname)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                                         shuffle=True, num_workers=num_workers, drop_last=True)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
netG_B2A = Generator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG_B2A = nn.DataParallel(netG_B2A, list(range(ngpu)))

netG_B2A.apply(weights_init)

netD_A = Discriminator(batch=int(batch_size/ngpu)).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD_A = nn.DataParallel(netD_A, list(range(ngpu)))

netD_A.apply(weights_init)

# Lossess
criterion_GAN = torch.nn.MSELoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(netG_B2A.parameters(), lr=lrG, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD_A.parameters(), lr=lrD, betas=(0.5, 0.999))

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(batch_size, 3, size, size)

# (30,30) for 256, (14,14) for 128, (6,6) for 64
out_size = 0
with torch.no_grad():
    a = Tensor(batch_size,3,256,256)
    out_size = netD_A(a)[0].shape[2]

target_real = torch.full((batch_size,3,out_size,out_size), real_label, dtype=torch.float, device=device)
target_fake = torch.full((batch_size,3,out_size,out_size), fake_label, dtype=torch.float, device=device)

fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

fake_A_buffer = ReplayBuffer()

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader

    for i, data in enumerate(dataloader):

        noise = torch.randn(batch_size, nz, 1, 1, device=device)

        real_A = Variable(input_A.copy_(data)) # image

        fake_A = netG_B2A(noise)  # G_B(B)

        ###### Generator #######
        set_requires_grad([netD_A], False)
        optimizer_G.zero_grad()

        pred_fake_A = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real * 0.9)

        loss_G = loss_GAN_B2A
        loss_G.backward()

        optimizer_G.step()

        ####### Discriminator #######
        set_requires_grad([netD_A], True)
        optimizer_D.zero_grad()

        fake_A = fake_A_buffer.push_and_pop(fake_A.detach())
        pred_real = netD_A(real_A)
        noise = random.uniform(0.7, 1.2)
        loss_D_real = criterion_GAN(pred_real, target_real*noise)

        pred_fake_A = netD_A(fake_A)
        noise = random.uniform(0.0, 0.3)
        loss_D_fake = criterion_GAN(pred_fake_A, target_fake + noise)

        loss_D_A_real = loss_D_real
        loss_D_A_real.backward()
        loss_D_A_fake = loss_D_fake
        loss_D_A_fake.backward()

        optimizer_D.step()

        step = epoch*(len(dataloader)) + i
        if i % 200 == 0:
            loss_dict = {
                'loss_G': loss_G.item(),
                'loss_D': (loss_D_A_fake + loss_D_A_real).item()*0.5,
            }
            writer.add_scalars('losses', loss_dict, step)

        if i % 600 == 0 or (i == (len(dataloader)-1)):
            with torch.no_grad():
                fake_A = netG_B2A(fixed_noise)

            img = (fake_A[0]*0.5)+0.5
            writer.add_image('fake_A', img, step)
    if epoch> 1000 and epoch % 20 == 0:
        filename = savedir + 'gan' + str(epoch) + '.pth'
        state = {'state_dict': netG_B2A.state_dict()}
        torch.save(state, filename)
        print('saved' + str(epoch))
