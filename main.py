import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms as Transforms
from torch.utils.tensorboard import SummaryWriter
import itertools
from torch.autograd import Variable
from PIL import Image
import numpy as np
import sys
import random

from nets.utils import ReplayBuffer, weights_init
from nets.discriminator import PatchMiniBatch as Discriminator
from nets.generator import ResnetGenerator as Generator
from data.custom_dataset import CustomDataset

lrG = 0.0001
lrD = 0.0004
num_epochs = 40
batch_size = 16
ngpu = 4  
num_workers = ngpu*4
size = 256

writer = SummaryWriter("/edward-slow-vol/Sketch2Model/sketch2model")

datarealname = "/edward-slow-vol/Sketch2Model/Sketch2Model/data/overlap_photo.csv"
simname = "/edward-slow-vol/Sketch2Model/Sketch2Model/data/overlap_sketch.csv"
savedir = "./"

dataset = CustomDataset(simname, datarealname)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                                         shuffle=True, num_workers=num_workers, drop_last=True)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG_A2B = nn.DataParallel(netG_A2B, list(range(ngpu)))
    netG_B2A = nn.DataParallel(netG_B2A, list(range(ngpu)))

netG_A2B.apply(weights_init)
netG_B2A.apply(weights_init)

netD_A = Discriminator(batch=int(batch_size/ngpu)).to(device)
netD_B = Discriminator(batch=int(batch_size/ngpu)).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD_A = nn.DataParallel(netD_A, list(range(ngpu)))
    netD_B = nn.DataParallel(netD_B, list(range(ngpu)))

netD_A.apply(weights_init)
netD_B.apply(weights_init)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lrG, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lrD, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lrD, betas=(0.5, 0.999))


# Establish convention for real and fake labels during training
real_label = 0.9
fake_label = 0.

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(batch_size, 3, size, size)
input_B = Tensor(batch_size, 3, size, size)

# (30,30) for 256, (14,14) for 128, (6,6) for 64
out_size = 0
with torch.no_grad():
    a = Tensor(batch_size,3,256,256)
    out_size = netD_A(a).shape[2]

target_real = torch.full((batch_size,3,out_size,out_size), real_label, dtype=torch.float, device=device)
target_fake = torch.full((batch_size,3,out_size,out_size), fake_label, dtype=torch.float, device=device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader

    for i, data in enumerate(dataloader):

        i = i*batch_size
        simdata, simname, data, realname = data

        b_size,channels,h,w = data.shape

        real_A = Variable(input_A.copy_(data)) # image
        real_B = Variable(input_B.copy_(simdata)) # sketch

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0

        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A.detach())
        pred_fake = netD_A(fake_A)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B.detach())
        pred_fake = netD_B(fake_B)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        if i % 1000 == 0:
            loss_dict = {
                'loss_G': loss_G.item(),
                'loss_G_identity': (loss_identity_A + loss_identity_B).item(),
                'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A).item(),
                'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB).item(),
                'loss_D': (loss_D_A + loss_D_B).item()

            }
            writer.add_scalars('losses', loss_dict, i)

        if i % 4000 == 0 or ((epoch == num_epochs-1) and (i == (len(dataloader)-1)*batch_size)):
            with torch.no_grad():
                fake_A = netG_B2A(real_B).detach().cpu().numpy()
                fake_B = netG_A2B(real_A).detach().cpu().numpy()

            img = (real_A[0]*0.5)+0.5
            writer.add_image('real_A', img, i)

            img = (real_B[0]*0.5)+0.5
            writer.add_image('real_B', img, i)

            img = (fake_A[0]*0.5)+0.5
            writer.add_image('fake_A', img, i)

            img = (fake_B[0]*0.5)+0.5
            writer.add_image('fake_B', img, i)
            writer.flush()

    filename = savedir + 'cycleGAN' + str(epoch) + '.pth'
    state = {'state_dict': netG_B2A.state_dict()}
    torch.save(state, filename)
    print('saved')