import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import numpy as np
import sys
import random

from nets.utils import ReplayBufferLabels, weights_init, set_requires_grad
from nets.discriminator import PatchMiniBatchNoiseLabel as Discriminator
from nets.unet_generator import UnetGenerator as Generator
from data.custom_dataset import CustomDataset

lrG = 0.0001
lrD = 0.0004
num_epochs = 40
batch_size = 80
ngpu = 4  
num_workers = ngpu*4
size = 256

writer = SummaryWriter("/edward-slow-vol/Sketch2Model/sketch2model/cgan")

datarealname = "/edward-slow-vol/Sketch2Model/Sketch2Model/data/combined_csv.csv"
simname = "/edward-slow-vol/Sketch2Model/Sketch2Model/data/overlap_sketch.csv"
savedir = "./"

dataset = CustomDataset(simname, datarealname)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                                         shuffle=True, num_workers=num_workers, drop_last=True)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
netG_B2A = Generator(use_dropout=True).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG_B2A = nn.DataParallel(netG_B2A, list(range(ngpu)))

netG_B2A.apply(weights_init)

netD_A = Discriminator(batch=int(batch_size/ngpu)).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD_A = nn.DataParallel(netD_A, list(range(ngpu)))

netD_A.apply(weights_init)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_classification = nn.CrossEntropyLoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(netG_B2A.parameters(), lr=lrG, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD_A.parameters(), lr=lrD, betas=(0.5, 0.999))


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(batch_size, 3, size, size)
input_B = Tensor(batch_size, 3, size, size)

input_A_labels = torch.Tensor(batch_size).type(torch.LongTensor).to(device)
input_B_labels = torch.Tensor(batch_size).type(torch.LongTensor).to(device)

# (30,30) for 256, (14,14) for 128, (6,6) for 64
out_size = 0
with torch.no_grad():
    a = Tensor(batch_size,3,256,256)
    out_size = netD_A(a)[0].shape[2]

target_real = torch.full((batch_size,3,out_size,out_size), real_label, dtype=torch.float, device=device)
target_fake = torch.full((batch_size,3,out_size,out_size), fake_label, dtype=torch.float, device=device)

fake_A_buffer = ReplayBufferLabels()
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader

    for i, data in enumerate(dataloader):

        i = i*batch_size
        simdata, sim_id, data, data_id, simname, realname = data

        real_A = Variable(input_A.copy_(data)) # image
        real_B = Variable(input_B.copy_(simdata)) # sketch

        data_id = Variable(input_A_labels.copy_(data_id)) # image_id
        sim_id = Variable(input_B_labels.copy_(sim_id)) # sketch_id

        fake_A = netG_B2A(real_B)  # G_B(B)

        ###### Generator #######
        set_requires_grad([netD_A], False)
        optimizer_G.zero_grad()

        pred_fake_A, _ = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real * 0.9)

        loss_G = loss_GAN_B2A
        loss_G.backward()

        optimizer_G.step()

        ####### Discriminator #######
        set_requires_grad([netD_A], True)
        optimizer_D.zero_grad()

        fake_A, fake_sim_id = fake_A_buffer.push_and_pop(fake_A.detach(), torch.reshape(sim_id.detach(),(batch_size,1)))
        pred_real, pred_real_A_labels = netD_A(real_A)
        noise = random.uniform(0.7, 1.2)
        loss_D_real = criterion_GAN(pred_real, target_real*noise)
        loss_classification_A_real = criterion_classification(pred_real_A_labels, data_id)

        pred_fake_A, pred_fake_A_labels = netD_A(fake_A)
        noise = random.uniform(0.0, 0.3)
        loss_D_fake = criterion_GAN(pred_fake_A, target_fake + noise)
        fake_sim_id = torch.squeeze(fake_sim_id, 1)
        loss_classification_A_fake = criterion_classification(pred_fake_A_labels, fake_sim_id)

        loss_D_A_real = loss_D_real + loss_classification_A_real
        loss_D_A_real.backward()
        loss_D_A_fake = loss_D_fake + loss_classification_A_fake
        loss_D_A_fake.backward()

        optimizer_D.step()

        step = epoch*(len(dataloader)-1)*batch_size + i
        if i % 1000 == 0:
            loss_dict = {
                'loss_G': loss_G.item(),
                'loss_G_GAN': (loss_GAN_B2A).item(),
                'loss_D': (loss_D_A_fake + loss_D_A_real).item()*0.5,
                'loss_D_classification': (loss_classification_A_fake + loss_classification_A_real).item(),
                'loss_D_GAN': ((loss_D_A_fake + loss_D_A_real) - (loss_classification_A_fake + loss_classification_A_real)).item()*0.5,

            }
            writer.add_scalars('losses', loss_dict, step)

        if i % 4000 == 0 or ((epoch == num_epochs-1) and (i == (len(dataloader)-1)*batch_size)):
            with torch.no_grad():
                fake_A = netG_B2A(real_B)

            img = (real_B[0]*0.5)+0.5
            writer.add_image('real_B '+ str(sim_id[0].item()), img, step)

            img = (fake_A[0]*0.5)+0.5
            writer.add_image('fake_A ' + str(sim_id[0].item()), img, step)

    filename = savedir + 'cycleGAN' + str(epoch) + '.pth'
    state = {'state_dict': netG_B2A.state_dict()}
    torch.save(state, filename)
    print('saved')
