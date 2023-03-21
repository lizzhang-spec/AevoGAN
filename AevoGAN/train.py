import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn


from models import Generator
from models import Discriminator
from utils import compute_layer_mask
from utils import ReplayBuffer
from utils import LambdaLR
from torchvision.datasets import ImageFolder

import numpy as np

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=201, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=6, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
opt = parser.parse_args()

device = torch.device('cuda')

first_conv_out = 64
mask_chns = []
mask_chns.append(first_conv_out)  # 1st conv
mask_chns.append(first_conv_out * 2)  # 2nd conv
mask_chns.append(first_conv_out * 4)  # 3rd conv 1~9 res_block
mask_chns.append(first_conv_out * 2)  # 1st trans_conv
mask_chns.append(first_conv_out)  # 2nd trans_conv
bit_len = 0
for mask_chn in mask_chns:
    bit_len += mask_chn


def train_from_mask():

    mask_input_A2B = np.loadtxt("./best_fitness_A2B.txt")
    mask_input_B2A = np.loadtxt("./best_fitness_B2A.txt")
    cfg_mask_A2B = compute_layer_mask(mask_input_A2B, mask_chns)
    cfg_mask_B2A = compute_layer_mask(mask_input_B2A, mask_chns)

    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netG_B2A.to(device)
    netG_A2B = Generator(opt.output_nc, opt.input_nc)
    netG_A2B.to(device)

    netD_A = Discriminator(opt.input_nc)
    netD_A.to(device)
    netD_B = Discriminator(opt.output_nc)
    netD_B.to(device)

    model_A2B = Generator(opt.input_nc, opt.output_nc)
    model_A2B.to(device)
    model_B2A = Generator(opt.input_nc, opt.output_nc)
    model_B2A.to(device)


    netD_A.load_state_dict(torch.load('./netD_A.pth'))
    netD_B.load_state_dict(torch.load('./netD_B.pth'))

    cfg_id = 0
    start_mask_A2B = np.ones(3)
    start_mask_B2A = np.ones(3)
    end_mask_A2B = cfg_mask_A2B[cfg_id]
    end_mask_B2A = cfg_mask_B2A[cfg_id]


    for [m0, m1] in zip(model_A2B.modules(), model_B2A.modules()):

        if isinstance(m0, nn.Conv2d) and isinstance(m1, nn.Conv2d):

            if hasattr(m0,'flag') and hasattr(m1,'flag'): continue

            mask_A2B=np.ones(m0.weight.data.shape)
            mask_bias_A2B = np.ones(m0.bias.data.shape)
            mask_B2A = np.ones(m0.weight.data.shape)
            mask_bias_B2A = np.ones(m0.bias.data.shape)

            cfg_mask_start_A2B = np.ones(start_mask_A2B.shape) - start_mask_A2B
            cfg_mask_end_A2B = np.ones(end_mask_A2B.shape) - end_mask_A2B
            cfg_mask_start_B2A = np.ones(start_mask_B2A.shape) - start_mask_B2A
            cfg_mask_end_B2A = np.ones(end_mask_B2A.shape) - end_mask_B2A

            idx0_A2B = np.squeeze(np.argwhere(np.asarray(cfg_mask_start_A2B)))
            idx1_A2B = np.squeeze(np.argwhere(np.asarray(cfg_mask_end_A2B)))
            idx0_B2A = np.squeeze(np.argwhere(np.asarray(cfg_mask_start_B2A)))
            idx1_B2A = np.squeeze(np.argwhere(np.asarray(cfg_mask_end_B2A)))

            temp_shape_A2B = mask_A2B.shape
            mask_A2B = mask_A2B.reshape([-1])

            mask_A2B[idx0_A2B.tolist()] = 0
            mask_A2B[idx1_A2B.tolist()] = 0
            mask_A2B = mask_A2B.reshape(temp_shape_A2B)

            mask_bias_A2B[idx1_A2B.tolist()] = 0

            temp_shape_B2A = mask_B2A.shape
            mask_B2A = mask_B2A.reshape([-1])

            mask_B2A[idx0_B2A.tolist()] = 0
            mask_B2A[idx1_B2A.tolist()] = 0
            mask_B2A = mask_B2A.reshape(temp_shape_B2A)

            mask_bias_B2A[idx1_B2A.tolist()] = 0

            m0.weight.data = m0.weight.data * torch.FloatTensor(mask_A2B).to(device)
            m0.bias.data = m0.bias.data * torch.FloatTensor(mask_bias_A2B).to(device)
            m1.weight.data = m1.weight.data * torch.FloatTensor(mask_B2A).to(device)
            m1.bias.data = m1.bias.data * torch.FloatTensor(mask_bias_B2A).to(device)

            m0.weight.data[:, idx0_A2B.tolist()].requires_grad = False
            m0.weight.data[idx1_A2B.tolist()].requires_grad = False
            m0.bias.data[idx1_A2B.tolist()].requires_grad = False

            m1.weight.data[:, idx0_B2A.tolist()].requires_grad = False
            m1.weight.data[idx1_B2A.tolist()].requires_grad = False
            m1.bias.data[idx1_B2A.tolist()].requires_grad = False

            cfg_id += 1
            start_mask_A2B = end_mask_A2B
            start_mask_B2A= end_mask_A2B
            if cfg_id < len(cfg_mask_A2B):
                end_mask_A2B = cfg_mask_A2B[cfg_id]
                end_mask_B2A = cfg_mask_B2A[cfg_id]
            continue

        elif isinstance(m0, nn.ConvTranspose2d) and isinstance(m1, nn.ConvTranspose2d):

            mask_A2B = np.ones(m0.weight.data.shape)
            mask_bias_A2B = np.ones(m0.bias.data.shape)
            mask_B2A = np.ones(m0.weight.data.shape)
            mask_bias_B2A = np.ones(m0.bias.data.shape)

            cfg_mask_start_A2B = np.ones(start_mask_A2B.shape) - start_mask_A2B
            cfg_mask_end_A2B = np.ones(end_mask_A2B.shape) - end_mask_A2B
            cfg_mask_start_B2A = np.ones(start_mask_B2A.shape) - start_mask_B2A
            cfg_mask_end_B2A = np.ones(end_mask_B2A.shape) - end_mask_B2A

            idx0_A2B = np.squeeze(np.argwhere(np.asarray(cfg_mask_start_A2B)))
            idx1_A2B = np.squeeze(np.argwhere(np.asarray(cfg_mask_end_A2B)))
            idx0_B2A = np.squeeze(np.argwhere(np.asarray(cfg_mask_start_B2A)))
            idx1_B2A = np.squeeze(np.argwhere(np.asarray(cfg_mask_end_B2A)))

            temp_shape_A2B = mask_A2B.shape
            mask_A2B = mask_A2B.reshape([-1])

            mask_A2B[idx0_A2B.tolist()] = 0
            mask_A2B[idx1_A2B.tolist()] = 0
            mask_A2B = mask_A2B.reshape(temp_shape_A2B)

            mask_bias_A2B[idx1_A2B.tolist()] = 0

            temp_shape_B2A = mask_B2A.shape
            mask_B2A = mask_B2A.reshape([-1])

            mask_B2A[idx0_B2A.tolist()] = 0
            mask_B2A[idx1_B2A.tolist()] = 0
            mask_B2A = mask_B2A.reshape(temp_shape_B2A)

            mask_bias_B2A[idx1_B2A.tolist()] = 0

            m0.weight.data = m0.weight.data * torch.FloatTensor(mask_A2B).to(device)
            m0.bias.data = m0.bias.data * torch.FloatTensor(mask_bias_A2B).to(device)
            m1.weight.data = m1.weight.data * torch.FloatTensor(mask_B2A).to(device)
            m1.bias.data = m1.bias.data * torch.FloatTensor(mask_bias_B2A).to(device)

            m0.weight.data[idx0_A2B.tolist(),:].requires_grad = False
            m0.weight.data[idx1_A2B.tolist()].requires_grad = False
            m0.bias.data[idx1_A2B.tolist()].requires_grad = False

            m1.weight.data[idx0_B2A.tolist(),:].requires_grad = False
            m1.weight.data[idx1_B2A.tolist()].requires_grad = False
            m1.bias.data[idx1_B2A.tolist()].requires_grad = False

            cfg_id += 1
            start_mask_A2B = end_mask_A2B
            start_mask_B2A = end_mask_A2B
            end_mask_A2B = cfg_mask_A2B[cfg_id]
            end_mask_B2A = cfg_mask_B2A[cfg_id]
            continue


    model_A2B.load_state_dict(torch.load('./model*_A2B.pth'))
    model_B2A.load_state_dict(torch.load('./model*_A2B.pth'))

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    lamda_loss_ID = 5.0
    lamda_loss_G = 1.0
    lamda_loss_cycle = 10.0
    optimizer_G = torch.optim.Adam(itertools.chain(model_A2B.parameters(), model_B2A.parameters()), lr=opt.lr,
                                   betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                       opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)

    transforms_ = transforms.Compose([
        transforms.Scale(int(opt.size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    dataloader = DataLoader(ImageFolder('./horse2zebra/train', transform=transforms_),
                            batch_size=opt.batchSize, shuffle=True, drop_last=True)

    loss_G = 0
    loss_identity_A = 0
    loss_identity_B = 0
    loss_GAN_A2B = 0
    loss_GAN_B2A = 0
    loss_cycle_ABA = 0
    loss_cycle_BAB = 0

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, (batch, label) in enumerate(dataloader):


            label_npindexes_A = np.argwhere(np.array(label) == 0)
            if len(label_npindexes_A) == 0:
                continue
            input_A = batch[label_npindexes_A.reshape([-1])]

            label_npindexes_B = np.argwhere(np.array(label) == 1)
            if len(label_npindexes_B) == 0:
                continue
            input_B = batch[label_npindexes_B.reshape([-1])]

            real_A = input_A.to(device)
            real_B = input_B.to(device)

            optimizer_G.zero_grad()

            same_B = model_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * lamda_loss_ID  # initial 5.0

            same_A = model_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * lamda_loss_ID  # initial 5.0


            # GAN loss
            fake_B = model_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real[label_npindexes_A.reshape([-1])]) * lamda_loss_G  # initial 1.0


            fake_A = model_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real[label_npindexes_B.reshape([-1])]) * lamda_loss_G  # initial 1.0



            recovered_A = model_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * lamda_loss_cycle  # initial 10.0

            recovered_B = model_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * lamda_loss_cycle  # initial 10.0


            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()


            optimizer_D_A.zero_grad()


            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real[label_npindexes_A.reshape([-1])])


            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake[label_npindexes_B.reshape([-1])])

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real[label_npindexes_B.reshape([-1])])

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake[label_npindexes_A.reshape([-1])])

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()

        print("epoch:%d  Loss G:%4f  LossID_A:%4f LossID_B:%4f  Loss_G_A2B:%4f  Loss_G_B2A:%4f  Loss_Cycle_ABA:%4f  Loss_Cycle_BAB:%4f " % (
            epoch, loss_G, loss_identity_A, loss_identity_B, loss_GAN_A2B, loss_GAN_B2A, loss_cycle_ABA,loss_cycle_BAB))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if epoch % 20 == 0:
            # Save models checkpoints
            torch.save(model_A2B.state_dict(), './log/output/A2B_%d.pth' % (epoch))
            torch.save(model_B2A.state_dict(), './log/output/B2A_%d.pth' % (epoch))




if __name__ == "__main__":
    train_from_mask()

