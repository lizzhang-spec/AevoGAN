import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
import numpy as np

from models import Generator
from torchvision.datasets import ImageFolder

import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', type=bool, default=True , help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

device=torch.device('cuda')


first_conv_out=64
mask_chns=[]
mask_chns.append(first_conv_out) #1st conv
mask_chns.append(first_conv_out*2) #2nd conv
mask_chns.append(first_conv_out*4) #3rd conv 1~9 res_block
mask_chns.append(first_conv_out*2) #1st trans_conv
mask_chns.append(first_conv_out) #2nd trans_conv
bit_len=0
for mask_chn in mask_chns:
    bit_len+= mask_chn

model_A2B = Generator(opt.input_nc, opt.output_nc)
model_A2B.load_state_dict(torch.load('./log/output/A2B_200.pth'))

model_B2A = Generator(opt.input_nc, opt.output_nc)
model_B2A.load_state_dict(torch.load('./log/output/B2A_200.pth'))



if opt.cuda:
    model_A2B.to(device)
    model_B2A.to(device)
               

model_A2B.eval()
model_B2A.eval()

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)

transforms_ = transforms.Compose([ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])


dataloader = DataLoader(ImageFolder('./horse2zebra/test', transform=transforms_), batch_size=opt.batchSize, shuffle=False)

log_dir='./log/GA/horse2zebra/'

# Create output dirs if they don't exist
if not os.path.exists(log_dir+'A'):
    os.makedirs(log_dir+'A')
if not os.path.exists(log_dir+'B'):
    os.makedirs(log_dir+'B')

for i, (batch,label) in enumerate(dataloader):
    
    label_npindexes=np.argwhere(np.array(label)==0)
    if len(label_npindexes)==0:
        continue
    batch_A=batch[label_npindexes.reshape([-1])]
        
    real_A = batch_A.to(device)

    label_npindexes=np.argwhere(np.array(label)==1)
    if len(label_npindexes)==0:
        continue
    batch_B=batch[label_npindexes.reshape([-1])]

    real_B = batch_B.to(device)


    fake_A = 0.5*(model_B2A(real_B).data + 1.0)
    fake_B = 0.5*(model_A2B(real_A).data + 1.0)


    for number in range(fake_B.size(0)):
        save_image(fake_B[number:number+1], log_dir+'B/%04d-%04d.png' % (i+1,number))
    
    for number in range(fake_A.size(0)):
        save_image(fake_A[number:number+1], log_dir+'A/%04d-%04d.png' % (i+1,number))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

 
sys.stdout.write('\n')
