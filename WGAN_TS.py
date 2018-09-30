from __future__ import print_function
import argparse
import random
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
import os
import shutil
from glob import *

from cvxopt import matrix, spmatrix, sparse, solvers
import numpy as np
from copy import copy

import models.dcgan as dcgan
import models.mlp as mlp

####################################################################################
# WGAN-TS (A Two-Step Computation of the Exact GAN Wasserstein Distance, ICML 2018)
# Code is adopted from WGAN and modified by Huidong Liu (Hui-Dong Liu)
# Email: huidliu@cs.stonybrook.edu; h.d.liew@gmail.com
####################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist | cifar10 | lsun | imagenet | folder | lfw')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs for training')
parser.add_argument('--epoch', type=int, default=1, help='starting epoch (to continue training)')
parser.add_argument('--Giter', type=int, default=0, help='starting Generator iteration (to continue training)')
parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate for Critic, default=1e-4')
parser.add_argument('--lrG', type=float, default=1e-4, help='learning rate for Generator, default=1e-4')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--LAMBDA', type=float, default=10.0, help='lambda for optimal transport regularization')
parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--retrain', action='store_true', default=False, help='re-train or not')
parser.add_argument('--pin_mem', action='store_true', help='use pin memory or not')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--cuda_id', type=int, default=0, help='GPU id')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--Diters', type=int, default=1, help='number of D iters')
parser.add_argument('--ws', action='store_true', help='perform weight scaling or not')
parser.add_argument('--DOptIters', type=int, default=5, help='number of iters of regression of D, default=5')
parser.add_argument('--BN_G', action='store_true', help='use batchnorm for G')
parser.add_argument('--BN_D', action='store_true', help='use batchnorm for D')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--result_path', default=None, help='path to store samples and models')
parser.add_argument('--result_folder', default=None, help='folder to store samples and models')
parser.add_argument('--RMSprop', action='store_true', help='Whether to use RMSprop (default is Adam)')
parser.add_argument('--Adam', action='store_true', help='Whether to use Adam (default is Adam)')
args = parser.parse_args()
print(args)

ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
n_extra_layers = int(args.n_extra_layers)
batchSize = int(args.batchSize)
Diters = int(args.Diters)
DOptIters = int(args.DOptIters)
epochs = int(args.epochs)
epoch = int(args.epoch)
Giter = int(args.Giter)
LAMBDA = args.LAMBDA
cuda_id = int(args.cuda_id)
ws = args.ws

result_path = args.result_path
result_folder = args.result_folder

if result_folder is None:
    print("WARNING: No result folder provided. Results will be saved to temp_res")
    result_folder = 'temp_res'
if result_path is None:
    print("WARNING: No result path provided. Results will be saved to current directory")
    result_folder = '.'

result_folderPath = '{0}/{1}'.format(result_path, result_folder)
if not os.path.exists(result_folderPath):
    os.makedirs(result_folderPath)
else:
    if args.retrain:
        shutil.rmtree(result_folderPath)
        os.makedirs(result_folderPath)

img_path = '{0}/images'.format(result_folderPath)
model_path = '{0}/models'.format(result_folderPath)
if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

log_fileName = '{0}/{1}_log.txt'.format(result_folderPath, result_folder)
with open(log_fileName, 'a') as f:
    f.write('\n{}\n'.format(str(args)))

args.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda:{0}".format(cuda_id) if torch.cuda.is_available() else "cpu")

args.cuda = not args.no_cuda and torch.cuda.is_available()
if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if args.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(args.imageSize),
                                   transforms.CenterCrop(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc = 3
elif args.dataset == 'lsun':
    dataset = dset.LSUN(db_path=args.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(args.imageSize),
                            transforms.CenterCrop(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc = 3
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
    nc = 3
elif args.dataset == 'mnist':
    dataset = dset.MNIST(root=args.dataroot,
                         train=True, # download=True,
                         transform=transforms.Compose([
                            transforms.Scale(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ])
    )
    nc = 1

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers),
                                         drop_last=True, pin_memory=args.pin_mem)



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if args.BN_G:
    netG = dcgan.DCGAN_G(args.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
elif args.mlp_G:
    netG = mlp.MLP_G(args.imageSize, nz, nc, ngf, ngpu)
else:
    netG = dcgan.DCGAN_G_nobn(args.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)
if args.netG != '':  # load checkpoint if needed
    netG.load_state_dict(torch.load(args.netG))
print(netG)

if args.BN_D:
    netD = dcgan.DCGAN_D(args.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
elif args.mlp_D:
    netD = mlp.MLP_D(args.imageSize, nz, nc, ndf, ngpu)
else:
    netD = dcgan.DCGAN_D_nobn_bias(args.imageSize, nc, ndf, ngpu, n_extra_layers)

netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

data_dim = nc * args.imageSize * args.imageSize
real = torch.FloatTensor(batchSize, nc, args.imageSize, args.imageSize).to(device)
noise = torch.FloatTensor(batchSize, nz, 1, 1).to(device)
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1).to(device)
one = torch.FloatTensor([1])
mone = one * -1
ones = torch.ones(batchSize)
one, mone, ones = one.to(device), mone.to(device), ones.to(device)
netD = netD.to(device)
netG = netG.to(device)

criterion = torch.nn.MSELoss()


def set_optimizerD(Optmizer='Adam'):
    # setup optimizer
    if Optmizer == 'RMSprop':
        optimizerD = optim.RMSprop(netD.parameters(), lr = args.lrD)
    elif Optmizer == 'Adam':
        optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
    elif Optmizer == 'Adagrad':
        optimizerD = optim.Adagrad(netD.parameters(), lr=args.lrD)
    else:
        optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

    return optimizerD


def set_optimizerG(Optmizer='Adam'):
    # setup optimizer
    # setup optimizer
    if Optmizer == 'RMSprop':
        optimizerG = optim.RMSprop(netG.parameters(), lr=args.lrG)
    elif Optmizer == 'Adam':
        optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    elif Optmizer == 'Adagrad':
        optimizerG = optim.Adagrad(netG.parameters(), lr=args.lrG)
    else:
        optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))

    return optimizerG


def load_last_model(netD, netG, model_path):
    models = glob('{}/*.pth'.format(model_path))
    model_ids = [(int(f.split('_')[2]), int(f.split('_')[4]), f) for f in [p.split('/')[-1] for p in models]]
    if not model_ids:
        epoch = 1
        Giter = 1
        print('No netD or netG loaded!')
    else:
        epoch, Giter, _ = max(model_ids, key=lambda item: item[1])
        netD.load_state_dict(torch.load('{}/netD_epoch_{}_Giter_{}_.pth'.format(model_path, epoch, Giter)))
        print('netD_epoch_{}_Giter_{}_.pth loaded!'.format(epoch, Giter))
        netG.load_state_dict(torch.load('{}/netG_epoch_{}_Giter_{}_.pth'.format(model_path, epoch, Giter)))
        print('netG_epoch_{}_Giter_{}_.pth loaded!'.format(epoch, Giter))
        Giter += 1

    return epoch, Giter


###############################################################################
###################### Prepare linear programming solver ######################
solvers.options['show_progress'] = False
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

A = spmatrix(1.0, range(batchSize), [0]*batchSize, (batchSize,batchSize))
for i in range(1,batchSize):
    Ai = spmatrix(1.0, range(batchSize), [i]*batchSize, (batchSize,batchSize))
    A = sparse([A,Ai])

D = spmatrix(-1.0, range(batchSize), range(batchSize), (batchSize,batchSize))
DM = D
for i in range(1,batchSize):
    DM = sparse([DM, D])

A = sparse([[A],[DM]])

cr = matrix([-1.0/batchSize]*batchSize)
cf = matrix([1.0/batchSize]*batchSize)
c = matrix([cr,cf])

pStart = {}
pStart['x'] = matrix([matrix([1.0]*batchSize),matrix([-1.0]*batchSize)])
pStart['s'] = matrix([1.0]*(2*batchSize))
###############################################################################


def read_data(data_iter, batch_id):
    data = data_iter.next()
    batch_id += 1
    real_cpu, _ = data
    real_data = real_cpu.clone().to(device)
    real.resize_as_(real_data).copy_(real_data)
    noise.resize_(batchSize, nz, 1, 1).normal_(0, 1)
    with torch.no_grad():
        fake = netG(noise).detach()

    return real, fake, real_cpu, noise, batch_id


def computDif(real, fake, output_real, output_fake):                                                                                        
    num_r = real.size(0)
    num_f = fake.size(0)
    output_real_2D = output_real.unsqueeze(1).expand(num_r, num_f)
    output_fake_2D = output_fake.unsqueeze(0).expand(num_r, num_f)
    output_dif = output_real_2D - output_fake_2D

    return output_dif 


def comput_dist(real, fake):
    num_r = real.size(0)
    num_f = fake.size(0)
    real_flat = real.view(num_r, -1)
    fake_flat = fake.view(num_f, -1)

    real3D = real_flat.unsqueeze(1).expand(num_r, num_f, data_dim)
    fake3D = fake_flat.unsqueeze(0).expand(num_r, num_f, data_dim)
    # compute L1 distance
    dist = (torch.abs(real3D - fake3D)).sum(2).squeeze()

    return dist


def Wasserstein_LP(dist):
    b = matrix(dist.cpu().double().numpy().flatten())
    # sol = solvers.lp(c, A, b, primalstart=pStart, solver='glpk')
    sol = solvers.lp(c, A, b)
    offset = 0.5 * (sum(sol['x'])) / batchSize
    sol['x'] = sol['x'] - offset
    # pStart['x'] = sol['x']
    # pStart['s'] = sol['s']

    return sol


def approx_OT(sol):
    ###########################################################################
    ################ Compute the OT mapping for each fake data ################
    ResMat = np.array(sol['s']).reshape((batchSize,batchSize))
    mapping = torch.from_numpy(np.argmin(ResMat, axis=0)).long().to(device)

    return mapping
    ###########################################################################


###############################################################################
################## Optimal Transport Regularization ###########################
###############################################################################
## f(y) = inf { f(x) + c(x,y) }
## 0 \in grad_x { f(x) + c(x,y) }
## 0 \in grad_x f(x) + sign(x-y), since c(x,y) = ||x-y||_1
## regularize || grad_x f(x) - sign(y-x) ||^2
###############################################################################
def OT_regularization(netD, fake, RF_dif_sign):
    fake.requires_grad_()
    _, fake_output = netD(fake)
    fake_grad_output = torch.ones(fake_output.size()).to(device)

    gradients = autograd.grad(outputs=fake_output, inputs=fake,
                              grad_outputs=fake_grad_output,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    RegLoss = ((gradients - RF_dif_sign) ** 2).sum() / gradients.size(0)
    fake.requires_grad = False

    return RegLoss


def weight_scaling(dist, output_dif, n_layers):
    mask_gt0 = dist > 0
    dist_pos = dist.masked_select(mask_gt0)
    dif_pos = output_dif.masked_select(mask_gt0)
    max_scaling = torch.max(dif_pos / dist_pos)

    if max_scaling > 1:
        scaling_factor = 1 / pow(max_scaling, 1/n_layers)
        for p in netD.parameters():
            p.data = scaling_factor * p.data

    return max_scaling


def save_model(model_path, epoch, Giter):
    torch.save(netG.state_dict(), '{0}/netG_epoch_{1}_Giter_{2}_.pth'.format(model_path, epoch, Giter))
    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}_Giter_{2}_.pth'.format(model_path, epoch, Giter))


def save_images(img_path, real_cpu):
    real_cpu = real_cpu.mul(0.5).add(0.5)
    vutils.save_image(real_cpu, '{0}/real_samples.png'.format(img_path))
    with torch.no_grad():
        fake = netG(fixed_noise)
    fake.data = fake.cpu().data.mul(0.5).add(0.5)
    vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(img_path, Giter))

###############################################################################

n_layers = 8 + 2*n_extra_layers
max_scaling = -1
num_batches = len(dataloader)
data_iter = iter(dataloader)
batch_id = 0
Diters = int(args.Diters)
DOptIters = int(args.DOptIters)
optimizerD, optimizerG = set_optimizerD(), set_optimizerG()
epoch, Giter = load_last_model(netD, netG, model_path)
WD = torch.FloatTensor(1)

while epoch <= epochs:
    ###########################################################################
    #               (1) Update the Discriminator networks D
    ###########################################################################
    for p in netD.parameters(): # reset requires_grad
        p.requires_grad = True # they are set to False below in netG update

    netD.train()
    ###########################################################################
    #                    Deep Regression for discriminator
    ###########################################################################
    ##################### perform deep regression for D #######################
    j = 0
    while j < Diters:
        j += 1
        if batch_id >= num_batches:
            data_iter = iter(dataloader)
            batch_id = 0
            if epoch % 10 == 0:
                save_model(model_path, epoch, Giter)
            epoch += 1

        real, fake, real_cpu, noise, batch_id = read_data(data_iter, batch_id)

        dist = comput_dist(real, fake)
        sol = Wasserstein_LP(dist)
        if LAMBDA > 0:
            mapping = approx_OT(sol)
            real_ordered = real[mapping]  # match real and fake
            RF_dif = real_ordered - fake
            RF_dif_sign = torch.sign(RF_dif)

        # construct target
        target = torch.from_numpy(np.array(sol['x'])).float()
        target = target.squeeze().to(device)

        for k in range(DOptIters):
            netD.zero_grad()
            output_R_mean, output_real = netD(real)
            output_F_mean, output_fake = netD(fake)
            output_RF = torch.cat((output_real, output_fake), 0)
            L2LossD = criterion(output_RF, target)
            if LAMBDA > 0:
                RegLossD = OT_regularization(netD, fake, RF_dif_sign)
                TotalLoss = L2LossD + LAMBDA * RegLossD
            else:
                TotalLoss = L2LossD
            TotalLoss.backward()
            optimizerD.step()

        WD = output_R_mean - output_F_mean  # Wasserstein Distance

    if ws:
        output_dif = computDif(real, fake, output_real, output_fake)
        max_scaling = weight_scaling(dist, output_dif, n_layers)    
    #################### Discriminator Regression done ########################

    ###########################################################################
    #                   (2) Update the Generator network G
    ###########################################################################
    for p in netD.parameters():
        p.requires_grad = False # frozen D
        
    ###########################################################################
    ##                               Update G
    ###########################################################################
    netG.zero_grad()
    fake = netG(noise)
    output_F_mean_after, output_fake = netD(fake)
    output_F_mean_after.backward(mone)
    optimizerG.step()
    Giter += 1

    G_growth = output_F_mean_after - output_F_mean

    if Giter % 10 == 0:
        if LAMBDA > 0:
            log_str = '[{:d}/{:d}][{:d}] | WD {:.3f} | real_mean {:.3f} | fake_mean {:.3f} | G_growth {:.3f} | ' \
                      'L2LossD {:.3f} | RegLossD {:.3f} | TotalLoss {:.3f} | scaling {:.3f}'.format(
                epoch, epochs, Giter,
                WD.item(), output_R_mean.item(), output_F_mean.item(), G_growth.item(), L2LossD.item(),
                RegLossD.item(), TotalLoss.item(), max_scaling)
        else:
            log_str = '[{:d}/{:d}][{:d}] | WD {:.3f} | real_mean {:.3f} | fake_mean {:.3f} | G_growth {:.3f} | ' \
                      'L2LossD {:.3f} | scaling {:.3f}'.format(
                epoch, epochs, Giter,
                WD.item(), output_R_mean.item(), output_F_mean.item(), G_growth.item(), L2LossD.item(),
                max_scaling)
        print(log_str)
        with open(log_fileName, 'a') as f:
            f.write('{}\n'.format(log_str))

    if Giter % 500 == 0:
        save_images(img_path, real_cpu)

    if Giter % 10000 == 0:
        save_model(model_path, epoch, Giter)


