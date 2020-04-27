import torch
import numpy as np
import torchvision
import math
import matplotlib.pyplot as plt
import scipy
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch import optim
from IPython.display import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def svhn_sampler(root, train_batch_size, test_batch_size, valid_split=0):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform = transforms.Compose((
        transforms.ToTensor(),
        normalize))
    train = datasets.SVHN(root, split='train', download=True, transform=transform)
    valid = datasets.SVHN(root, split='train', download=True, transform=transform)
    test = datasets.SVHN(root, split='test', download=True, transform=transform)

    idxes = np.arange(len(train))
    split = int(valid_split * len(idxes))
    train_sampler = SubsetRandomSampler(idxes[split:])
    valid_sampler = SubsetRandomSampler(idxes[:split])
    train_loader = torch.utils.data.DataLoader(train, batch_size=train_batch_size,
                                               sampler=train_sampler, num_workers=4, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=train_batch_size,
                                               sampler=valid_sampler, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test, batch_size=test_batch_size, num_workers=4)
    return train_loader, valid_loader, test_loader,


class Critic(nn.Module):
    def __init__(self, h_dim=64, distance='em', lp_coeff=10):
        super(Critic, self).__init__()

        x = [nn.Conv2d(3, h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim, 2*h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(2*h_dim, 4*h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(4*h_dim, 1, 4, 1, 0)]

        self.x = nn.Sequential(*x)
        self.distance = distance
        self.lp_coeff = lp_coeff

    def forward(self, x):
        return self.x(x).squeeze()

    def get_distance(self, samples_0, samples_1):
        if self.distance =='sh':
            d = -1*self.vf_squared_hellinger(samples_0, samples_1) #definitions of x/y p/q got reversed
        else: #self.distance =='em': :
            d = self.vf_wasserstein_distance(samples_0, samples_1)
        return d

    def get_loss(self, samples_0, samples_1):
        loss = -self.get_distance(samples_0, samples_1) + self.lp_coeff*self.lp_reg(samples_0, samples_1)
        return loss

    def lp_reg(self, x, y):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.empty_like(x).uniform_(0, 1)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * x + ((1 - alpha) * y)).requires_grad_(True)
        interpolates_forward = self.forward(interpolates)
        # need a fake grad output
        fake = torch.ones_like(interpolates_forward)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=interpolates_forward,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            only_inputs=True)[0]
        return ((torch.norm(gradients, p=2, dim=1) - 1)**2).mean()

    def vf_wasserstein_distance(self, p, q):
        V_p = self(p).to(device)
        V_q = self(q).to(device)
        return torch.mean(V_p) - torch.mean(V_q)

    def vf_squared_hellinger(self, x, y):
        V_x = self(x).to(device)
        V_y = self(y).to(device)
        
        t_x = 1 - torch.exp(-V_x)
        t_y = 1 - torch.exp(-V_y)
        return torch.mean(t_x) - torch.mean(t_y/(torch.tensor([1.]).to(device)-t_y)) 


class Generator(nn.Module):
    def __init__(self, z_dim=100, h_dim=64):
        super(Generator, self).__init__()

        decoder = [nn.ConvTranspose2d(z_dim, 4*h_dim, 4, 1, 0),
                   nn.BatchNorm2d(4*h_dim),
                   nn.ReLU(True),
                   nn.ConvTranspose2d(4*h_dim, 2*h_dim, 4, 2, 1),
                   nn.BatchNorm2d(2*h_dim),
                   nn.ReLU(True),
                   nn.ConvTranspose2d(2*h_dim, h_dim, 4, 2, 1),
                   nn.BatchNorm2d(h_dim),
                   nn.ReLU(True),
                   nn.ConvTranspose2d(h_dim, 3, 4, 2, 1),
                   nn.Tanh()
                   ]
        self.decoder = nn.Sequential(*decoder)
        self.z_dim = z_dim

    def forward(self, z):
        return self.decoder(z.view(z.shape[0], z.shape[1], 1, 1))
    
    def generate(self, z):
        with torch.no_grad():
            return self.decoder(z.view(z.shape[0], z.shape[1], 1, 1).to(device))


def generate_samples(model, number_of_images_to_create=9, columns=8, plot=True, path=None):
    with torch.no_grad():
      image_name = 'test.png'
      if path:
        image_name = path+image_name
      image_tensors = model.generate(torch.randn(number_of_images_to_create, 100).to(device))
      torchvision.utils.save_image(image_tensors, image_name, columns, normalize=True)
      if plot:
        image_grid_of_images = Image(image_name)
        display(image_grid_of_images)


def disentangle(model, dimension=[0, 1, 2], epsilon=0.02, size=10):
  with torch.no_grad():
    z = torch.randn(1, 100).to(device)
    for i, d in enumerate(dimension):
      
      interpolation = z.repeat(size, 1)
      for j, s in enumerate(interpolation):
        interpolation[j][d] += (j - size / 2) * epsilon
      image_name = 'temp_disentangled.png'
      image_tensors = model.generate(interpolation.to(device))
      torchvision.utils.save_image(image_tensors, image_name, size, normalize=True)      
      image_grid_of_images = Image(image_name)
      display(image_grid_of_images)


def latent_interpolation(model):
  with torch.no_grad():
    z0 = torch.randn(1, 100)
    z1 = torch.randn(1, 100)
    alphas = np.arange(0, 1.0, 0.1)
    size = len(alphas)

    z = z1.repeat(size, 1)
    for i, alpha in enumerate(alphas):
        z[i] = alpha*z0 + (1-alpha)*z1
        
    image_name = 'temp_latent_interpolation.png'
    image_tensors = model.generate(z.to(device))
    torchvision.utils.save_image(image_tensors, image_name, size, normalize=True)      
    image_grid_of_images = Image(image_name)
    display(image_grid_of_images)


def output_interpolation(model):
    with torch.no_grad():
      z0 = torch.randn(1, 100)
      x0 = model.generate(z0.to(device))

      z1 = torch.randn(1, 100)
      x1 = model.generate(z1.to(device))

      alphas = np.arange(0, 1.0, 0.1)
      size = len(alphas)

      x = torch.Tensor(size, x0.shape[1], x0.shape[2], x0.shape[3]) #x1.repeat(size, 0)
      for i, alpha in enumerate(alphas):
          x[i] = alpha*x0[0] + (1-alpha)*x1[0]

      image_name = 'temp_output_interpolation.png'
      torchvision.utils.save_image(x, image_name, size, normalize=True)      
      image_grid_of_images = Image(image_name)
      display(image_grid_of_images)


def main():
    data_root = './'
    train_batch_size = 64
    test_batch_size = 64
    train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100
    lp_coeff = 10 # Lipschitz penalty coefficient

    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic(lp_coeff=lp_coeff).to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    n_epochs = 50 # 
    n_critic_updates = 5 # N critic updates per generator update

    # COMPLETE TRAINING PROCEDURE
    # put these into training mode
    generator.train()
    critic.train()
    iterations = 0
    for epoch in range(1, n_epochs):
        running_loss_generator = 0.0
        running_loss_critic = 0.0
        for i, data in enumerate(train_loader, 0):
            iterations += 1
            ### Critic training :
            optim_generator.zero_grad()
            optim_critic.zero_grad()
            
            x_real = data[0].to(device)
            z = torch.randn(train_batch_size, generator.z_dim).to(device)
            x_fake = generator.forward(z)

            loss_critic = critic.get_loss(x_real, x_fake)
            loss_critic.backward()
            optim_critic.step()
            running_loss_critic -= loss_critic.item()
            
            ### Generator training :
            if i % n_critic_updates == 0:
                optim_generator.zero_grad()
                optim_critic.zero_grad()

                z = torch.randn(train_batch_size, generator.z_dim).to(device)
                x_fake = generator.forward(z)
                x_fake_critic = critic.forward(x_fake)
                loss_generator = -torch.mean(x_fake_critic)
                loss_generator.backward()

                optim_generator.step()
                running_loss_generator -= loss_generator.item()

        print('epoch: {:2d} loss_critic: {:6.2f} loss_generator: {:6.2f} iterations: {:10d}'.format(
            epoch, (running_loss_critic / i), (running_loss_generator / (i // n_critic_updates)), iterations))


    # Put these back into evaluation mode
    generator.eval()
    critic.eval()
    generate_samples(generator, number_of_images_to_create=64, columns=8, plot=True, path='/content/')
    disentangle(generator, dimension=range(z_dim), epsilon=1, size=10)
    latent_interpolation(generator)
    output_interpolation(generator)

if __name__ == '__main__':
    main()