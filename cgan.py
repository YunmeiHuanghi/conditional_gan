import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.datasets.mnist import load_data
import torchvision
from torch.utils.tensorboard import SummaryWriter
import argparse

# load (and normalize) mnist dataset
# set up the data  


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN on MNIST dataset')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200_000, help='number of epochs to train for')
    parser.add_argument('--k', type=int, default=1, help='number of times to update discriminator per generator update')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate for optimizer')
    parser.add_argument('--leaky_relu', help='use leaky relu in Generator() ',action="store_true")
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of latent space')
    parser.add_argument('--context_dim', type=int, default=10, help='dimensionality of conditional information')
    parser.add_argument('--output_dim', type=int, default=28*28, help='dimensionality of output image')
    parser.add_argument('--input_dim', type=int, default=28*28, help='dimensionality of input image')
    parser.add_argument('--device', type=str, default='cuda', help='device to train on (cuda or cpu)')
    parser.add_argument('--logdir', type=str, default='run_results', help='directory for storing tensorboard logs')

    args = parser.parse_args()
    return args

 


(real_image_train, train_label), (real_img_test, test_label) = load_data()
real_image_train = np.float32(real_image_train) / 255.

 
#use one-hot encoding to get indices for different labels
def get_batch(batch_size, device):
    indices = torch.randperm(real_image_train.shape[0])[:batch_size]

    return torch.tensor(real_image_train[indices], dtype=torch.float).reshape(batch_size, -1).to(
        device), torch.nn.functional.one_hot(torch.tensor(train_label[indices], dtype=torch.long), num_classes=10).to(
        device).type(torch.float)

def generate_sample_noise(size, device, dim=100):
    return torch.rand((size, dim), device=device)

# define the generator network

class Generator(nn.Module):

    def __init__(self, latent_dim=100, context_dim=10, output_dim=28 * 28,leaky_relu=False):
        super(Generator, self).__init__()
        
        #self.hidden1_z = nn.Sequential(nn.Linear(latent_dim, 200), nn.Dropout(p=0.5),  nn.ReLU(), )
        self.linear=nn.Linear(latent_dim, 200)
        self.dropout =nn.Dropout(p=0.5)
        self.relu=nn.ReLU()
         # one layer for conditional information
        
        self.linear_c=nn.Linear(context_dim, 1000)
        self.dropout_c =nn.Dropout(p=0.5)
        self.relu_c=nn.ReLU()
        self.linear_h = nn.Linear(1200,1200)
        self.dropout_h=nn.Dropout(p=0.5)
        if leaky_relu:
            self.relu_h=nn.LeakyReLU()
        else:
            self.relu_h=nn.ReLU()

   
        self.out_l = nn.Sequential(nn.Linear(1200, output_dim), nn.Sigmoid(), )
       
        
    def forward(self, noise, context):
        z = self.linear(noise)
        z = self.dropout(z)
        z = self.relu(z)
        
     

        context = self.linear_c(context)
        context = self.dropout_c(context)
        context=self.relu_c(context)
        h = torch.cat(( z , context), dim=1)
        h = self.linear_h(h)
        h=self.dropout_h(h)
        h=self.relu_h(h)
        return self.out_l(h)
    

# define the discriminator network

class Discriminator(nn.Module):
   
    def __init__(self, input_dim=28 * 28, context_dim=10):
        super(Discriminator, self).__init__()
         #one layer for input data
        self.hidden1_x = nn.Sequential(nn.Linear(input_dim, 240), nn.Dropout(p=0.5), nn.LeakyReLU(), )
        # one layer for conditional information
        self.hidden1_context = nn.Sequential(nn.Linear(context_dim, 50), nn.Dropout(p=0.5), nn.LeakyReLU(), )
        self.hidden2 = nn.Sequential(nn.Linear(290, 240), nn.Dropout(p=0.5), nn.LeakyReLU(), )
        self.out_layer = nn.Sequential(nn.Linear(240, 1), nn.Sigmoid(), )
    def forward(self, x, context):
        h = torch.cat((self.hidden1_x(x), self.hidden1_context(context)), dim=1)
        h = self.hidden2(h)
        return self.out_layer(h)

 

def train(generator, discriminator, generator_optimizer, discriminator_optimizer, schedulers, num_epochs, k=1,
          batch_size=100,loss_func=torch.nn.BCELoss()):
    training_loss_gen  =  []
    training_loss_disc =  []

   # {'generative': [], 'discriminator': []}
    for epoch in tqdm(range(num_epochs)):

        ### Train the disciminator
        for _ in range(k):
            # Sample a minibatch of m noise samples
            z = generate_sample_noise(batch_size, device)
            # Sample a minibatch of m examples from the data generating distribution
            x, label = get_batch(batch_size, device)
           

            # Update the discriminator by ascending its stochastic gradient
            #add condition to generator
            generator_output = generator(z,label)

   
            f_loss = loss_func(discriminator(generator_output, label).reshape(batch_size),
                                        torch.zeros(batch_size, device=device))
            

            r_loss = loss_func(discriminator(x, label).reshape(batch_size),
                                        torch.ones(batch_size, device=device))
            loss = (r_loss + f_loss) / 2
            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()
            
            training_loss_disc.append(loss.item())
            writer.add_scalar('Loss/Discrimintor',  loss.item(), len( training_loss_disc)-1 )
       # save data to tensorboard
        if epoch%50000 ==0:
            g_output_images =generator_output.detach().cpu().reshape((-1,1,28,28))
            g_grid_img= torchvision.utils.make_grid(g_output_images,nomoralize=True)
            writer.add_image("generator image",g_grid_img,global_step =epoch )


        ### Train the generator
        # Sample a minibatch of m noise samples
        z = generate_sample_noise(batch_size, device)
        _, label = get_batch(batch_size, device)
        # Update the generator by descending its stochastic gradient
        loss = loss_func(discriminator(generator(z, label), label).reshape(batch_size),
                                  torch.ones(batch_size, device=device))
        generator_optimizer.zero_grad()
        loss.backward()
        generator_optimizer.step()
        training_loss_gen.append(loss.item())
        writer.add_scalar('Loss/generative',  loss.item(), len(training_loss_gen)-1 )

    
 
        for scheduler in schedulers:
            scheduler.step()
    #save loss
    #for j,Dics_loss in  enumerate(training_loss['discriminator']):
     #   writer.add_scalar('Loss/discriminator', Dics_loss,j)
      #  print("Dics_loss", i ,"=" ,Dics_loss)
    return training_loss_disc,training_loss_gen



if __name__ == "__main__":
    args=parse_args()
    device = args.device  
    writer =SummaryWriter(f"{args.logdir}")
    discriminator = Discriminator(args.input_dim,args.context_dim).to(device)
    generator = Generator(args.latent_dim,args.context_dim,args.output_dim,args.leaky_relu).to(device)
    # optimizer 
    optimizer_d = optim.SGD(discriminator.parameters(), lr=args.learning_rate, momentum=0.5)
    optimizer_g = optim.SGD(generator.parameters(), lr=args.learning_rate, momentum=0.5)
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer_d, 1 / 1.00004),
                  torch.optim.lr_scheduler.ExponentialLR(optimizer_g, 1 / 1.00004)]

    loss = train(generator, discriminator, optimizer_g, optimizer_d, schedulers, args.num_epochs, batch_size=args.batch_size)

    
    plt.figure(figsize=(12, 12))
    num_images = 10
    for i in range(10):
        # generate noise input

        z = generate_sample_noise(num_images, device)
        # create context vector for each class


        context = torch.nn.functional.one_hot(torch.ones(num_images, dtype=torch.long) * i, num_classes=10).to(
            device).type(torch.float)
        
        # generate image with the given noise input and context vector
        x = generator(z, context)
        # plot and save each generated image
        
    """   
        for j in range(num_images):
            plt.subplot(10, 10, 10 * i + 1 + j)
            #plt.axis('off')
            plt.imshow(x[j].data.cpu().numpy().reshape(28, 28), cmap='gray')
    plt.savefig('cgan.png')
    """ 
    #write_fake.add_image("mnist fake images",img_fake_grid,global_step=step)
    
    writer.close()
#writer.add_image('images', image, 0)
#writer.add_graph(model, images)
