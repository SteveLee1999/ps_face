import torch
import torch.nn.functional as F
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.Conv0 = torch.nn.Conv2d(image.get_shape()[-1], self.df_dim, 5, strides=[1,2,2,1])
        self.Conv1 = torch.nn.Conv2d(self.df_dim, self.df_dim * 2, 5, strides=[1,2,2,1])
	    self.Conv2 = torch.nn.Conv2d(self.df_dim * 2, self.df_dim * 4, 5, strides=[1,2,2,1])           
        self.Conv3 = torch.nn.Conv2d(self.df_dim * 4, self.df_dim * 8, 5, strides=[1,2,2,1])

    def forward(self, image):
        h0 = self.Conv0(image)
        h0 = F.leaky_relu(h0, 0.2, inplace=True)
        h1 = self.Conv1(h0)
        h1 = nn.LayerNorm(h1)
        h1 = F.leaky_relu(h1, 0.2, inplace=True)
        h2 = self.Conv2(h1)
        h2 = nn.LayerNorm(h2)
        h2 = F.leaky_relu(h2, 0.2, inplace=True)
        h3 = self.Conv3(h2)
        h3 = nn.LayerNorm(h3)
        h3 = F.leaky_relu(h3, 0.2, inplace=True)
        h4 = torch.reshape(h3, [self.batch_size, -1])
        h4 = nn.linear(h4)
        ret = F.sigmoid(h4)
        
        return ret