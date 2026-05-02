import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

#data: smb, lr: 0.001, epoch: 10000, weight decay: 100 step, gamma:0.01, parameters: 1365607
# vae 3

class Encoder(nn.Module):
    def __init__(self, num_features=7):
        super(Encoder, self).__init__()
        # Encoder
        self.cnv1 = nn.Conv2d(num_features, 256, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.cnv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.cnv3 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.cnv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(512, 256)
        self.batch_norm5 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(256, 64)

    def forward(self, x):
        x = F.leaky_relu(self.batch_norm1(self.cnv1(x)))
        x = F.leaky_relu(self.batch_norm2(self.cnv2(x)))
        x = F.leaky_relu(self.batch_norm3(self.cnv3(x)))
        x = F.leaky_relu(self.batch_norm4(self.cnv4(x)))
        x = x.reshape(-1, 512)
        x = F.leaky_relu(self.batch_norm5(self.fc1(x)))
        return self.fc2(x), self.fc3(x)

class Decoder(nn.Module):
    def __init__(self, num_features=7):
        super(Decoder, self).__init__()
        # Decoder
        self.fc1 = nn.Linear(64, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.cnv1 = nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.cnv2 = nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.cnv3 = nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.cnv4 = nn.ConvTranspose2d(256, num_features, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = x.view(-1, 32, 4, 4)
        x = F.relu(self.batch_norm3(self.cnv1(x)))
        x = F.relu(self.batch_norm4(self.cnv2(x)))
        x = F.relu(self.batch_norm5(self.cnv3(x)))
        x = self.cnv4(x)
        x = F.softmax(x,dim=1)
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        y = self.reparameterize(mu, logvar)
        z = self.decoder(y)
        return z, mu, logvar