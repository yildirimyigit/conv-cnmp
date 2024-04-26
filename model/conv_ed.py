import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvED(nn.Module):
    def __init__(self, nof_past_frames, nof_future_frames, filter_sizes=[1024,512,256,128]):
        super(ConvED, self).__init__()
        self.nof_past_frames = nof_past_frames
        self.nof_future_frames = nof_future_frames
        nof_layers = len(filter_sizes)

        # Encoder
        layers = []
        layers.append(nn.Conv2d(nof_past_frames, filter_sizes[0], kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU)
        layers.append(nn.MaxPool2d(2))
        for i in range(1, nof_layers):
            layers.append(nn.Conv2d(filter_sizes[i-1], filter_sizes[i], kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU)
            layers.append(nn.MaxPool2d(2))

        self.encoder = nn.Sequential(*layers)
        
        # Decoder
        layers = []
        for i in range(nof_layers, 1):
            layers.append(nn.ConvTranspose2d(filter_sizes[i], filter_sizes[i-1], kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU)
        layers.append(nn.ConvTranspose2d(filter_sizes, nof_future_frames, kernel_size=3, stride=2, padding=1, output_padding=1))
        layers.append(nn.Sigmoid())  # Sigmoid so output is between 0 and 1

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):  # x: (batch_size, nof_past_frames, H, H)
        latent = self.encoder(x)
        decoded_frames = self.decoder(latent)  # x: (batch_size, nof_future_frames, H, H)
        return decoded_frames

    def loss(self, prediction, target):
        return F.mse_loss(prediction, target)

