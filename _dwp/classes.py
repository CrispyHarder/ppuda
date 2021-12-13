import torch
import torch.nn as nn

class Decoder3x3(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder3x3, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, hidden_dim * 2, 1),
            nn.ELU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3),
            nn.ELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ELU(),
        )

        self.fc_mu = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.fc_var = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, 1)
        )

    def forward(self, input):
        x = self.decoder(input)
        mu = self.fc_mu(x)
        var = torch.FloatTensor([1e-3]) # Variable(torch.FloatTensor([1e-3]).cuda())  # F.softplus(self.fc_var(x))
        return mu