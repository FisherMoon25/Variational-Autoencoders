from torch import nn
import torch

class VAE(nn.Module):

    def __init__(self,in_channel,hidden_dims,latent_dim):
        super(VAE,self).__init__()
        #TO DO
        
        #Build Encoder, Use Convolutional layers
        model = []
        
        for hidden_dim in hidden_dims:
            model.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=hidden_dim,
                        kernel_size=(3,3)
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU()     
                )
            )
            in_channel = hidden_dim
        
        self.encoder = nn.Sequential(*model)
        
        self.mu = nn.Linear(hidden_dims[-1],latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1],latent_dim)

        #Build Decoder
        model = []
        hidden_dims.reverse()
        
        
        #TO DO  : Finish building the decoder
        
        for hidden_dim in hidden_dims:
            model.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=5
                    )
                )
            )

        self.decoder = nn.Sequential(*model)
        
        
    
    def encode(self,x):
        #TO DO
        res = self.encoder(x) 
        mu = self.mu(res)
        log_var = self.log_var(res)
        return [mu,log_var]
    
    def decode(self,z):
        #TO DO
        return self.decoder(z)
    
    def reparamterize(self,mu,log_var):
        """"
        Reparameterization trick to sample from N(mu,var)
        """
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    
    def forward(self,x):
        
        mu,log_var = self.encoder(x)
        
        z = self.reparamterize(mu,log_var)
        
        decoded = self.decoder(z)
        
        return decoded,mu,log_var
    
    
    def loss_function(self):
        """
        Compute VAE Loss function
        
        """
        
        #TO DO 
        return
        
        
    
    