from .base_network import *

class StudentNetwork(BaseNetwork):
    def __init__(self,
        in_dim  : int,
        out_dim : int,
        width   : int,
    ):
        super(StudentNetwork, self).__init__()

        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.width   = width

        self.layers = nn.Sequential(

            # nn.Linear(in_dim, width),
            # nn.ReLU(), # ReLU/Softplus/Tanh/Softsign/ELU
            # nn.Linear(width, 128),
            # # nn.LayerNorm(128),
            # nn.ReLU(), # ReLU/Softplus/Tanh/Softsign/ELU
            # # nn.Dropout(0.1),
            # # nn.Linear(256, 256),
            # # # nn.LayerNorm(256),
            # # nn.Tanh(),
            # # # nn.Dropout(0.1),
            # # nn.Linear(256, 256),
            # # # nn.LayerNorm(256),
            # # nn.Tanh(),
            # # # nn.Dropout(0.1),
            # # nn.Linear(256, 128),
            # # nn.LayerNorm(128),
            # # nn.PReLU(),
            # # nn.Dropout(0.1),
            # # nn.Linear(128, 128),
            # # # nn.LayerNorm(128),
            # # nn.PReLU(),
            # # nn.Dropout(0.1),
            # nn.Linear(128, 64),
            # # nn.LayerNorm(64),
            # nn.ReLU(),
            # # nn.Dropout(0.1),
            # nn.Linear(64, 32),
            # # nn.LayerNorm(32),
            # nn.ReLU(),
            # # nn.Dropout(0.1),
            # nn.Linear(32, 16),
            # # nn.LayerNorm(16),
            # nn.ReLU(),
            # # nn.Dropout(0.1),
            # nn.Linear(16, 8), ###
            # # nn.LayerNorm(8),
            # nn.ReLU(), ###
            # # nn.Dropout(0.1),
            # nn.Linear(8, 4),  ###
            # # nn.LayerNorm(4),
            # nn.ReLU(),
            # # nn.Dropout(0.1),
            # nn.Linear(4, out_dim),
            # # nn.Softsign(),


            # nn.Linear(in_dim, width),
            # nn.Softplus(),
            # nn.Linear(width, width),
            # nn.Softplus(),
            # nn.Linear(width, 512),
            # nn.Softplus(),
            # nn.Linear(512, 256),
            # nn.Softplus(),
            # nn.Linear(256, 128),
            # nn.Softplus(),
            # nn.Linear(128, 64),
            # nn.Softplus(),
            # nn.Linear(64, 32),
            # nn.Softplus(),
            # nn.Linear(32, 16),
            # nn.Softplus(),
            # nn.Linear(16, 8),
            # nn.Softplus(),
            # nn.Linear(8, out_dim),

            #
            # #
            # nn.Linear(in_dim, width),
            # nn.ReLU(),
            # nn.Linear(width, width),
            # nn.ReLU(),
            # nn.Linear(width, out_dim),

            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, out_dim)


        )

    def forward(self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)
