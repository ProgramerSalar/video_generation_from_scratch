import torch 
from torch import nn 
from diffusers.models.activations import GELU


class FeedForward(nn.Module):

    def __init__(self,
                 dim: int,
                 act_fn: str = "gelu",
                 dropout: float = 0.0,
                 mult: int = 4,
                 bias: bool = True,
                 final_dropout: bool = True
                 ):
        super().__init__()
        self.inner_dim = int(mult * dim)
        if act_fn == "gelu":
            act_fn = GELU(dim_in=dim,
                          dim_out=self.inner_dim,
                          bias=bias)

        self.network = nn.ModuleList([])
        # 1. projection input 
        self.network.append(act_fn)
        # 2. Dropout 
        self.network.append(nn.Dropout(dropout))
        # 3. Projection output 
        linear_layer = nn.Linear(in_features=self.inner_dim,
                                 out_features=dim,
                                 bias=bias)
        self.network.append(linear_layer)
        # 4. Dropout 
        if final_dropout: 
            self.network.append(nn.Dropout(dropout))

    def forward(self,
                hidden_state: torch.Tensor) -> torch.Tensor:
        

        for module in self.network:
            hidden_state = module(hidden_state)

        return hidden_state
    






if __name__ == "__main__":
    # ffd = FeedForward(dim=128)
    # print(ffd)
    # x = torch.randn(2, 128)
    # out = ffd(x)
    # print(out.shape)
    # ---------------------------------------------------------
    pass




