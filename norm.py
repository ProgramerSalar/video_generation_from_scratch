import torch 
from torch import nn 
import numbers
from typing import Optional, Tuple
from diffusers.utils import is_torch_version


if is_torch_version(">=", "2.1.0"):
    LayerNorm = nn.LayerNorm

else:
    assert print("make sure the torch version is grater then `2.1.0`")



class RMSNorm(nn.Module):

    def __init__(self,
                 dim,
                 eps: float,
                 elementwise_affine: bool = True):
        
        super().__init__()
        self.eps = eps 

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            
    def forward(self, hidden_state):
        input_dtype = hidden_state.dtype 

        # calculate the mean to the last dimension
        varience = hidden_state.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_state = hidden_state * torch.sqrt(varience + self.eps)
        
        if self.weight is not None:
            # convert into half-percision if necessary 
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_state = hidden_state.to(self.weight.dtype)

            hidden_state = hidden_state * self.weight
            
        hidden_state = hidden_state.to(input_dtype)
        return hidden_state
    


class AdaptiveLayerGroupNormContinuous(nn.Module):

    def __init__(self,
                 conditional_embedding_dim: int,
                 embedding_dim: int,
                 bias: bool = True,
                 norm_type="layer_norm",
                 eps=1e-5,
                 elementwise_affine=True):
        super().__init__()

        self.linear = nn.Linear(in_features=conditional_embedding_dim,
                                out_features=embedding_dim * 2,
                                bias=bias)
        self.silu = nn.SiLU()
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)

    def forward(self, 
                x: torch.Tensor,
                conditional_embedding: torch.Tensor,            
                ):
        
    
        emb = self.linear(self.silu(conditional_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x 
    


class AdaLayerNormContinuous(nn.Module):

    def __init__(self,
                 embedding_dim: int, 
                 conditional_embedding_dim,
                 bias
                 ):
        super().__init__()

        self.linear = nn.Linear(in_features=conditional_embedding_dim,
                                out_features=embedding_dim * 2,
                                bias=bias)
        self.silu = nn.SiLU()

    def forward(self, 
                x: torch.Tensor,
                emb: torch.Tensor
                ):
        
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
        
    
        

class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None):
        super().__init__()
        self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward_with_pad(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
        hidden_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_length: [[20, 30], [30, 40], [50, 60]]
        # x: [bs, seq_len, dim]
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)

        emb = self.linear(self.silu(emb))
        batch_emb = torch.zeros_like(x).repeat(1, 1, 6)
    
        i_sum = 0
        num_stages = len(hidden_length)
        for i_p, length in enumerate(hidden_length):
            batch_emb[:, i_sum:i_sum+length] = emb[i_p::num_stages][:,None]
            i_sum += length

        batch_shift_msa, batch_scale_msa, batch_gate_msa, batch_shift_mlp, batch_scale_mlp, batch_gate_mlp = batch_emb.chunk(6, dim=2)
        x = self.norm(x) * (1 + batch_scale_msa) + batch_shift_msa
        return x, batch_gate_msa, batch_shift_mlp, batch_scale_mlp, batch_gate_mlp

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
        hidden_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_length is not None:
            return self.forward_with_pad(x, timestep, class_labels, hidden_dtype, emb, hidden_length)
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward_with_pad(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
        hidden_length: Optional[torch.Tensor] = None,
    ):
        emb = self.linear(self.silu(emb))
        batch_emb = torch.zeros_like(x).repeat(1, 1, 3)

        i_sum = 0
        num_stages = len(hidden_length)
        for i_p, length in enumerate(hidden_length):
            batch_emb[:, i_sum:i_sum+length] = emb[i_p::num_stages][:,None]
            i_sum += length

        batch_shift_msa, batch_scale_msa, batch_gate_msa = batch_emb.chunk(3, dim=2)

        x = self.norm(x) * (1 + batch_scale_msa) + batch_shift_msa

        return x, batch_gate_msa
        


if __name__ == "__main__":
    # Norm = RMSNorm(dim=128,
    #                eps=1e-6)
    
    # x = torch.randn(2, 128, dtype=torch.float16)
    # out = Norm(x)
    # print(out.shape)
    # ------------------------------------------------
    embedding_dim = 4 
    conditional_dim = 3
    # Norm = AdaptiveLayerGroupNormContinuous(conditional_embedding_dim=conditional_dim,
    #                               embedding_dim=embedding_dim)
    
    # # batch to 2 image with 3 tokens 
    # x = torch.randn(2, 3, embedding_dim)
    # cond_x = torch.randn(2, conditional_dim)

    # out = Norm(x, cond_x)

    # -----------------------------------------------------------

    norm = AdaLayerNormZero(embedding_dim=embedding_dim)
    x = torch.randn(2, embedding_dim)
    out = norm(x)

    