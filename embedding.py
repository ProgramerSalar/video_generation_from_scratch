import torch, math
import numpy as np 
from typing import Optional
from diffusers.models.activations import get_activation

def get_1d_sincos_pos_embed(
        embed_dim,
        num_frames,
        cls_token=True,
        extra_tokens=7
   ):
    
    t = np.arange(num_frames, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim=embed_dim,
                                                  pos=t)   # (T, D)
    
    if cls_token and extra_tokens > 0:
        # pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]),
        #                             pos_embed], axis=0)

        # we create a "blank" row of zeros. This acts as placeholder, a seperate learnable vector to added in this position.
        cls_token_zero = np.zeros([extra_tokens, embed_dim])
        pos_embed = np.concatenate([cls_token_zero, pos_embed], axis=0)
        
        
    return pos_embed



def get_1d_sincos_pos_embed_from_grid(embed_dim,
                                      pos):
    
    """
    This code generate a unique vector (pattern) for every postion in 
    sequence so the model knows the "word A" come before "word B"

    embed_dim: output dim for each position 
    pos: a list of positions to be encoded: size (M,) out: (M, D)
    """

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2.")
    

    # create indices for half the dimensions
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    # normalize these indices to range from 0 to 1.
    omega /= embed_dim / 2.0
    
    # Raises 10,000 to these powers. This creates a curve where low indices have high frequencies (change rapidly) 
    # and high indices have low frequencies (change slowly).
    # 10k ** omega  => 10k ** 0. => 1.0
    # 1.0 / 1.0 => 1.0

    # 10k ** 0.5 => 100, 1/100 => 0.01
    omega = 1.0 / 10000**omega  # (D/2,)
    
    # flatten the position in one-dim
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d -> md", pos, omega)  # (M, D/2)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1) # (M, D)
    return emb


    
def get_2d_sincos_pos_embed(
        embed_dim,
        grid_size,
        cls_token=False,
        extra_tokens=0,
        interpolation_scale=1.0,
        base_size=16
    ):

    """
    This function is designed for Vision Transformer (ViT). while the previous function handled 
    1D sequences (like time), this one handle 2D images (height and width)
    It creates a "grid" of postional embeddings so the model understands where a specific image 
    patch is located (e.g., "top-left" vs. "bottom-right")

    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or 
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim=embed_dim,
                                                  grid=grid)
    
    if cls_token and extra_tokens > 0:
        # we create a "blank" row of zeros. This acts as placeholder, a seperate learnable vector to added in this position.
        cls_token_zero = np.zeros([extra_tokens, embed_dim])
        pos_embed = np.concatenate([cls_token_zero, pos_embed], axis=0)

    return pos_embed


    
def get_2d_sincos_pos_embed_from_grid(embed_dim,
                                      grid):
    
    if embed_dim % 2 != 0:
        raise ValueError("embed dim must be divisible by 2")
    

    # use half of dimension to encode grid_h 
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim=embed_dim // 2,
                                              pos=grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim=embed_dim // 2,
                                              pos=grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)    # (H*W, D)
    return emb 



def get_timestep_embedding(
        timesteps: torch.Tensor,
        embedding_dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1,
        scale: float = 1,
        max_period: int = 10000
    ):

    """
    This function is a cornerstone of DDPM
    In the DDPM models, the nural network needs to know how much noise i currently in the image.
    we tell it this by passing a "timestep" (e.g., t=500 out of 1000). This function converts 
    that single number t into a rich vector (embedding) that the nural network can understand.

    timesteps: a 1-D Tensor of N indices, one per batch element. These may be functional.
    embedding_dim: the dimension of the output. 
    max_period: controls the minimum frequency of the embeddings. :return: an [N x dim] Tensor of positional embeddings.
    downscale_freq_shift: This is a subtile tuning knob. By default (shift=1), 
                            it adjust the divisor so the frequencies span excatly from 1 down to 1/max_period
    """

    # Ensure `timesteps` is a flat list of numbers (e.g., [200, 450, 10]) not a matrix.
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2 
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    # (N) -> (N, 1), (D/2) -> (1, D/2)
    emb = timesteps[:, None].float() * emb[None, :]
    
    # scale embeddings 
    emb = scale * emb 
    
    # concat sine and cosine embeddings 
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings 
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]],
                        dim=-1)
        
    # zero pad 
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    
    return emb


class Timesteps(torch.nn.Module):

    def __init__(self,
                 num_channels: int,
                 flip_sin_to_cos: bool, 
                 downscale_freq_shift: float):
        
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):

        t_emb = get_timestep_embedding(
            timesteps=timesteps,
            embedding_dim=self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift
        )
        
        return t_emb


class TimestepEmbedding(torch.nn.Module):

    def __init__(
            self,
            in_channels: int,
            time_embed_dim: int,
            act_fn: str = "silu",
            sample_proj_bias=True
    ):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.act = get_activation(act_fn)
        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim, sample_proj_bias)


    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample
    

class TextProjection(torch.nn.Module):

    def __init__(self,
                 in_features,
                 hidden_size,
                 act_fn="silu"):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features=in_features,
                                        out_features=hidden_size,
                                        bias=True)
        self.act_1 = get_activation(act_fn)
        self.linear_2 = torch.nn.Linear(in_features=hidden_size,
                                        out_features=hidden_size,
                                        bias=True)
        

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
    

class CombinedTimestepConditionEmbeddings(torch.nn.Module):

    """
    THis class is "Fusion" module. It combines two very different types of information into a single vector: 
        1. Time: "How much noise is in the image?" (via `timestep`)
        2. Text: "What is the image about?" (via `pooled_projection` from a text encoder like CLIP or T5)

    This technique is common in newer models (like stable Diffusion XL or flux) where the model 
    needs to know both the diffusion stage and the global context of the prompt simultaneously.
    """




    def __init__(self,
                 embedding_dim,
                 pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256,
                                   flip_sin_to_cos=True,
                                   downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256,
                                                   time_embed_dim=embedding_dim)
        self.text_embedder = TextProjection(pooled_projection_dim,
                                            embedding_dim,
                                            act_fn="silu")
        

    def forward(self, timestep, pooled_projection):

        # Convert raw time `t` --> Sin/Cos waves --> learned Time vector.
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))    # (N, D)

        # Context Text Summary --> Learned Text vector
        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = timesteps_emb + pooled_projections

        return conditioning
    

class CombinedTimestepEmbeddings(torch.nn.Module):

    def __init__(self,
                 embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep):
        timesteps_proj = self.time_proj(timestep)
        timestep_emb = self.timestep_embedder(timesteps_proj)
        return timestep_emb




if __name__ ==  "__main__":
    # get_1d_sincos_pos_embed(embed_dim=512,
    #                         num_frames=16,
    #                         )

    # get_1d_sincos_pos_embed_from_grid(embed_dim=4,
    #                                   pos=np.array([1]))

    # get_2d_sincos_pos_embed(embed_dim=4,
    #                         grid_size=(16, 16))

    timestep = torch.randn([10])
    # get_timestep_embedding(timesteps=timestep,
    #                        embedding_dim=4,
    #                        )

    # model = Timesteps(num_channels=128,
    #           flip_sin_to_cos=False,
    #           downscale_freq_shift=1.0)
    
    # out = model(timestep)
    # print(out.shape)
    
    # -----------------------------------------------
    # raw_dim = 320 
    # # we want to project this to size 1280 (common in stable diffusion)
    # learned_dim = 1280
    # model = TimestepEmbedding(in_channels=raw_dim,
    #                           time_embed_dim=learned_dim,
    #                           )
    # raw_embedding = torch.randn(2, 320)
    # out = model(raw_embedding)
    # print(out.shape)
    # -----------------------------------------------------------
    t = torch.tensor([500, 800])
    # Text: Batch size 2, size 768 (vectors for "Dog" and "Cat")
    text_pooled = torch.randn(2, 768)

    model = CombinedTimestepConditionEmbeddings(embedding_dim=1023,     # Target size
                                                pooled_projection_dim=768  # Input text size 
                                                )
    out = model(t, text_pooled)
    print(out.shape)
