import torch 
from torch import nn 
from einops import rearrange

from embedding import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed


class PatchEmbed3D(nn.Module):

    """
    THis class is the "Entry Point" for a 3D video Transformer (like stable-video-diffusion or sora)

    It's job is to take a raw block of video latents (Time, Height, width) and chop it up into a 
    sequence of vectors ("tokens") that the Transformer can understand. 
    It also stamps these tokens with location data (Space and Time) so the model knows where and when 
    each piece belongs.
    """

    def __init__(self,
                 height=128,
                 width=128,
                 patch_size=2,
                 in_channels=16,
                 embed_dim=1536,
                 layer_norm=False,
                 bias=True,
                 interpolation_scale=1,
                 pos_embed_type="sincos",
                 temp_pos_embed_type="sincos",
                 pos_embed_max_size=192,
                 max_num_frames=64,
                 add_temp_pos_embed=False,
                 interp_condition_pos=False):
        
        """
            height: Image height,
            width: Image Width 
            patch_size: we will chop the video into 2x2 sequares.
            in_channels: THe input is not RGB but likely Latent Noise (16 channels) typical for Diffusion models.
            embed_dim: `1536` dimension is huge-large dim model.
            pos_embed_max_size: `192` we pre-calculate a massive  positional map (192x192 patches) to handle varied aspect ratio later.
        """

        super().__init__()
        num_patches = (height // patch_size) * (width // patch_size)
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        # THis is the most important line. A Conv with `kernel_size=patch_size` and `stride=patch_size` doesn't slide over pixel smoothly;
        # it "steps" over them without overlap. It perfectly converts every 2x2 block of pixels into a single vector of size `embed_dim` (1536)
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias
        )

        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        self.add_temp_pos_embed = add_temp_pos_embed

        # Calculate positional embeddings based on max size or default 
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches ** 0.5)

        
        if pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim,
                                                grid_size=grid_size,
                                                base_size=self.base_size,
                                                interpolation_scale=self.interpolation_scale)
            
            persistent = True if pos_embed_max_size else False
            # 'register_buffer': tells pyTorch "save this tensor with the model, but do not update it with gradient (it's fixed)."
            self.register_buffer("pos_embed", 
                                 torch.from_numpy(pos_embed).float().unsqueeze(0), 
                                 persistent=persistent)
            
            if add_temp_pos_embed and temp_pos_embed_type == "sincos":
                time_pos_embed = get_1d_sincos_pos_embed(embed_dim=embed_dim,
                                                         num_frames=max_num_frames)
                
                # The Time Map: Creates a separate 1D strip of embeddings for time steps (Frame1, Frame2, Frame3, ....)
                self.register_buffer("temp_pos_embed", 
                                     torch.from_numpy(time_pos_embed).float().unsqueeze(0), 
                                     persistent=True)
                
        self.pos_embed_type = pos_embed_type
        self.temp_pos_embed_type = temp_pos_embed_type
        self.interp_condition_pos = interp_condition_pos

    def cropped_pos_embed(self, 
                          height, 
                          width, 
                          ori_height,
                          ori_width):
        
        """Crops positional embeddings for SD3 Compatibility."""

        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")
        
        height = height // self.patch_size
        width = width // self.patch_size

        if ori_height and ori_width is not None:
            ori_height = ori_height // self.patch_size
            ori_width = ori_width // self.patch_size

            assert ori_height >= height, "The ori_height needs >= height."
            assert ori_width >= width, "The ori_width needs >= width."

        
           
        # 192 - 16 => 176 / 2 => 88
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2 

        
        # torch.Size([1, 36864, 1024]) -> torch.size[(1, 192, 192, 1024)]
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        # torch.size[(1, 192, 192, 1024)] -> torch.size[(1, 16, 16, 1024)]
        spatial_pos_embed = spatial_pos_embed[:,
                                                top : top + height, 
                                                left : left + width, 
                                                :]
        
        
        # torch.size[(1, 16, 16, 1024)] -> [1, 256, 1024]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        
        return spatial_pos_embed
            



        


    def forward_func(self,
                     latent,
                     time_index=0,
                     ori_height=None,
                     ori_width=None):
        
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
            
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        bs = latent.shape[0]
        temp = latent.shape[2]

        latent = rearrange(latent, 'b c t h w -> (b t) c h w')
        latent = self.proj(latent)
        latent = latent.flatten(2).transpose(1, 2)  # (bt)chw -> (bt)NC, N-> sequence_length

        if self.layer_norm:
            latent = self.norm(latent)

        if self.pos_embed_type == "sincos":

            # spatial position embedding, Interpolate or crop positional embeddings as needed
            if self.pos_embed_max_size:
                pos_embed = self.cropped_pos_embed(height=height,
                                                   width=width,
                                                   ori_height=ori_height,
                                                   ori_width=ori_width)
                
            
            if self.add_temp_pos_embed and self.temp_pos_embed_type == "sincos":
                latent_type = latent.dtype 
                latent = latent + pos_embed
                # [16, 256, 1024] -> [512, 8, 1024]
                latent = rearrange(latent, '(b t) n c -> (b n) t c', t=temp)
                
                latent = latent + self.temp_pos_embed[:, 
                                                    time_index:time_index + temp, 
                                                    :]
                latent = latent.to(latent_type)
                # [512, 8, 1024] -> [2, 8, 256, 1024]
                latent = rearrange(latent, '(b n) t c -> b t n c', b=bs)
                
        return latent










    def forward(self, latent):

        """
        Arguments:
            past_condition_latents (Torch.FloatTensor): The past latent during the generation
            flatten_input (bool): True indicate flatten the latent into 1D sequence.
        """

        if isinstance(latent, list):
            output_list = []

            for latent_ in latent:
                assert TypeError

        else:
            hidden_state = self.forward_func(latent)
            # torch.Size([2, 8, 256, 1024]) --> torch.Size([2, 2048, 1024])
            hidden_state = rearrange(hidden_state, "b t n c -> b (t n) c")
            
            return hidden_state


            

        

        
if __name__ == "__main__":
    model = PatchEmbed3D(height=32, 
                         width=32,
                         embed_dim=1024,
                         in_channels=16,
                         add_temp_pos_embed=True,
                         )
    
    latent = torch.randn(2, 16, 8, 32, 32)
    out = model(latent)
    print(out.shape)

    


        

