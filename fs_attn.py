import torch 
from torch import nn 
from diffusers.models.activations import GELU
from einops import rearrange
from flash_attn.bert_padding import index_first_axis
from flash_attn.flash_attn_interface import flash_attn_varlen_func



class VariableLengthFlashSelfAttentionWithT5Mask(nn.Module):

    """
    It calculates self-attention between `query` sequence (like text) and an `encoder` sequence (like image Feature)
    It handle variable sequence length efficiently by "packing" tokens (removing padding)
    Before passing them to optimized `flash_attention_variable_length_function` 
    and Then "unpack" them back to their original batch shapes.
    """
    def __init__(self):
        pass 


    def __call__(self,
                 query,
                 key, 
                 value,
                 encoder_query,
                 encoder_key,
                 encoder_value,
                 heads,
                 scale,
                 hidden_length,
                 encoder_attention_mask):
        
        batch_size = query.shape[0]
        output_hidden = torch.zeros_like(query)
        output_encoder_hidden = torch.zeros_like(encoder_query)
        encoder_length = encoder_query.shape[1]


        num_stages = len(hidden_length)
        qkv_list = []
    
    
        # 3 x [batch_size, seq_len, head, head_dim] 
        # [batch_size, seq_len, 3, head, head_dim] 
        encoder_qkv = torch.stack([encoder_query, encoder_key, encoder_value], dim=2)   # [2, 4, 3, 4, 16]
        qkv = torch.stack([query, key, value], dim=2)   # [2, 5, 3, 4, 16]

        i_sum = 0
        
        for index, length in enumerate(hidden_length):
        
            # 0:, 1:
            # [start:stop:step]
            # [2, 4, 3, 4, 16] -> [1, 4, 3, 4, 16]
            encoder_qkv_tokens = encoder_qkv[index::num_stages]
            # print(encoder_qkv_tokens.shape)
            
            
            # [2, 5, 3, 4, 16] -> [1, 3, 3, 4, 16], [1, 5, 3, 4, 16]
            qkv_tokens = qkv[index::num_stages, i_sum:i_sum+length]
            # print(qkv_tokens.shape)
            # print(qkv[:, i_sum].shape)
            
            # after (concat) -> torch.Size([1, 7, 3, 4, 16]),torch.Size([1, 9, 3, 4, 16])
            concat_qkv_tokens = torch.cat([encoder_qkv_tokens, qkv_tokens], dim=1)
            
            indices = encoder_attention_mask[index]['indices']
            # print(indices.shape)

            # # torch.Size([1, 7, 3, 4, 16]),torch.Size([1, 9, 3, 4, 16]) --> torch.Size([7, 3, 4, 16]),torch.Size([9, 3, 4, 16])
            tensor_rear = rearrange(concat_qkv_tokens, 
                                    'b s qkv h hd -> (b s) qkv h hd')
            
            # (after applied flash-attn) --> torch.Size([7, 3, 4, 16]),torch.Size([9, 3, 4, 16])
            apply_flash = index_first_axis(tensor_rear, indices)
            qkv_list.append(apply_flash)
            # i_sum+=length

        
        # [7, 9]
        token_lengths = [x_.shape[0] for x_ in qkv_list]
        # torch.Size([7, 3, 4, 16]),torch.Size([9, 3, 4, 16]) --> torch.Size([16, 3, 4, 16])
        qkv = torch.cat(qkv_list, dim=0)
        # torch.Size([16, 3, 4, 16]) -> 3 X torch.Size([16, 4, 16])
        unb_query, unb_key, unb_value = qkv.unbind(1)
        
        # cumulative seq_len
        # -->  [tensor([7]), tensor([9])]
        cu_seqlens = [x_['seqlens_in_batch'] for x_ in encoder_attention_mask]
        cu_seqlens = torch.cat(cu_seqlens, dim=0)

        max_seqlen_q = cu_seqlens.max().item() # tensor([9])
        max_seqlen_k = max_seqlen_q        # tensor([9])

       
        # [tensor([7]), tensor([9])] ---> tensor([ 7., 16.])
        cum_sum = torch.cumsum(cu_seqlens, dim=0, dtype=torch.float32)

        # tensor([ 7., 16.])  ---> tensor([ 0.,  7., 16.])
        cu_seqlen_q = nn.functional.pad(cum_sum, pad=(1, 0))    
        cu_seqlen_k = cu_seqlen_q.clone()

        output = flash_attn_varlen_func(
            q=unb_query,
            k=unb_key,
            v=unb_value,
            cu_seqlens_q=cu_seqlen_q,
            cu_seqlens_k=cu_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            causal=False,
            softmax_scale=scale

        )


            


if __name__ == "__main__":

    attention_layer = VariableLengthFlashSelfAttentionWithT5Mask()

    # Hyperparameter 
    batch_size = 2
    num_heads = 4 
    head_dim = 16 
    total_dim = num_heads * head_dim # 64

        ############################### Understand the no of heads 
        # [ head 0 -> might look at the first 16 number (texture) ]
        # [ head 1 -> might look at the next 16 number (shape)]
        # [ head 2 -> might look at the next 16 (color) ]
        # [ head 3 -> might look at the final 16 (Context)]

        ############################### Undarstand the encoder_length 
        # [`4` means that image was split into 4 patches (e.g -> Top-left,
        #                                                        Top-right, Bottom-left, Bottom-Right)]


    # <<<<<<<<<<<<<<<<<<< The 4D shape [Batch, seq_len, Heads, Head_dim] -> [2, 4, 4, 16]  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 1. Dimension 0: `batch_size` --> (2)
    #     layer 1: The "Dog" Image Wrapper 
    #     layer 2: The "Cat" Image Wrapper
    #
    # 2. Dimension 1: `encoder_len` -> (4)
    #           Inside the "Dog" Wrapper, we have 4 distinct vectors.
    #       Item 0: The Dog's Ear (patch 1)
    #       Item 1: The Dog's Nose (patch 2)
    #       Item 2: The Dog's Leg (patch 3)
    #       Item 3: The Dog's Tail (patch 4)
    # 
    # 3. Dimension 2: `num_heads` -> (4)
    #       Let's look at **Item 0 (The Dogs Ear)**  -> it is not just one vector. it split into 4 "viewpoints".                                                                                  
    #                                                                               "viewpoints 0", "viewpoints 1", "viewpoints 2", "viewpoints 3"
    # 
    # 4. Dimension 3: `head_dim` -> (16)
    #       Inside Viewpoint 0 of The Dog's Ear, there is a list of 16 floting-point numbers.
    # 

    # <------------------ Create a Image Encoder 
    # Input Image [2, 3, 16, 16] --> q,k,w [2, 4, 4, 16]
    encoder_len = 4 

    # shape: [Batch_size, seq_len, Heads, Head_dim] -> [2, 4, 4, 16]
    enc_q = torch.randn(batch_size, encoder_len, num_heads, head_dim) 
    enc_k = torch.randn(batch_size, encoder_len, num_heads, head_dim) 
    enc_v = torch.randn(batch_size, encoder_len, num_heads, head_dim) 

    # <------------------ Create Query (Text) Tensors ------------------------->
    max_text_len = 5  # we allocated 5 slots because the longest sentence in the batch has 5 words.

    # shape [2, 5, 4, 16]
    q = torch.randn(batch_size, max_text_len, num_heads, head_dim)
    k = torch.randn(batch_size, max_text_len, num_heads, head_dim)
    v = torch.randn(batch_size, max_text_len, num_heads, head_dim)

    # for actual text length for each sample in the batch.
    hidden_length = [3, 5]

    
    # THis usually comes from a tokenizer or data collector.
    # IT's tells the code how to "pack" the data (remove padding)
    encoder_attention_mask = [
        {
            # Sample A: 4 image tokens + 3 text tokens = 7 tokens
            # It's maps the flattened valid tokens to their positions in the unpadded stream.
            'indices': torch.tensor([0, 1, 2, 3, 4, 5, 6]),
            'seqlens_in_batch':torch.tensor([7])
        },
        {
            # Sample B: 4 image tokens + 5 text tokens = 9 tokens
            'indices': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            'seqlens_in_batch': torch.tensor([9])
        }
    ]


    out = attention_layer.__call__(query=q,
                                   key=k,
                                   value=v,
                                   encoder_query=enc_q,
                                   encoder_key=enc_k,
                                   encoder_value=enc_v,
                                   heads=4,
                                   scale=1.0,
                                   hidden_length=hidden_length,
                                   encoder_attention_mask=encoder_attention_mask)