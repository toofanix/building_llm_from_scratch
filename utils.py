import torch
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), "d_out must be divisible by num_heads" # remainder should be zero

        self.d_out  = d_out
        self.num_heads = num_heads
        self.head_dim = d_out//num_heads # This is the quotient

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = dropout
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # the matrices are split by adding a num_heads dimension
        # all of them will have the shape (batchs, context_length, num_heads, dimension of each head)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Now to calculate the attention weights we need to perform Matmul
        # For this we need (context_length, head_dim)
        # So transpose num_tokens, and num_heads for keys, queries and values
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3) # tranpose b/c the dimensions have to align for matmul (m X n) @ (n X m)
        # the mask-fill is split into 2 lines here.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Convert scores to weights
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Caluclate the context vector
        context_vec = (attn_weights @ values).transpose(1, 2) # convert back to batches, num_tokens, num_heads, head_dim

        # combine the 2 heads
        # Contiguous memory refers to a block of memory addresses that are sequential and adjacent. 
        # Think of it like a line of houses on a street, all next to each other, with no gaps. 
        # This arrangement is crucial for efficiency in many computing tasks because it allows the computer to access data more quickly.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # note that d_out = num_heads * head_dim

        # Add an optional linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec


