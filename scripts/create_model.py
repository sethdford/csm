import torch
import torch.nn as nn
import os

def create_model():
    # Model dimensions
    embed_dim = 2048
    num_layers = 8
    num_heads = 8
    num_kv_heads = 2
    vocab_size = 128_256
    
    # Create state dict
    state_dict = {}
    
    # Token embedding
    state_dict['token_embedding.weight'] = torch.randn(vocab_size, embed_dim)
    
    # Transformer layers
    for i in range(num_layers):
        layer_prefix = f'layer_{i}'
        
        # Self attention
        state_dict[f'{layer_prefix}.self_attn.q_proj.weight'] = torch.randn(embed_dim, embed_dim)
        state_dict[f'{layer_prefix}.self_attn.q_proj.bias'] = torch.randn(embed_dim)
        
        state_dict[f'{layer_prefix}.self_attn.k_proj.weight'] = torch.randn(embed_dim, embed_dim)
        state_dict[f'{layer_prefix}.self_attn.k_proj.bias'] = torch.randn(embed_dim)
        
        state_dict[f'{layer_prefix}.self_attn.v_proj.weight'] = torch.randn(embed_dim, embed_dim)
        state_dict[f'{layer_prefix}.self_attn.v_proj.bias'] = torch.randn(embed_dim)
        
        state_dict[f'{layer_prefix}.self_attn.o_proj.weight'] = torch.randn(embed_dim, embed_dim)
        state_dict[f'{layer_prefix}.self_attn.o_proj.bias'] = torch.randn(embed_dim)
        
        # Layer norms
        state_dict[f'{layer_prefix}.norm1.weight'] = torch.ones(embed_dim)
        state_dict[f'{layer_prefix}.norm1.bias'] = torch.zeros(embed_dim)
        
        state_dict[f'{layer_prefix}.norm2.weight'] = torch.ones(embed_dim)
        state_dict[f'{layer_prefix}.norm2.bias'] = torch.zeros(embed_dim)
        
        # MLP
        intermediate_dim = embed_dim * 4
        state_dict[f'{layer_prefix}.mlp.w1.weight'] = torch.randn(intermediate_dim, embed_dim)
        state_dict[f'{layer_prefix}.mlp.w1.bias'] = torch.randn(intermediate_dim)
        
        state_dict[f'{layer_prefix}.mlp.w2.weight'] = torch.randn(embed_dim, intermediate_dim)
        state_dict[f'{layer_prefix}.mlp.w2.bias'] = torch.randn(embed_dim)
    
    # Final layer norm
    state_dict['norm.weight'] = torch.ones(embed_dim)
    state_dict['norm.bias'] = torch.zeros(embed_dim)
    
    # Output projection
    state_dict['output.weight'] = torch.randn(vocab_size, embed_dim)
    state_dict['output.bias'] = torch.randn(vocab_size)
    
    # Save weights
    os.makedirs('models', exist_ok=True)
    torch.save(state_dict, 'models/llama-1B.pth')
    print("Model weights saved to models/llama-1B.pth")

if __name__ == '__main__':
    create_model() 