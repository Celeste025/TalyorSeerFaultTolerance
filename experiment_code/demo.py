import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ====== Mini Transformer Block (带 hook 支持) ======
class MiniTransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ff=256):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.ffn1 = nn.Linear(d_model, d_ff)
        self.ffn2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 保存中间输出
        self.intermediate_outputs = {}

    def forward(self, x):
        B, T, D = x.shape

        # --- Q/K/V ---
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        self.intermediate_outputs['q_proj'] = q
        self.intermediate_outputs['k_proj'] = k
        self.intermediate_outputs['v_proj'] = v

        # --- Multihead Attention ---
        q_ = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k_ = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_ = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q_ @ k_.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = (attn_weights @ v_).transpose(1, 2).contiguous().view(B, T, D)
        self.intermediate_outputs['attn_out'] = attn_out

        out1 = self.o_proj(attn_out)
        x1 = self.norm1(x + out1)
        self.intermediate_outputs['post_attn'] = x1

        # --- FeedForward ---
        ff = F.relu(self.ffn1(x1))
        ff_out = self.ffn2(ff)
        x2 = self.norm2(x1 + ff_out)
        self.intermediate_outputs['ffn'] = ff_out
        self.intermediate_outputs['block_output'] = x2

        return x2

# ====== 数据准备 ======
torch.manual_seed(0)
B, T, D = 1, 20, 64
x = torch.randn(B, T, D)
block = MiniTransformerBlock(D, n_heads=4)
block.eval()

# ====== 注错 hook (可控) ======
def make_injection_hook(token_idx, hidden_idx, new_value):
    def hook(module, input, output):
        output = output.clone()
        output[0, token_idx, hidden_idx] = new_value
        return output
    return hook

# 记录 clean 输出
with torch.no_grad():
    clean_out = block(x)
    clean_intermediates = block.intermediate_outputs.copy()

# 注入错误
token_to_modify = 3
hidden_to_modify = 10
injected_value = 50

hook_handle = block.v_proj.register_forward_hook(
    make_injection_hook(token_to_modify, hidden_to_modify, injected_value)
)

with torch.no_grad():
    corrupted_out = block(x)
    corrupted_intermediates = block.intermediate_outputs.copy()

hook_handle.remove()

# ====== 可视化函数 ======
def plot_module_outputs(clean_dict, corrupted_dict, modules):
    n_modules = len(modules)
    fig, axs = plt.subplots(n_modules, 3, figsize=(15, 3*n_modules))
    for i, mod_name in enumerate(modules):
        clean = clean_dict[mod_name].squeeze().cpu().numpy()
        corrupted = corrupted_dict[mod_name].squeeze().cpu().numpy()
        diff = corrupted - clean

        im0 = axs[i,0].imshow(clean, aspect='auto', cmap='viridis')
        axs[i,0].set_title(f"{mod_name} clean")
        fig.colorbar(im0, ax=axs[i,0])

        im1 = axs[i,1].imshow(corrupted, aspect='auto', cmap='viridis')
        axs[i,1].set_title(f"{mod_name} corrupted")
        fig.colorbar(im1, ax=axs[i,1])

        im2 = axs[i,2].imshow(diff, aspect='auto', cmap='coolwarm')
        axs[i,2].set_title(f"{mod_name} diff")
        fig.colorbar(im2, ax=axs[i,2])

    plt.tight_layout()
    plt.savefig("demo.png")
    print("saved to 'demo.png'")

modules_to_plot = ['q_proj', 'k_proj', 'v_proj', 'attn_out', 'post_attn', 'ffn', 'block_output']
plot_module_outputs(clean_intermediates, corrupted_intermediates, modules_to_plot)
