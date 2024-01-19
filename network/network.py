import torch
from torch import nn, einsum
from torch.nn import functional as F
from network.SPT import ShiftedPatchTokenization
from network.net_utils import DropPath
from network.vision_transformer import ViT as vision_trans

import einops as ein
import einops.layers.torch as ein_layer

import numpy as np

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), ** kwargs)


 
class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout = 0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )            
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., is_LSA=False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias = False)
        init_weights(self.to_qkv)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
            
        if is_LSA:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))    
            self.mask = torch.eye(self.num_patches+1, self.num_patches+1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: ein.rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v) 
            
        out = ein.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches+1)
        else:
            flops += (self.dim+2) * self.inner_dim * 3 * self.num_patches  
            flops += self.dim * self.inner_dim * 3  


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0., is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, is_LSA=is_LSA)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout = dropout))
            ]))            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    
    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):       
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x            
            self.scale[str(i)] = attn.fn.scale
        return x



class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio=2, dim_head=16, channels=3, 
                 dropout=0., emb_dropout=0., stochastic_depth=0.1, is_LSA=True, is_SPT=True, mlp_head='original'):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes
       
        if not is_SPT:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )
            
        else:
            self.to_patch_embedding = ShiftedPatchTokenization(3, self.dim, patch_size, is_pe=True)
         
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))
            
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout, 
                                       stochastic_depth, is_LSA=is_LSA)

        if mlp_head == 'original':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )
        elif mlp_head == 'strategy_1':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 200),
                nn.Linear(200, 10)
            )
        elif mlp_head == 'strategy_2':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 180),
                nn.Linear(180, 60),
                nn.Linear(60, 10)
            )
        elif mlp_head == 'strategy_3':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 768),
                nn.Linear(768, 10)
            )
        else:
            self.mlp_head = mlp_head
        
        self.apply(init_weights)
        
        

    def forward(self, img):
        # patch embedding
        
        x = self.to_patch_embedding(img)
            
        b, n, _ = x.shape
        
        cls_tokens = ein.repeat(self.cls_token, '() n d -> b n d', b = b)
      
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)      
        
        return self.mlp_head(x[:, 0])

        
        
class DNM(nn.Module):
    def __init__(self, in_channel, out_channel, num_branch=5, synapse_activation=nn.Sigmoid, dendritic_activation=nn.Sigmoid, soma=nn.Softmax):
        super(DNM, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.nb = num_branch

        
        # synapse init
        self.sn0 = nn.LayerNorm(in_channel)
        self.sn3 = nn.LayerNorm(in_channel)
        self.sa = synapse_activation
        self.sw = nn.Parameter(torch.randn(out_channel, num_branch, in_channel))
        self.sb = nn.Parameter(torch.randn(out_channel, num_branch, in_channel))
        
        
        # dendritic init
        self.dn = nn.LayerNorm([num_branch, in_channel])
        self.da = dendritic_activation
        self.dl = nn.Linear(num_branch, 1)
        
        # soma init (for classification, softmax is used)
        self.soma = soma        
        
    def forward(self, x):
        # input shape (b, in_channel), output shape (b, out_channel)
        x = self.sn0(x)    # 0. norm
        
        if len(x.shape) == 2:
            b, _ = x.shape
            x = ein.repeat(x, 'b d -> b o m d', o=self.out_channel, m=self.nb)
        if len(x.shape) == 3:
            b, _, _ = x.shape
            x = ein.repeat(x, 'b m d -> b o m d', o=self.out_channel)
        
        # synapse (norm -> wx + b (element-wise) -> norm -> activation)
            
        
        
        sw = ein.repeat(self.sw, 'o m d -> b o m d', b=b)
        sb = ein.repeat(self.sb, 'o m d -> b o m d', b=b)  
        #                   # 1. repeat input data and weight
        
        x = sw * x + sb     # 2. wx + b
        
        # x = self.sn3(x)     # 3. norm 
        
        if self.sa is not None:
            x = self.sa(x)  # 4. activation
        
        
        # dendritic (norm -> each branch sum -> activation)
        x = self.dn(x)      # 0. norm for each dnm cell
        
        x = x.sum(dim=3)    # 1. each branch sum (b o m d -> b o m)
        
        if self.da is not None:
            x = self.da(x)  # 2. activation 
        
        
        # membrane (each dnm cell sum to final result)
        # x = x.sum(dim=2)    # 0. each dnm cell sum (b o m -> b o)
        x = self.dl(x).squeeze(2)

        
        # soma 
        if self.soma is not None:
            x = self.soma(x)
        
        return x
        
class DVT(nn.Module):
    def __init__(self, config, device='cpu'):
        super(DVT, self).__init__()
        
        vit_config = config['ViT']
        dnm_config = self.str2func(config['dnm'])
        
        if config['mlp_head'] == 'original':
            self.mlp_head = 'original'
        elif config['mlp_head'] == 'strategy_1':
            self.mlp_head = 'strategy_1'
        elif config['mlp_head'] == 'strategy_2':
            self.mlp_head = 'strategy_2'
        elif config['mlp_head'] == 'strategy_3':
            self.mlp_head = 'strategy_3'
        elif config['mlp_head'] == 'dnm':
            self.mlp_head = DNM(**dnm_config).to(device)

            
        self.net = ViT(**vit_config, mlp_head=self.mlp_head).to(device)
    
    def str2func(self, config):
        # sigmoid, relu, gelu, softmax
        
        for k, func_str in config.items():
            if not isinstance(func_str, str):
                continue
            if func_str.casefold() == 'sigmoid'.casefold():
                func_str = nn.Sigmoid()
            elif func_str.casefold() == 'relu'.casefold():
                func_str = nn.ReLU()
            elif func_str.casefold() == 'gelu'.casefold():
                func_str = nn.GELU()
            elif func_str.casefold() == 'softmax'.casefold():
                func_str = nn.Softmax(dim=1)
            elif func_str.casefold() == 'none'.casefold():
                func_str = None
            config[k] = func_str
        return config
        
    def forward(self, x):
        return self.net(x)
        
class OriViT(nn.Module):
    def __init__(self, config, device='cpu'):
        super(OriViT, self).__init__()
        
        vit_config = config['ViT']
        
        
        if config['mlp_head'] == 'original':
            self.mlp_head = 'original'
        elif config['mlp_head'] == 'strategy_1':
            self.mlp_head = 'strategy_1'
        elif config['mlp_head'] == 'strategy_2':
            self.mlp_head = 'strategy_2'
        elif config['mlp_head'] == 'strategy_3':
            self.mlp_head = 'strategy_3'

            
        self.net = vision_trans(**vit_config, mlp_head=self.mlp_head).to(device)
    
    def str2func(self, config):
        # sigmoid, relu, gelu, softmax
        
        for k, func_str in config.items():
            if not isinstance(func_str, str):
                continue
            if func_str.casefold() == 'sigmoid'.casefold():
                func_str = nn.Sigmoid()
            elif func_str.casefold() == 'relu'.casefold():
                func_str = nn.ReLU()
            elif func_str.casefold() == 'gelu'.casefold():
                func_str = nn.GELU()
            elif func_str.casefold() == 'softmax'.casefold():
                func_str = nn.Softmax(dim=1)
            elif func_str.casefold() == 'none'.casefold():
                func_str = None
            config[k] = func_str
        return config
        
    def forward(self, x):
        return self.net(x)
        
        

        

'''     
config = {
    "ViT": {
        "img_size": 32,
        "patch_size": 4,
        "num_classes": 10,
        "dim": 192,
        "depth": 9,
        "heads": 12,
        "mlp_dim_ratio": 2,
        "dim_head": 16
    },
    "dnm": {
        "in_channel": 192, 
        "out_channel": 10, 
        "num_branch": 10, 
        "synapse_activation": "Softmax", 
        "dendritic_activation": "None",
        "soma": "None"
    },
    "mlp_head": "dnm"
}

net = TransDNM(config, 'cuda:0')

img = torch.randn(10, 3, 32, 32).to('cuda:0')
print(net(img).shape)
'''   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
