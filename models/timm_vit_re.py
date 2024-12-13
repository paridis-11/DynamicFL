import math
import torch
import torch.nn as nn
from functools import partial

import torch
import torch.nn as nn


# Helper functions
def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    with torch.no_grad():
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # uniform sample, and then scale to truncation limits
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # inverse cdf for gaussian
        tensor.erfinv_()


        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
    
    return tensor

class DropPath(nn.Module):
    """ Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


# Vision Transformer (ViT) Implementation
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, num_patches, embed_dim)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)


        self.branches_fc1 = None
        self.branches_fc2 = None
        self.num_branches = 0 

    def forward(self, x):
        if self.branches_fc1 is None and self.branches_fc2 is None:

            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            return self.drop(x)


        out = self.fc1(x)


        if self.branches_fc1 is not None:
            for branch in self.branches_fc1:
                linear, bn = branch
                branch_out = linear(x)
                branch_out = branch_out.permute(0, 2, 1) 
                branch_out = bn(branch_out)
                branch_out = branch_out.permute(0, 2, 1)  
                out += branch_out

        out = self.act(out)
        out = self.drop(out)

        out2 = self.fc2(out)


        if self.branches_fc2 is not None:
            for branch in self.branches_fc2:
                linear, bn = branch
                branch_out = linear(out)
                branch_out = branch_out.permute(0, 2, 1)
                branch_out = bn(branch_out)
                branch_out = branch_out.permute(0, 2, 1)
                out2 += branch_out

        return self.drop(out2)

    def expand(self, num_branches, alpha=0.99):

        if num_branches <= 0:
            return

        device = self.fc1.weight.device

        assert 0.0 <= alpha <= 1.0, "alpha 必须在 [0, 1] 之间"


        self.fc1.weight.data *= alpha
        if self.fc1.bias is not None:
            self.fc1.bias.data *= alpha

        self.fc2.weight.data *= alpha
        if self.fc2.bias is not None:
            self.fc2.bias.data *= alpha


        remaining_fraction = (1.0 - alpha) / num_branches

        self.branches_fc1 = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.fc1.in_features, self.fc1.out_features), nn.BatchNorm1d(self.fc1.out_features)])
            for _ in range(num_branches)
        ]).to(device)
        

        for i, branch in enumerate(self.branches_fc1):
            linear, bn = branch
            if i < num_branches - 1:  
                linear.weight.data.copy_(self.fc1.weight.data * remaining_fraction / alpha)
                if self.fc1.bias is not None:
                    linear.bias.data.copy_(self.fc1.bias.data * remaining_fraction / alpha)
                else:
                    linear.bias.data.zero_()
            else:  
                linear.weight.data.copy_(self.fc1.weight.data / alpha * (1-alpha) - sum([b[0].weight.data for b in self.branches_fc1[:-1]]))
                if self.fc1.bias is not None:
                    linear.bias.data.copy_(self.fc1.bias.data / alpha * (1-alpha) - sum([b[0].bias.data for b in self.branches_fc1[:-1]]))
                else:
                    linear.bias.data.zero_()


        self.branches_fc2 = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.fc2.in_features, self.fc2.out_features), nn.BatchNorm1d(self.fc2.out_features)])
            for _ in range(num_branches)
        ]).to(device)


        for i, branch in enumerate(self.branches_fc2):
            linear, bn = branch
            if i < num_branches - 1: 
                linear.weight.data.copy_(self.fc2.weight.data * remaining_fraction / alpha)
                if self.fc2.bias is not None:
                    linear.bias.data.copy_(self.fc2.bias.data * remaining_fraction / alpha)
                else:
                    linear.bias.data.zero_()
            else:  
                linear.weight.data.copy_(self.fc2.weight.data / alpha * (1-alpha) - sum([b[0].weight.data for b in self.branches_fc2[:-1]]))
                if self.fc2.bias is not None:
                    linear.bias.data.copy_(self.fc2.bias.data / alpha * (1-alpha) - sum([b[0].bias.data for b in self.branches_fc2[:-1]]))
                else:
                    linear.bias.data.zero_()

        self.num_branches += num_branches


    def merge(self):

        if self.branches_fc1 is not None:
            with torch.no_grad():
                for branch in self.branches_fc1:
                    linear, bn = branch
                    weight, bias = self._fuse_bn(linear.weight, linear.bias, bn)
                    self.fc1.weight += weight
                    if self.fc1.bias is not None:
                        self.fc1.bias += bias

        if self.branches_fc2 is not None:
            with torch.no_grad():
                for branch in self.branches_fc2:
                    linear, bn = branch
                    weight, bias = self._fuse_bn(linear.weight, linear.bias, bn)
                    self.fc2.weight += weight
                    if self.fc2.bias is not None:
                        self.fc2.bias += bias


        self.branches_fc1 = None
        self.branches_fc2 = None

    def _fuse_bn(self, weight, bias, bn):

        bn_mean = bn.running_mean
        bn_var_sqrt = torch.sqrt(bn.running_var + bn.eps)  # std
        bn_weight = bn.weight                              # gamma
        bn_bias = bn.bias                                  # beta

        fused_weight = weight * (bn_weight / bn_var_sqrt).reshape([-1, 1])
        fused_bias = (bias - bn_mean) / bn_var_sqrt * bn_weight + bn_bias

        return fused_weight, fused_bias

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x

    def expand(self, num_branches):

        self.mlp.expand(num_branches)

    def merge(self):

        self.mlp.merge()

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])

    def expand(self, num_branches):

        for blk in self.blocks:
            blk.expand(num_branches)

    def merge(self):

        for blk in self.blocks:
            blk.merge()
    def get_num_params(self):

        return sum(p.numel() for p in self.parameters() if p.requires_grad)





def test_expand_and_merge():
    model = VisionTransformer(img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12).train()

    input_tensor = torch.randn(20, 3, 224, 224) 

    _ = model(input_tensor)
    _ = model(input_tensor)
    _ = model(input_tensor)

    model.eval()


    output_before_expand = model(input_tensor)


    model.expand(4)


    model.eval()
    output_after_expand = model(input_tensor)


    model.merge()


    model.eval()
    output_after_merge = model(input_tensor)
    print('The difference: ', (output_after_expand-output_before_expand).abs().sum())

    print('The difference: ', (output_after_merge-output_after_expand).abs().sum())

    print('The difference: ', (output_after_merge-output_before_expand).abs().sum())

    # assert torch.allclose(output_before_expand, output_after_expand, atol=1e-5), 
    # assert torch.allclose(output_after_expand, output_after_merge, atol=1e-5), 





def test_mlp_expand_and_merge():


    mlp = Mlp(in_features=128, hidden_features=256, out_features=128).train()
    input = torch.randn(4, 196, 128)

    _ = mlp(input)
    _ = mlp(input)
    _ = mlp(input)

    mlp.eval() 


    output_before_expand = mlp(input)


    mlp.expand(2)


    mlp.eval()
    output_after_expand = mlp(input)


    mlp.merge()


    mlp.eval()
    output_after_merge = mlp(input)

    print('The difference: ', (output_after_expand-output_before_expand).abs().sum())

    print('The difference: ', (output_after_merge-output_after_expand).abs().sum())

    print('The difference: ', (output_after_merge-output_before_expand).abs().sum())





    model = Mlp(in_features=192, hidden_features=384, out_features=192)
    input = torch.randn(4, 196, 192)
    
    # update the mean and std of batch normalizations
    _ = model(input)
    _ = model(input)
    _ = model(input)

    # test the model before merging and after merging
    model.eval()
    output1 = model(input).clone()
    model.merge()
    output2 = model(input)
    print('The difference: ', (output1-output2).abs().sum())







    


if __name__ == "__main__":

    model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=9,
            embed_dim=1280,
            depth=64,
            num_heads=16,
            mlp_ratio=4,
            drop_rate=0.1,
            attn_drop_rate=0.1,
        )

    input_tensor = torch.randn(1, 3, 224, 224)  # 模拟输入图像




    model.expand(num_branches = 7)
    
    model.merge()
