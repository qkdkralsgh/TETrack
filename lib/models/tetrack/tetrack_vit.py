from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from timm.models.layers import DropPath, Mlp
from lib.utils.box_ops import box_xyxy_to_cxcywh

from lib.utils.misc import is_main_process
from .head import build_box_head
from .utils import to_2tuple
from lib.models.tetrack.pos_utils import get_2d_sincos_pos_embed

import numpy as np


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None
            
        
    def forward(self, x, t_h, t_w, policy=None):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape  # [B, 864, 1024] : Large model
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # qkv(x) : [B, 864, 3072], qkv : [3(qkv), B, 16(head), 864, 64]
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)  # 각각 [B, 16(head), 864, 64]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [q_s : [B, 12(head), 324, 64],  k : [B, 12, 388, 64]]
        
        if policy is None:
            attn = attn.softmax(dim=-1)  # Base : [B, 12, 324, 388]
        else:
            policy = torch.cat((policy[0], policy[1]), dim=1)
            attn = attn.softmax(dim=-1)
            # attn = attn*policy[1].unsqueeze(1)
            keep_attn = attn*policy.unsqueeze(1)
            attn = (attn + keep_attn)/2
        
        attn_map = attn.clone()
    
        attn = self.attn_drop(attn_map.clone())
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_map[:, :, t_h*t_w*2:, t_h*t_w*2:]
        

class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    

    ### Original ###
    def forward(self, x, t_h, t_w, policy=None):
        attn_output, attn_map = self.attn(self.norm1(x), t_h, t_w, policy=policy)
        x = x + self.drop_path1(attn_output)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, attn_map


class MultiheadPredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, num_heads=6, embed_dim=384):
        super().__init__()

        self.num_heads=num_heads
        self.embed_dim = embed_dim
        onehead_in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim // num_heads),
            nn.Linear(embed_dim // num_heads, embed_dim // num_heads),
            nn.GELU()
        )

        onehead_out_conv = nn.Sequential(
            nn.Linear(embed_dim // num_heads, embed_dim // num_heads  // 2),
            nn.GELU(),
            nn.Linear(embed_dim // num_heads // 2, embed_dim // num_heads // 4),
            nn.GELU(),
            nn.Linear(embed_dim // num_heads // 4, 2),
            #nn.LogSoftmax(dim=-1)
        )

        in_conv_list = [onehead_in_conv for _ in range(num_heads)]
        out_conv_list = [onehead_out_conv for _ in range(num_heads)]

        self.in_conv = nn.ModuleList(in_conv_list)
        self.out_conv = nn.ModuleList(out_conv_list)

    def forward(self, x, policy):  # policy = previous decision

        multihead_score = 0
        multihead_softmax_score = 0
        for i in range(self.num_heads):
            x_single = x[:,:,self.embed_dim//self.num_heads*i:self.embed_dim//self.num_heads*(i+1)]   #([96, 196, 64])
            x_single = self.in_conv[i](x_single)
            B, N, C = x_single.size()       #([96, 196, 64])
            local_x = x_single[:,:, :C//2]  #([96, 196, 32])
            global_x = (x_single[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)  #([96, 1, 32])
            x_single = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)  #([96, 196, 64])
            x_single = self.out_conv[i](x_single) #([96, 196, 2])

            # for placeholder
            m = nn.Softmax(dim=-1)
            score_softmax = m(x_single)
            multihead_softmax_score += score_softmax

            # for gumble
            n = nn.LogSoftmax(dim=-1)
            score_single = n(x_single)
            multihead_score += score_single

        # for gumble
        multihead_score = multihead_score / self.num_heads  # ([96, 196, 2])

        # for placeholder
        multihead_softmax_score = multihead_softmax_score / self.num_heads  # get softmax keep/drop probability
        
        # token_scores = multihead_softmax_score[:, :, 0]/torch.max(multihead_softmax_score[:, :, 0])

        return multihead_score, multihead_softmax_score #, token_scores  #, represent_token, placeholder_weights, placeholder_score3


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size_s=256, img_size_t=128, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, 
                 pruning_loc=None):
        super(VisionTransformer, self).__init__(img_size=224, patch_size=patch_size, in_chans=in_chans,
                                                num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, weight_init=weight_init,
                                                norm_layer=norm_layer, act_layer=act_layer)

        self.patch_embed = embed_layer(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)])
        
        predictor_list = [MultiheadPredictorLG(num_heads,embed_dim) for _ in range(len(pruning_loc))]
        self.score_predictor = nn.ModuleList(predictor_list)

        self.pruning_loc = pruning_loc
        
        self.grid_size_s = img_size_s // patch_size
        self.grid_size_t = img_size_t // patch_size
        self.num_patches_s = self.grid_size_s ** 2
        self.num_patches_t = self.grid_size_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim), requires_grad=False)

        self.init_pos_embed()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_t ** .5),
                                            cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_s ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, x_t, x_ot, x_s):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        x_t = self.patch_embed(x_t)  # BCHW-->BNC
        x_ot = self.patch_embed(x_ot)
        x_s = self.patch_embed(x_s)
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.grid_size_s
        H_t = W_t = self.grid_size_t

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x = torch.cat([x_t, x_ot, x_s], dim=1)
        x = self.pos_drop(x)
        
        p_count = 0
        out_pred_prob = []
        init_n_t = H_t*W_t
        init_n_s = H_s*W_s
        sparse = []
        score_dict = {}
        prev_decision_ot = torch.ones(B, init_n_t*2, 1, dtype=x.dtype, device=x.device)
        prev_decision_s = torch.ones(B, init_n_s, 1, dtype=x.dtype, device=x.device)
        update_threshold = 0
        policy = None
        attn_list = []
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                if not self.training and i == 7:  ### Test 할때만 가능하게 해야함
                   update_threshold = torch.sum(prev_decision_ot[:, H_t*W_t:, :], dim=1).item()
                   print(update_threshold)
                ### token select for target template ###            
                ot_pred_score, ot_softmax_score = self.score_predictor[p_count](x[:, :init_n_t*2, :], prev_decision_ot)
                ot_pred_score = ot_pred_score.reshape(B, -1, 2)
                ot_softmax_score = ot_softmax_score.reshape(B, -1, 2)
                
                ot_hard_keep_decision = F.gumbel_softmax(ot_pred_score, hard=False)[:, :, 0:1] * prev_decision_ot
                
                ### token select for search region ###
                s_pred_score, s_softmax_score = self.score_predictor[p_count](x[:, init_n_t*2:, :], prev_decision_s)
                s_pred_score = s_pred_score.reshape(B, -1, 2)
                s_softmax_score = s_softmax_score.reshape(B, -1, 2)
                
                s_hard_keep_decision = F.gumbel_softmax(s_pred_score, hard=False)[:, :, 0:1] * prev_decision_s
                
                if self.training:
                    out_pred_prob.append(s_hard_keep_decision.reshape(B, init_n_s))
                    policy = [ot_hard_keep_decision, s_hard_keep_decision]
                    x, _ = blk(x, H_t, W_t, policy=policy)
                    prev_decision_ot = ot_hard_keep_decision
                    prev_decision_s = s_hard_keep_decision
                else:
                    out_pred_prob.append(s_hard_keep_decision.reshape(B, init_n_s))
                    policy = [ot_hard_keep_decision, s_hard_keep_decision]
                    x, attn_map = blk(x, H_t, W_t, policy=policy)
                    prev_decision_ot = ot_hard_keep_decision
                    prev_decision_s = s_hard_keep_decision
                    zeros, unzeros = test_irregular_sparsity(p_count, s_hard_keep_decision)
                    sparse.append([zeros, unzeros])
                    score = s_pred_score[:, :, 0:1].cpu().numpy().tolist()
                    score_dict[p_count] = score[0]
                p_count += 1
                        
            else:
                x, attn_map = blk(x, H_t, W_t, policy=policy)
                
            attn_list.append(attn_map)
            
        x_t, x_ot, x_s = torch.split(x, [H_t*W_t, H_t*W_t, H_s*W_s], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        # update_threshold = torch.sum(prev_decision_ot[:, H_t*W_t:, :], dim=1).item()
        # print(update_threshold)
        return x_t_2d, x_ot_2d, x_s_2d, attn_list, out_pred_prob, update_threshold  

    
def get_tetrack_vit(config, train):
    img_size_s = config.DATA.SEARCH.SIZE
    img_size_t = config.DATA.TEMPLATE.SIZE
    
    PRUNING_LOC = [3,7,11]  # if L-model, [3, 7, 11, 15, 19, 23]
    if config.MODEL.VIT_TYPE == 'large_patch16':
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1
            , pruning_loc=PRUNING_LOC)
    elif config.MODEL.VIT_TYPE == 'base_patch16':
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1
            , pruning_loc=PRUNING_LOC)
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'large_patch16' or 'base_patch16'")

    if config.MODEL.BACKBONE.PRETRAINED and train:
        ckpt_path = config.MODEL.BACKBONE.PRETRAINED_PATH
        ckpt = torch.load(ckpt_path, map_location='cpu')['model']
        new_dict = {}
        for k, v in ckpt.items():
            if 'pos_embed' not in k and 'mask_token' not in k:    # use fixed pos embed
                new_dict[k] = v
        missing_keys, unexpected_keys = vit.load_state_dict(new_dict, strict=False)
        if is_main_process():
            print("Load pretrained backbone checkpoint from:", ckpt_path)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained ViT done.")

    return vit


class TETrack(nn.Module):
    def __init__(self, backbone, box_head, head_type="CORNER"):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.head_type = head_type
        

    def forward(self, template, online_template, search):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5: 
            online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, online_template, search, attn_map, out_pred_score, template_decision = self.backbone(template, online_template, search)
        # search shape: (b, 384, 20, 20)
        # Forward the corner head
        out_dict, outputs_coord_new = self.forward_box_head(search)
        out_dict["token_scores"] = out_pred_score
        
        return out_dict, outputs_coord_new, attn_map, template_decision
    
    
    def forward_box_head(self, search):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "CORNER" in self.head_type:
            # run the corner head
            b = search.size(0)
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(search))
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out_dict = {'pred_boxes': outputs_coord_new}
            return out_dict, outputs_coord_new
        else:
            raise KeyError


def build_tetrack_vit(cfg, train=True) -> TETrack:
    backbone = get_tetrack_vit(cfg, train)
    box_head = build_box_head(cfg)  # a simple corner head
    model = TETrack(
        backbone,
        box_head,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    return model


def test_irregular_sparsity(name,matrix):

    # continue
    zeros = np.sum(matrix.cpu().detach().numpy() == 0)

    non_zeros = np.sum(matrix.cpu().detach().numpy() != 0)

    # print(name, non_zeros)
    #print(" {}, all weights: {}, irregular zeros: {}, irregular sparsity is: {:.4f}".format( name, zeros+non_zeros, zeros, zeros / (zeros + non_zeros)))
    # print(non_zeros+zeros)
    # total_nonzeros += 128000

    return zeros,non_zeros
