import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.sammodel import ImageEncoderViT, MaskDecoder, TwoWayTransformer

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple
from mmcv.cnn import ConvModule
from torch import einsum

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)
    return iou.mean()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.

    removed forward_with_coords which is 这个方法可以用于在对图像做处理时，对非归一化的点坐标进行位置编码，以便在后续的模型中使用
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

class BIMLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=64, norm_cfg=None):
        super(BIMLAHead, self).__init__()
        self.head2_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                     nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head3_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                     nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head4_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                     nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head5_1 = nn.Sequential(nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
                                     nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False), nn.BatchNorm2d(mlahead_channels), nn.ReLU())
        self.head6_1 = nn.Sequential(
            nn.ConvTranspose2d(mla_channels, mlahead_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mlahead_channels), nn.ReLU(),
            nn.ConvTranspose2d(mlahead_channels, mlahead_channels, 16, stride=8, padding=4, bias=False),
            nn.BatchNorm2d(mlahead_channels), nn.ReLU())

    def forward(self, mla_b2, mla_b3, mla_b4, mla_b6):
        head2_1 = self.head2_1(mla_b2)
        head3_1 = self.head3_1(mla_b3)
        head4_1 = self.head4_1(mla_b4)
        head6_1 = self.head6_1(mla_b6)
        return torch.cat([head2_1, head3_1, head4_1, head6_1], dim=1)

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super(ChannelAtt, self).__init__()
        self.conv_1x1 = ConvModule(out_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        """Forward function."""
        atten = torch.mean(x, dim=(2, 3), keepdim=True)
        print(atten.shape)
        atten = self.conv_1x1(atten)
        return  x,atten

class AFD(nn.Module):
    "Active fusion decoder"
    def __init__(self, s_channels,c_channels, conv_cfg, norm_cfg, act_cfg, h=8):
        super(AFD, self).__init__()
        self.s_channels = s_channels
        self.c_channels = c_channels
        self.h = h
        self.scale = h ** - 0.5
        self.spatial_att = ChannelAtt(s_channels, s_channels, conv_cfg, norm_cfg, act_cfg)
        self.context_att = ChannelAtt(c_channels, c_channels, conv_cfg, norm_cfg, act_cfg)
        self.qkv = nn.Linear(s_channels + c_channels,(s_channels + c_channels) * 3,bias = False)
        self.proj = nn.Linear(s_channels + c_channels, s_channels + c_channels)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, sp_feat, co_feat):
        # **_att: B x C x 1 x 1
        print(sp_feat.shape,co_feat.shape)
        s_feat, s_att = self.spatial_att(sp_feat)
        c_feat, c_att = self.context_att(co_feat)
        b = s_att.shape[0] # h = 1, w = 1
        sc_att = torch.cat([s_att,c_att],1).view(b,-1)
        qkv = self.qkv(sc_att).reshape(b,1,3,self.h, (self.c_channels + self.s_channels) // self.h).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2] # [B,h,1,2C // h]
        k_softmax = k.softmax(dim = 1) # channel-wise softmax operation
        k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
        fuse_weight = self.scale * einsum("b h n k, b h k v -> b h n v", q,
                            k_softmax_T_dot_v)
        fuse_weight = fuse_weight.transpose(1,2).reshape(b,-1)
        fuse_weight = self.proj(fuse_weight)
        fuse_weight = self.proj_drop(fuse_weight)
        fuse_weight = fuse_weight.reshape(b,-1,1,1) # [B,C,1,1]
        fuse_s,fuse_c = fuse_weight[:,:self.s_channels],fuse_weight[:,-self.c_channels:]
        out = (1 + fuse_s) * s_feat + (1 + fuse_c) * c_feat
        return s_feat, c_feat, out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(x.shape)
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class MaxPool(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


nonlinearity = partial(F.relu, inplace=True)
class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class Conv11(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv11(x)

class Up1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SingleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.shape)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        # x = self.up(x)
        # print(x.shape)
        return self.conv(x)

class Upsample(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 1"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # print(x.shape)
        return self.single_conv(x)

@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None, mla_channels=256, mlahead_channels=128):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.mla_channels = mla_channels
        self.mlahead_channels = mlahead_channels
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False

        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

        self.norm_cfg = dict(type='BN', requires_grad=True)
        self.mlahead = BIMLAHead(mla_channels=self.mla_channels, mlahead_channels=self.mlahead_channels,
                                 norm_cfg=self.norm_cfg)
        self.global_features = nn.Sequential(
            nn.Conv2d(4 * self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 3, padding=1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU(),
            nn.Conv2d(self.mlahead_channels, self.mlahead_channels, 1),
            nn.BatchNorm2d(self.mlahead_channels), nn.ReLU())
        self.edge = nn.Conv2d(self.mlahead_channels, 1, 1)

        self.inc = (DoubleConv(1, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (MaxPool(256, 256))

        self.dblock = Dblock(256)

        self.encode11 = (Conv11(32, 21))
        self.encode12 = (Conv11(32, 21))
        self.encode21 = (Conv11(64, 21))
        self.encode22 = (Conv11(64, 21))
        self.encode31 = (Conv11(128, 21))
        self.encode32 = (Conv11(128, 21))
        self.encode41 = (Conv11(256, 21))
        self.encode42 = (Conv11(256, 21))
        self.encode51 = (Conv11(256, 21))
        self.encode52 = (Conv11(256, 21))
        self.encode53 = (Conv11(256, 21))

        self.conv11 = (Conv11(21, 1))

        self.deconv2 = (nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.deconv3 = (nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)))
        self.deconv4 = (nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)))
        self.deconv5 = (nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)))

        self.score_final = nn.Conv2d(5, 1, 1)

        self.up1 = (Up1(512, 256))
        self.up2 = (Up(256, 128, bilinear=False))
        self.up3 = (Up(128, 64, bilinear=False))
        self.up4 = (Upsample(64, 32, bilinear=False))
        self.conv1 = (SingleConv(32, 32, mid_channels=None))
        self.outc = (OutConv(32, 1))

    def set_input(self, input, gt_mask, bgt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)
        self.bgt_mask = bgt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def forward(self):
        bs = 1
        self.seg_loss=0
        self.edge_loss=0
        self.fusion_loss=0

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        self.features, self.edgefeatures = self.image_encoder(self.input)

        # Predict edge masks
        edge = self.mlahead(self.edgefeatures[0], self.edgefeatures[1], self.edgefeatures[2], self.edgefeatures[3])
        edge = self.global_features(edge)
        edge = self.edge(edge)
        self.edge_mask = edge

        # Predict semantic masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks

        # Fusion block
        x= masks - edge
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dblock(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv1(x)
        logits = self.outc(x)
        self.fine = logits

    def infer(self, input):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        self.features, self.edgefeatures = self.image_encoder(input)

        # Predict edge masks
        edge = self.mlahead(self.edgefeatures[0], self.edgefeatures[1], self.edgefeatures[2], self.edgefeatures[3])
        edge = self.global_features(edge)
        edge = self.edge(edge)
        self.edge_mask = edge

        # Predict semantic masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks

        #fusion block
        x = masks - edge
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dblock(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv1(x)
        logits = self.outc(x)
        self.fine = logits
        return masks, edge, logits

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask) + self.criterionBCE(self.edge_mask, self.bgt_mask) * 0.8 + self.criterionBCE(self.fine, self.gt_mask)
        if self.loss_mode == 'iou':
            self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)
            self.loss_G += _iou_loss(self.edge_mask, self.bgt_mask) * 0.8
            self.loss_G += _iou_loss(self.fine, self.gt_mask)
        self.seg_loss= self.criterionBCE(self.pred_mask, self.gt_mask)+_iou_loss(self.pred_mask, self.gt_mask)
        self.edge_loss = self.criterionBCE(self.edge_mask, self.bgt_mask) + _iou_loss(self.edge_mask, self.bgt_mask)
        self.fusion_loss = self.criterionBCE(self.fine, self.gt_mask) + _iou_loss(self.fine, self.gt_mask)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
