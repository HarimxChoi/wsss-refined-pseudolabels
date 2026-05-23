import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import class_names, new_class_names_coco, BACKGROUND_CATEGORY_COCO
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_coco_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP_Plus.PAR import PAR


def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label.squeeze(0)


class WeCLIP_Plus(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, dino_model=None,
                 dino_fts_dim=768, decoder_layers=3, embedding_dim=256,
                 in_channels=512, dataset_root_path=None, clip_flag=16, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.dino_fts_fuse_dim = dino_fts_dim
        self.clip_flag = clip_flag

        # ===== CLIP encoder =====
        self.encoder, _ = clip.load(clip_model, device=device)
        for name, param in self.encoder.named_parameters():
            if clip_flag == 14 and '23' not in name:
                param.requires_grad=False
            if clip_flag == 16 and "11" not in name:
                param.requires_grad=False
        self.encoder.eval()

        # ===== DINO encoder (frozen) =====
        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', dino_model)
        for _, p in self.dino_encoder.named_parameters():
            p.requires_grad = False

        self.in_channels = in_channels

        # ===== heads / decoders =====
        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,
                                              embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=1)
        self.dino_decoder_fts_fuse = SegFormerHead(
            in_channels=[self.dino_fts_fuse_dim]*4, embedding_dim=self.embedding_dim,
            num_classes=self.num_classes, index=1
        )
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=decoder_layers,
                                          heads=8, output_dim=self.num_classes)

        # ===== text features =====
        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_COCO,
                                                    ['a clean origami {}.'], self.encoder)  # [N_bg, C_txt]
        self.fg_text_features = zeroshot_classifier(new_class_names_coco,
                                                    ['a clean origami {}.'], self.encoder)  # [N_fg, C_txt]
        self.num_bg = self.bg_text_features.shape[0]
        self.bg_txt_dim = self.bg_text_features.shape[1]  # 512 (ViT-B/16) or 768 (ViT-L/14)


        # ===== CAM / PAR =====
        self.root_path = os.path.join(dataset_root_path, 'SegmentationClass')
        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers,
                                reshape_transform=reshape_transform, clip_flag=clip_flag)
        self.cam_bg_thres = 1.0
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()

        # ===== runtime =====
        self.iter_num = 0
        self.require_all_fts = True

        # ===== teacher prob for CAM merge =====
        self._teacher_seg_prob = None
        self.high_conf_merge_iter = 25000
        self.teacher_cam_tau = float(os.environ.get("TEACHER_CAM_TAU", "0.75"))

    # ---- external API (train loopfrom call) ----
    @torch.no_grad()
    def set_teacher_seg_prob(self, prob: torch.Tensor | None):

        self._teacher_seg_prob = None if prob is None else prob.detach()

    # ---- param groups ----
    def get_param_groups(self):
        groups = [[], [], [], []]
        for p in list(self.decoder.parameters()):
            groups[3].append(p)
        for p in list(self.decoder_fts_fuse.parameters()):
            groups[3].append(p)
        for p in list(self.dino_decoder_fts_fuse.parameters()):
            groups[3].append(p)
    
        return groups


    def forward(self, img, img_names='2007_000032', mode='train'):
        cam_list = []
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1

        # ===== CLIP features (all layers) =====
        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder,
                                                    require_all_fts=True, clip_flag=self.clip_flag)

        # ===== DINO features (frozen) =====
        with torch.no_grad():
            dino_img_h, dino_img_w = (h//14)*14, (w//14)*14
            dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w),
                                    mode='bilinear', align_corners=False)
            dino_fts_raw = self.dino_encoder.forward_features(dino_img)['x_norm_patchtokens']

        fts_all_stack = torch.stack(fts_all, dim=0)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        Ctok = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, Ctok, h//self.clip_flag, w//self.clip_flag)
        all_img_tokens = all_img_tokens[-1].unsqueeze(0)

        fts = self.decoder_fts_fuse(all_img_tokens)
        _, _, fts_h, fts_w = fts.shape

        if isinstance(dino_fts_raw, list):
            for i, dino_fts_single in enumerate(dino_fts_raw):
                dino_fts_raw[i] = dino_fts_single.reshape([b, dino_img_h // 14, dino_img_w // 14, -1]).permute(0, 3, 1, 2)
            dino_fts_raw = torch.stack(dino_fts_raw)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts_raw)
        else:
            dino_fts = dino_fts_raw.reshape([b, dino_img_h//14, dino_img_w//14, -1]).permute(0,3,1,2)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))
        dino_fts = F.interpolate(dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)

        seg_clip, seg_attn_weight_list_clip = self.decoder(fts)
        seg_dino, seg_attn_weight_list_dino = self.decoder(dino_fts)

        bg_prob_for_cam = None  
        
        # ===== CAM seed create =====
        seg_dino_prob = F.softmax(0.5 * seg_dino + 0.5 * seg_clip, dim=1).detach()
        
        if (self._teacher_seg_prob is not None) and (self.iter_num >= self.high_conf_merge_iter):
            tprob = self._teacher_seg_prob
            if tprob.shape[2:] != seg_dino_prob.shape[2:]:
                tprob = F.interpolate(tprob, size=seg_dino_prob.shape[2:], 
                                    mode='bilinear', align_corners=False)
            tconf, _ = tprob.max(dim=1, keepdim=True)
            seg_dino_prob = torch.where((tconf >= self.teacher_cam_tau), tprob, seg_dino_prob)
        
        # Affinity
        clip_dino_fts = torch.cat([fts, dino_fts], dim=1)
        attn_fts = F.interpolate(clip_dino_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
        attn_pred = torch.sigmoid(
            attn_fts.reshape(b, -1, fts_h*fts_w).transpose(2,1).bmm(
                attn_fts.reshape(b, -1, fts_h*fts_w)
            )
        )
        
        
        # ===== CAM create =====
        if self.training:
            for i, img_name in enumerate(img_names):
                img_path = os.path.join(self.root_path, 'train', str(img_name) + '.png')
                cam_fts = fts_all_stack[-1].unsqueeze(0).permute(2,1,0,3)[i]
                cam_attn = attn_weight_stack[i]
                seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]
                
                cam_refined_list, keys, w0, h0 = perform_single_coco_cam(
                    img_path, img[i], cam_fts, cam_attn, seg_attn,
                    self.bg_text_features, self.fg_text_features, self.grad_cam,
                    mode=mode, require_seg_trans=True,
                    seg_dino_cam=seg_dino_prob[i], clip_flag=self.clip_flag
                )
                
                cam_dict = generate_cam_label(cam_refined_list, keys, w0, h0)
                cams = cam_dict['refined_cam'].cuda()
                
                bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], 
                    self.cam_bg_thres).cuda()
                cams = torch.cat([bg_score, cams], dim=0).cuda()
                
                valid_key = torch.from_numpy(
                    np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
                ).cuda()
                
                with torch.no_grad():
                    cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
                
                cam_list.append(cam_labels)
            
            all_cam_labels = torch.stack(cam_list, dim=0)
            return seg_clip, seg_dino, all_cam_labels, attn_pred
        else:
            return seg_clip, seg_dino
if __name__=="__main__":
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    wetr = WeCLIP_Plus('mit_b1', num_classes=20, embedding_dim=256, pretrained=True)
    wetr._param_groups()
    dummy_input = torch.rand(2,3,512,512)
    wetr(dummy_input)