 #!/usr/bin/env python3
""" Training for F3Set """
import os
import argparse
from contextlib import nullcontext
import random
import numpy as np
import torch
import math
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from torch.utils.data import DataLoader
import torchvision
from itertools import groupby
import timm
from tqdm import tqdm

from model.common import step, BaseRGBModel
from model.shift import make_temporal_shift
from model.slowfast import ResNet3dSlowFast
from model.resnet3d import ResNet3d
from model.x3d import X3D
from model.stgcn import STGCN
from model.msg3d import MSG3D
from model.aagcn import AAGCN
from model.ctrgcn import CTRGCN
from model.vggish import *
from model.modules import *
from dataset.frame_process_fusion import ActionSeqDataset, ActionSeqVideoDataset
from util.eval import edit_score, non_maximum_suppression, non_maximum_suppression_np
from util.io import load_json, store_json, clear_files
from util.dataset import DATASETS, load_classes
from thop import profile
import warnings
warnings.filterwarnings("ignore")

STAGE = 1
EPOCH_NUM_FRAMES = 200000 if STAGE == 2 else 100000
# EPOCH_NUM_FRAMES = 500000 if STAGE == 2 else 100000
BASE_NUM_WORKERS = 4
BASE_NUM_VAL_EPOCHS = 20
INFERENCE_BATCH_SIZE = 4
HIDDEN_DIM = 368

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('frame_dir', type=str, help='Path to extracted frames')

    parser.add_argument(
        '-m', '--feature_arch', type=str, required=True, choices=[
            # rgb-based feature extractor
            'rn50',
            'rn50_tsm',
            'rn50_gsm'
            'rny002',
            'rny002_tsm',
            'rny002_gsm',
            'rny008',
            'rny008_tsm',
            'rny008_gsm',
            'slowfast',
            # skeleton-based feature extractor
            'stgcn',
            'aagcn',
            'ctrgcn',
            'msg3d',
            'stgcn++',
            'rny002_tsm-msg3d',
            'rny002_tsm-stgcn++'
        ], help='architecture for feature extraction')

    parser.add_argument(
        '-t', '--temporal_arch', type=str, default='gru',
        choices=['gru', 'deeper_gru', 'mstcn', 'asformer', 'actionformer', 'gcn', 'tcn', 'fc'])
    parser.add_argument(
        '-ctx', '--use_ctx', action='store_true',
        help='Whether include the contextual module; if not, just use multi-label classifier')
    parser.add_argument(
        '-aud', '--use_audio', action='store_true',
        help='Whether use audio information')

    parser.add_argument('--clip_len', type=int, default=96)
    parser.add_argument('--crop_dim', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--window', type=int, default=5,
                        help='Non maximun suppression window size')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')

    parser.add_argument('--warm_up_epochs', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=50)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-s', '--save_dir', type=str, required=True,
                        help='Dir to save checkpoints and predictions')

    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint in <save_dir>')

    parser.add_argument('--start_val_epoch', type=int, default=20)
    parser.add_argument('--criterion', choices=['edit', 'loss'], default='edit')
    parser.add_argument('--dilate_len', type=int, default=0,
                        help='Label dilation when training')

    parser.add_argument('-j', '--num_workers', type=int,
                        help='Base number of dataloader workers')

    parser.add_argument('-mgpu', '--gpu_parallel', action='store_true')
    return parser.parse_args()


class F3Set(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, step=1, window=5, use_ctx=True,
                     use_audio=False, device='cuda'):
            super().__init__()
            is_rgb = True
            self._device = device
            self._num_classes = num_classes
            self._window = window
            self._use_ctx = use_ctx
            self._use_audio = use_audio

            if 'rn50' in feature_arch:
                resnet_name = feature_arch.split('_')[0].replace('rn', 'resnet')
                rgb_feat = getattr(torchvision.models, resnet_name)(pretrained=True)
                rgb_feat_dim = features.fc.in_features
                rgb_feat.fc = nn.Identity()


            elif feature_arch.startswith(('rny002', 'rny008')):
                rgb_feat = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny008': 'regnety_008',
                }[feature_arch.rsplit('_', 1)[0]], pretrained=True)
                rgb_feat_dim = rgb_feat.head.fc.in_features
                rgb_feat.head.fc = nn.Identity()


            elif 'slowfast' in feature_arch:
                # rgb_feat = ResNet3dSlowFast(None, slow_upsample=8)  # slowfast 4x16
                rgb_feat = ResNet3dSlowFast(None, resample_rate=4, speed_ratio=4, fusion_kernel=7, slow_upsample=4)  # slowfast 8x8
                rgb_feat.load_pretrained_weight()
                rgb_feat_dim = 2304

            # optical flow
            flow_feat = timm.create_model('regnety_002', pretrained=True)
            # print(flow_feat)
            flow_feat_dim = flow_feat.head.fc.in_features
            # flow_feat_dim = flow_feat.fc.in_features
            flow_feat.head.fc = nn.Identity()
            # flow_feat.fc = nn.Identity()
            flow_feat.stem.conv = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # flow_feat.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            # skeleton
            sk_feat = None
            sk_feat_dim = 0
            if 'stgcn' in feature_arch.split('-'):
                sk_feat = STGCN(in_channels=2, data_bn_type='MVC', graph_cfg=dict(layout='coco', mode='stgcn_spatial'))
                sk_feat_dim = 256
                
            elif 'aagcn' in feature_arch.split('-'):
                sk_feat = AAGCN(dict(layout='coco', mode='spatial'), in_channels=2, data_bn_type='MVC')
                sk_feat_dim = 256

            elif 'ctrgcn' in feature_arch.split('-'):
                sk_feat = CTRGCN(dict(layout='coco', mode='spatial'), in_channels=2)
                sk_feat_dim = 256

            elif 'msg3d' in feature_arch.split('-'):
                sk_feat = MSG3D(dict(layout='coco', mode='binary_adj'), in_channels=2)
                sk_feat_dim = 384

            elif 'stgcn++' in feature_arch.split('-'):
                sk_feat = STGCN(in_channels=2, data_bn_type='MVC', gcn_adaptive='init', gcn_with_res=True,
                                tcn_type='mstcn', graph_cfg=dict(layout='coco', mode='spatial'))
                sk_feat_dim = 256


            # Add Temporal Shift Modules
            self._require_clip_len = clip_len
            if '_tsm' in feature_arch:
                make_temporal_shift(rgb_feat, clip_len, is_gsm=False, step=step)
                make_temporal_shift(flow_feat, clip_len, is_gsm=False, step=step)
                self._require_clip_len = clip_len
            elif '_gsm' in feature_arch:
                make_temporal_shift(rgb_feat, clip_len, is_gsm=True)
                make_temporal_shift(flow_feat, clip_len, is_gsm=True)
                self._require_clip_len = clip_len

            self._rgb_feat = rgb_feat  # rgb feature extractor
            self._flow_feat = flow_feat  # optical flow feature extractor
            self._sk_feat = sk_feat  # skeleton feature extractor
            # self._feat_dim = rgb_feat_dim + sk_feat_dim
            self._is_3d = 'slowfast' in feature_arch

            # head modules
            d_model = HIDDEN_DIM #min(HIDDEN_DIM, self._feat_dim)
            if temporal_arch == 'gru':  # single layer GRU
                self._rgb_head = GRU(rgb_feat_dim, d_model, num_layers=1)
                self._flow_head = GRU(flow_feat_dim, d_model, num_layers=1)
                self._sk_head = GRU(sk_feat_dim, d_model, num_layers=1)
            elif temporal_arch == 'deeper_gru':  # deeper GRU
                self._rgb_head = GRU(rgb_feat_dim, d_model, num_layers=3)
                self._flow_head = GRU(flow_feat_dim, d_model, num_layers=3)
                self._sk_head = GRU(sk_feat_dim, d_model, num_layers=3)
            else:
                raise NotImplementedError(temporal_arch)

            # Freeze the feature extractor
            # if STAGE > 2:
            #     # for name, param in self._rgb_feat.named_parameters():
            #     #     if 's4' not in name:
            #     #         param.requires_grad = False

            #     for param in self._rgb_feat.parameters():
            #         param.requires_grad = False
            #     for param in self._rgb_head.parameters():
            #         param.requires_grad = False
            #     for param in self._flow_feat.parameters():
            #         param.requires_grad = False
            #     for param in self._flow_head.parameters():
            #         param.requires_grad = False
            if STAGE > 1:
                for param in self._sk_feat.parameters():
                    param.requires_grad = False
                for param in self._sk_head.parameters():
                    param.requires_grad = False
            

            # # use audio
            # self._audio_dec = nn.Linear(d_model, 64)

            # binary predictor, hit or not
            self._coarse_pred = nn.Linear(d_model, 2)
            self._coarse_pred_ = nn.Linear(d_model, 2)

            # multi-label fine predictor
            self._fine_pred = nn.Linear(d_model, num_classes)
            self._fine_pred_ = nn.Linear(d_model, num_classes)

            # contextual module
            if use_ctx:
                self._ctx = GRUPrediction(num_classes + 1, num_classes + 1, d_model, num_layers=1)


        def forward(self, frame, flow, skeleton, coarse_label=None, fine_label=None, hand=None, max_seq_len=20, audio=None):

            rgb_feat, flow_feat, sk_feat = None, None, None
            # rgb visual embedding
            if frame is not None:
                batch_size, true_clip_len, channels, height, width = frame.shape

                clip_len = true_clip_len
                if self._require_clip_len > 0:
                    # TSM module requires clip len to be known
                    assert true_clip_len <= self._require_clip_len, \
                        'Expected {}, got {}'.format(
                            self._require_clip_len, true_clip_len)
                    if true_clip_len < self._require_clip_len:
                        frame = F.pad(
                            frame, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                        clip_len = self._require_clip_len

                # feature extractor
                if self._is_3d:
                    rgb_feat = self._rgb_feat(frame.transpose(1, 2)).transpose(1, 2)
                else:
                    rgb_feat = self._rgb_feat(frame.view(-1, channels, height, width)).reshape(batch_size, clip_len, -1)
                
                # head module
                rgb_feat = self._rgb_head(rgb_feat)
                enc_feat = rgb_feat

                # rgb2flow_feat = self._rgb2flow(rgb_feat)
                # rgb2sk_feat = self._rgb2sk(rgb_feat)

            # flow embedding
            if flow is not None:
                batch_size, true_clip_len, channels, height, width = flow.shape

                clip_len = true_clip_len
                if self._require_clip_len > 0:
                    # TSM module requires clip len to be known
                    assert true_clip_len <= self._require_clip_len, \
                        'Expected {}, got {}'.format(
                            self._require_clip_len, true_clip_len)
                    if true_clip_len < self._require_clip_len:
                        flow = F.pad(
                            flow, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                        clip_len = self._require_clip_len

                # feature extractor
                flow_feat = self._flow_feat(flow.view(-1, channels, height, width)).reshape(batch_size, clip_len, -1)
                
                # head module
                flow_feat = self._flow_head(flow_feat)
                if frame is None and skeleton is None:
                    enc_feat = flow_feat

                if STAGE >= 2:
                    # coarse-grained prediction
                    coarse_pred_ = self._coarse_pred_(flow_feat)

                    # fine-grained prediction
                    fine_pred_ = self._fine_pred_(flow_feat)



            if self._sk_feat is not None and skeleton is not None:
                batch_size, clip_len, _, _, _ = skeleton.shape

                # feature extractor
                sk_feat = self._sk_feat(skeleton.transpose(1, 2))
                sk_feat = sk_feat.mean(dim=1)
                sk_feat = sk_feat.mean(dim=-1)
                sk_feat = sk_feat.transpose(1, 2)
                sk_feat = sk_feat.view(batch_size, clip_len, -1)
                sk_feat = self._sk_head(sk_feat)

                # head module
                if frame is None and flow is None and audio is None:
                    enc_feat = sk_feat

            # coarse-grained prediction
            coarse_pred = self._coarse_pred(enc_feat)

            # fine-grained prediction
            fine_pred = self._fine_pred(enc_feat)

            # no contextual module
            if not self._use_ctx:
                if STAGE < 2 or flow is None:
                    return coarse_pred, fine_pred, rgb_feat, flow_feat, sk_feat
                else:
                    return coarse_pred + coarse_pred_, fine_pred + fine_pred_, rgb_feat, flow_feat, sk_feat

            coarse_pred_score = torch.softmax(coarse_pred, axis=2)
            fine_pred_score = torch.sigmoid(fine_pred).to(dtype=fine_pred.dtype)

            if coarse_label is None:
                coarse_label = non_maximum_suppression(coarse_pred_score, self._window)
                coarse_label = torch.argmax(coarse_label, axis=2)
            else:
                coarse_label = coarse_pred_score * coarse_label.unsqueeze(-1)
                coarse_label = torch.argmax(coarse_label, axis=2)

            if fine_label is None:
                fine_label = fine_pred_score

            seq_pred = torch.zeros(batch_size, max_seq_len, self._num_classes + 1, dtype=fine_label.dtype, device=self._device)
            seq_label = torch.zeros(batch_size, max_seq_len, self._num_classes + 1, dtype=fine_label.dtype, device=self._device)
            seq_mask = torch.ones((batch_size, max_seq_len), dtype=torch.bool, device=self._device)

            for i in range(batch_size):
                # sequence of ground truth labels
                selected_label = fine_label[i, coarse_label[i].bool()]
                seq_label[i, :selected_label.shape[0], 1:] = selected_label

                # sequence of predicted classes
                selected_pred = fine_pred_score[i, coarse_label[i].bool()]
                seq_pred[i, :selected_label.shape[0], 1:] = selected_pred
                for j in range(selected_label.shape[0]):
                    seq_pred[i, j, 0] = hand[i, int(torch.round(selected_pred[j, 0]))]
                    seq_label[i, j, 0] = hand[i, int(torch.round(selected_label[j, 0]))]

                # sequence mask
                seq_mask[i, :selected_label.shape[0]] = False

            # contextual module refine sequence
            seq_pred_refined = self._ctx(seq_pred)
            return coarse_pred, fine_pred, seq_pred_refined, seq_label, seq_mask

    def __init__(self, num_classes, feature_arch, temporal_arch, clip_len, step=1, window=5, use_ctx=True,
                 use_audio=False, device='cuda', multi_gpu=False):
        self._device = device
        self._multi_gpu = multi_gpu
        self._window = window
        self._use_ctx = use_ctx
        self._use_audio = use_audio
        self._model = F3Set.Impl(num_classes, feature_arch, temporal_arch, clip_len, step=step, window=window,
                                 use_ctx=use_ctx, use_audio=use_audio)

        if multi_gpu:
            self._model = nn.DataParallel(self._model)

        self._model.to(device)
        self._num_classes = num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None, acc_grad_iter=1, fg_weight=5, epoch=0):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        # coarse-grained frame binary classification weight
        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor([1, fg_weight]).to(self._device)

        epoch_loss = 0.
        with (torch.no_grad() if optimizer is None else nullcontext()):
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = loader.dataset.load_frame_gpu(batch, self._device)
                flow = loader.dataset.load_flow_gpu(batch, self._device)
                skeleton = loader.dataset.load_skeleton_gpu(batch, self._device)

                coarse_label = batch['coarse_label'].to(self._device)
                fine_label = batch['fine_label'].float().to(self._device)
                coarse_mask = batch['coarse_mask'].to(self._device)
                hand = batch['hand'].float().to(self._device)
                audio = batch['audio'].float().to(self._device)

                with torch.cuda.amp.autocast():
                    loss = 0.

                    # stage 0: optical flow extraction
                    if STAGE == 0:
                        coarse_pred, fine_pred, _, _, _ = self._model(None, flow, None)

                    # stage 1: skeleton extraction
                    elif STAGE == 1:
                        coarse_pred, fine_pred, _, _, _ = self._model(None, None, skeleton)
                    
                    # stage 2: rgb distillation
                    elif STAGE == 2:
                        _, _, rgb_feat, flow_feat, sk_feat = self._model(frame, flow, skeleton)

                        rgb2sk_loss = F.mse_loss(rgb_feat, sk_feat)
                        flow2sk_loss = F.mse_loss(flow_feat, sk_feat)
                        
                        loss += rgb2sk_loss
                        loss += flow2sk_loss

                        # if batch_idx % 50 == 0:
                        #     print(rgb2sk_loss.detach().item(), flow2sk_loss.detach().item())

                    # stage 3: rgb finetunning
                    elif STAGE == 3:
                        coarse_pred, fine_pred, _, _, _ = self._model(frame, flow, None)

                    # few-shot loss
                    if STAGE != 2:
                        # coarse-grained localization loss
                        coarse_loss = F.cross_entropy(coarse_pred.reshape(-1, 2), coarse_label.flatten(), **ce_kwargs)
                        if not math.isnan(coarse_loss.item()):
                            loss += coarse_loss

                        # fine-grained multi-label loss
                        fine_bce_loss = F.binary_cross_entropy_with_logits(fine_pred, fine_label.float(), reduction='none')
                        fine_bce_loss = fine_bce_loss * coarse_mask.unsqueeze(2).expand_as(fine_pred)
                        fine_mask = coarse_label.unsqueeze(2).expand_as(fine_pred)
                        masked_fine_loss = fine_bce_loss * fine_mask
                        fine_loss = masked_fine_loss.sum() / fine_mask.sum()
                        if not math.isnan(fine_loss.item()):
                            loss += fine_loss

                        # contextual loss
                        ctx_loss = 0.
                        if self._use_ctx and (~seq_mask).any():
                            ctx_loss = F.binary_cross_entropy_with_logits(seq_pred[~seq_mask], seq_label[~seq_mask])
                            if not math.isnan(ctx_loss.item()):
                                loss += ctx_loss


                    # if epoch >= 20 and epoch < 30:
                    #     coarse_pred, fine_pred, rgb_feat, sk_feat, _ = self._model(frame, skeleton)

                    #     rgb2sk_loss = F.mse_loss(rgb_feat, sk_feat)
                    #     loss += rgb2sk_loss

                    # # train with skeleton first
                    # else:
                    #     if epoch < 20:
                    #         # coarse_pred, fine_pred, seq_pred, seq_label, seq_mask = self._model(None, skeleton)
                    #         coarse_pred, fine_pred, _, _, _ = self._model(None, skeleton)
                    #     else:
                    #         coarse_pred, fine_pred, _, _, _ = self._model(frame, skeleton)

                    # # total_ops, total_params = profile(self._model, inputs=(frame, coarse_label, fine_label, hand))
                    # # if epoch >= 30:
                    #     # coarse-grained binary classification loss
                    #     # coarse_loss = F.cross_entropy(coarse_pred.reshape(-1, 2), coarse_label.flatten(), **ce_kwargs)
                    #     coarse_ce_loss = F.cross_entropy(coarse_pred.reshape(-1, 2), coarse_label.flatten(), **ce_kwargs, reduction='none')
                    #     masked_coarse_loss = coarse_ce_loss * coarse_mask.flatten()
                    #     coarse_loss = masked_coarse_loss.sum() / coarse_mask.sum() if coarse_mask.sum() > 0 else masked_coarse_loss.sum()
                    #     if not math.isnan(coarse_loss.item()):
                    #         loss += coarse_loss

                    #     # fine-grained multi-label loss
                    #     fine_bce_loss = F.binary_cross_entropy_with_logits(fine_pred, fine_label.float(), reduction='none')
                    #     fine_bce_loss = fine_bce_loss * coarse_mask.unsqueeze(2).expand_as(fine_pred)
                    #     fine_mask = coarse_label.unsqueeze(2).expand_as(fine_pred)
                    #     masked_fine_loss = fine_bce_loss * fine_mask
                    #     fine_loss = masked_fine_loss.sum() / fine_mask.sum()
                    #     if not math.isnan(fine_loss.item()):
                    #         loss += fine_loss

                    #     # contextual loss
                    #     ctx_loss = 0.
                    #     if self._use_ctx and (~seq_mask).any():
                    #         ctx_loss = F.binary_cross_entropy_with_logits(seq_pred[~seq_mask], seq_label[~seq_mask])
                    #         if not math.isnan(ctx_loss.item()):
                    #             loss += ctx_loss


                if optimizer is not None and loss != 0.:
                    step(optimizer, scaler, loss / acc_grad_iter, lr_scheduler=lr_scheduler,
                         backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                if loss != 0.:
                    epoch_loss += loss.detach().item()

                if batch_idx % 50 == 0:
                    print(epoch_loss / (batch_idx + 1))
                    
        return epoch_loss / len(loader)     # Avg loss

    def predict(self, frame, flow=None, skeleton=None, hand=None, use_amp=True):
        if not isinstance(frame, torch.Tensor):
            frame = torch.FloatTensor(frame)
        if len(frame.shape) == 4:  # (L, C, H, W)
            frame = frame.unsqueeze(0)
        frame = frame.to(self._device)

        if flow is not None:
            if not isinstance(flow, torch.Tensor):
                flow = torch.FloatTensor(flow)
            if len(frame.shape) == 4:  # (L, C, H, W)
                flow = flow.unsqueeze(0)
            flow = flow.to(self._device)
        
        if skeleton is not None:
            if not isinstance(skeleton, torch.Tensor):
                skeleton = torch.FloatTensor(skeleton)
            if len(skeleton.shape) == 4:  # (L, C, H, W)
                skeleton = skeleton.unsqueeze(0)
            skeleton = skeleton.to(self._device)
        
        hand = hand.to(self._device)

        self._model.eval()
        with torch.no_grad():

            # input_frame = torch.randn(1, 1, 2, 17, 2).to(self._device)
            # for _ in range(10):  # Perform 10 warm-up runs
            #     _ = self._model(input_frame)
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            #
            # # Start timing
            # start_event.record()
            # _ = self._model(input_frame)  # Perform inference
            # end_event.record()
            #
            # # Wait for events to complete
            # torch.cuda.synchronize()
            #
            # # Calculate elapsed time
            # inference_time = start_event.elapsed_time(end_event)  # Time in milliseconds
            # print(f"Inference time per frame: {inference_time:.3f} ms")
            #
            # exit(0)

            with torch.cuda.amp.autocast() if use_amp else nullcontext():
                if STAGE == 0:
                    coarse_pred, fine_pred, _, _, _ = self._model(None, flow, None)
                elif STAGE == 1:
                    coarse_pred, fine_pred, _, _, _ = self._model(None, None, skeleton)
                else:
                    coarse_pred, fine_pred, _, _, _ = self._model(frame, flow, None)
            coarse_pred = torch.softmax(coarse_pred, axis=2)
            coarse_pred = non_maximum_suppression(coarse_pred, self._window)
            coarse_pred_cls = torch.argmax(coarse_pred, axis=2)

            fine_pred_refine = fine_pred 
            # if self._use_ctx:
            #     for i in range(coarse_pred_cls.size(0)):
            #         shot_id = 0
            #         for j in range(coarse_pred_cls.size(1)):
            #             if coarse_pred_cls[i, j] == 1:
            #                 fine_pred_refine[i, j] = seq_pred[i, shot_id, 1:]
            #                 shot_id += 1

            fine_pred = torch.sigmoid(fine_pred_refine)
            return coarse_pred_cls.cpu().numpy(), coarse_pred.cpu().numpy(), fine_pred.cpu().numpy()


def evaluate(model, dataset, classes, delta=1, window=5, device='cuda'):
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, 2), np.float32),
            np.zeros((video_len, len(classes)), np.float32),
            np.zeros(video_len, np.int32))

    classes_inv = {v: k for k, v in classes.items()}
    classes_inv[0] = 'NA'

    # Do not up the batch size if the dataset augments
    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    for clip in tqdm(DataLoader(
            dataset, num_workers=BASE_NUM_WORKERS * 2, pin_memory=True,
            batch_size=batch_size
    )):

        if batch_size > 1:
            # Batched by dataloader
            _, batch_coarse_scores, batch_fine_scores = model.predict(clip['frame'], clip['flow'], clip['skeleton'], clip['hand'])
            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                coarse_scores, fine_scores, support = pred_dict[video]
                coarse_pred_scores = batch_coarse_scores[i]
                fine_pred_scores = batch_fine_scores[i]

                start = clip['start'][i].item()
                if start < 0:
                    coarse_pred_scores = coarse_pred_scores[-start:, :]
                    fine_pred_scores = fine_pred_scores[-start:, :]
                    start = 0
                end = start + coarse_pred_scores.shape[0]
                if end >= coarse_scores.shape[0]:
                    end = coarse_scores.shape[0]
                    coarse_pred_scores = coarse_pred_scores[:end - start, :]
                    fine_pred_scores = fine_pred_scores[:end - start, :]
                coarse_scores[start:end, :] += coarse_pred_scores
                fine_scores[start:end, :] += fine_pred_scores
                support[start:end] += 1

    # evaluation metrices
    f = open('error_sequences_fusion.txt', 'w')
    edit_scores = []
    f1_lcl = np.zeros((1, 3), int)
    f1_element = np.zeros((len(classes), 3), int)
    f1_event = dict()
    for video, (coarse_scores, fine_scores, support) in sorted(pred_dict.items()):
        coarse_label, fine_label = dataset.get_labels(video)
        coarse_scores /= support[:, None]
        fine_scores /= support[:, None]

        # argmax pred
        # coarse_scores = non_maximum_suppression_np(coarse_scores, window)
        coarse_pred = np.argmax(coarse_scores, axis=1)

        # dataset specific
        fine_pred = np.zeros_like(fine_scores, int)

        # # f3set-tennis
        # for i in range(len(fine_scores)):
        #     for start, end in [[0, 2], [2, 5], [5, 8], [16, 24], [25, 29]]:
        #         max_idx = np.argmax(fine_scores[i, start:end])
        #         fine_pred[i, start + max_idx] = 1
        #     if fine_scores[i, 24] > 0.5:  # approach
        #         fine_pred[i, 24] = 1
        #     if fine_pred[i, 5] != 1:  # not a serve
        #         for start, end in [[8, 10], [10, 16]]:
        #             max_idx = np.argmax(fine_scores[i, start:end])
        #             fine_pred[i, start + max_idx] = 1

        # shuttleset
        for i in range(len(fine_scores)):
            for start, end in [[0, 2], [2, 12]]: # far / near end player ; shot type.
                max_idx = np.argmax(fine_scores[i, start:end])
                fine_pred[i, start + max_idx] = 1

        # for i in range(len(fine_scores)):
        #     for start, end in [[0, 2], [2, 5]]:
        #         max_idx = np.argmax(fine_scores[i, start:end])
        #         fine_pred[i, start + max_idx] = 1
        #     if fine_scores[i, 13] > 0.5:  # approach
        #         fine_pred[i, 13] = 1
        #     if fine_pred[i, 2] != 1:  # not a serve
        #         for start, end in [[5, 7], [7, 13]]:
        #             max_idx = np.argmax(fine_scores[i, start:end])
        #             fine_pred[i, start + max_idx] = 1

        fine_pred = coarse_pred[:, np.newaxis] * fine_pred

        # print(coarse_pred.shape)
        # event localizer F1 scores
        for i in range(len(coarse_pred)):
            if coarse_pred[i] == 1 and sum(coarse_label[max(0, i - delta):min(len(coarse_pred), i + delta + 1)]) == 1:
                f1_lcl[0, 0] += 1  # tp
            if coarse_pred[i] == 1 and sum(coarse_label[max(0, i - delta):min(len(coarse_pred), i + delta + 1)]) == 0:
                f1_lcl[0, 1] += 1  # fp
            if coarse_label[i] == 1 and sum(coarse_pred[max(0, i - delta):min(len(coarse_pred), i + delta + 1)]) == 0:
                f1_lcl[0, 2] += 1  # fn

        # element F1 scores
        for i in range(len(fine_pred)):
            for j in range(len(fine_pred[0])):
                if fine_pred[i, j] == 1 and sum(
                        fine_label[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 1:
                    f1_element[j, 0] += 1  # tp
                if fine_pred[i, j] == 1 and sum(
                        fine_label[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 0:
                    f1_element[j, 1] += 1  # fp
                if fine_label[i, j] == 1 and sum(
                        fine_pred[max(0, i - delta):min(len(fine_pred), i + delta + 1), j]) == 0:
                    f1_element[j, 2] += 1  # fn

        print_preds, print_gts = [], []
        # tp, fp, fn
        for i in range(len(fine_pred)):
            if coarse_label[i] == 1:
                print_gt = []
                for j in range(len(fine_pred[0])):
                    if fine_label[i, j] == 1:
                        print_gt.append(classes_inv[j + 1])
                print_gts.append('_'.join(print_gt))
            # else:
            #     print_gts.append(str(i))
            if coarse_pred[i] == 1:
                print_pred = []
                for j in range(len(fine_pred[0])):
                    if fine_pred[i, j] == 1:
                        print_pred.append(classes_inv[j + 1])
                print_preds.append('_'.join(print_pred))

        fine_label = fine_label  # [:, :24]#[:, [0, 1, 5, 6, 7, 8, 9, 25, 26, 27, 28]]
        fine_pred = fine_pred  # [:, :24]#[:, [0, 1, 5, 6, 7, 8, 9, 25, 26, 27, 28]]

        labels = [int(''.join(str(x) for x in row), 2) for row in fine_label]
        preds = [int(''.join(str(x) for x in row), 2) for row in fine_pred]
        preds = coarse_pred * preds

        # event F1 scores
        for i in range(len(preds)):
            if preds[i] > 0 and preds[i] in labels[max(0, i - delta):min(len(preds), i + delta + 1)]:
                if preds[i] not in f1_event:
                    f1_event[preds[i]] = [1, 0, 0]
                else:
                    f1_event[preds[i]][0] += 1
            if preds[i] > 0 and sum(labels[max(0, i - delta):min(len(preds), i + delta + 1)]) == 0:
                if preds[i] not in f1_event:
                    f1_event[preds[i]] = [0, 1, 0]
                else:
                    f1_event[preds[i]][1] += 1
            if labels[i] > 0 and labels[i] not in preds[max(0, i - delta):min(len(preds), i + delta + 1)]:
                if labels[i] not in f1_event:
                    f1_event[labels[i]] = [0, 0, 1]
                else:
                    f1_event[labels[i]][2] += 1

        gt = [k for k, g in groupby(labels) if k != 0]
        pred = [k for k, g in groupby(preds) if k != 0]

        # record error sequence
        if len(pred) == len(gt):
            for j in range(len(pred)):
                if pred[j] != gt[j]:
                    f.write(video + '\n')
                    f.write('->'.join(print_preds) + '\n')
                    f.write('\n')
                    f.write('->'.join(print_gts) + '\n')
                    f.write('\n')
                    f.write('------------------------')
                    f.write('\n')
                    break
        else:
            f.write(video + '\n')
            f.write('->'.join(print_preds) + '\n')
            f.write('\n')
            f.write('->'.join(print_gts) + '\n')
            f.write('\n')
            f.write('------------------------')
            f.write('\n')

        edit_scores.append(edit_score(pred, gt))

    f.close()

    precision = f1_lcl[:, 0] / (f1_lcl[:, 0] + f1_lcl[:, 1] + 1e-10)
    recall = f1_lcl[:, 0] / (f1_lcl[:, 0] + f1_lcl[:, 2] + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    print('Mean F1 (LCL):', np.mean(f1))
    print()

    f1, count = 0, 0
    for value in f1_event.values():
        if sum(value) == 0:
            continue
        precision = value[0] / (value[0] + value[1] + 1e-10)
        recall = value[0] / (value[0] + value[2] + 1e-10)
        f1 += 2 * precision * recall / (precision + recall + 1e-10)
        count += 1
    f1 /= count

    print('Mean F1 (event):', np.mean(f1))
    print()

    precision = f1_element[:, 0] / (f1_element[:, 0] + f1_element[:, 1] + 1e-10)
    recall = f1_element[:, 0] / (f1_element[:, 0] + f1_element[:, 2] + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    print(f1)
    print('Mean F1 (element):', np.mean(f1))  # [:24]))
    print()

    # edit_scores = [x for x in edit_scores if not math.isnan(x)]
    print('Edit score:', sum(edit_scores) / len(edit_scores))
    return sum(edit_scores) / len(edit_scores)


def get_last_epoch(save_dir):
    max_epoch = -1
    for file_name in os.listdir(save_dir):
        if not file_name.startswith('optim_'):
            continue
        epoch = int(os.path.splitext(file_name)[0].split('optim_')[1])
        if epoch > max_epoch:
            max_epoch = epoch
    return max_epoch


def get_best_epoch_and_history(save_dir, criterion):
    data = load_json(os.path.join(save_dir, 'loss.json'))
    if criterion == 'edit':
        key = 'val_edit'
        best = max(data, key=lambda x: x[key])
    else:
        key = 'val'
        best = min(data, key=lambda x: x[key])
    return data, best['epoch'], best[key]


def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'elements.txt'))

    dataset_len = EPOCH_NUM_FRAMES // (args.clip_len * args.stride)
    dataset_kwargs = {
        'crop_dim': args.crop_dim, 'stride': args.stride
    }
    if args.use_audio:
        dataset_kwargs['audio_dir'] = 'vid_audio'

    print('Dataset size:', dataset_len)
    train_data = ActionSeqDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.clip_len, dataset_len, is_eval=False, dilate_len=args.dilate_len, stage=STAGE,
        num_samples=80,
        **dataset_kwargs)
    train_data.print_info()
    val_data = ActionSeqDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.clip_len, dataset_len // 4, dilate_len=args.dilate_len, stage=STAGE, num_samples=20,
        **dataset_kwargs)
    val_data.print_info()

    val_data_frames = None
    if args.criterion == 'edit':
        # Only perform edit score evaluation during training if criterion is edit
        val_data_frames = ActionSeqVideoDataset(
            classes, os.path.join('data', args.dataset, 'val.json'),
            args.frame_dir, args.clip_len, overlap_len=0, num_samples=20, **dataset_kwargs)

    return classes, train_data, val_data, None, val_data_frames


def load_from_save(
        args, model, optimizer, scaler, lr_scheduler
):
    assert args.save_dir is not None
    epoch = get_last_epoch(args.save_dir)

    print('Loading from epoch {}'.format(epoch))
    model.load(torch.load(os.path.join(
        args.save_dir, 'checkpoint_{:03d}.pt'.format(epoch))))

    if args.resume:
        opt_data = torch.load(os.path.join(
            args.save_dir, 'optim_{:03d}.pt'.format(epoch)))
        optimizer.load_state_dict(opt_data['optimizer_state_dict'])
        scaler.load_state_dict(opt_data['scaler_state_dict'])
        lr_scheduler.load_state_dict(opt_data['lr_state_dict'])

    losses, best_epoch, best_criterion = get_best_epoch_and_history(
        args.save_dir, args.criterion)
    return epoch, losses, best_epoch, best_criterion


def store_config(file_path, args, num_epochs, classes):
    config = {
        'dataset': args.dataset,
        'num_classes': len(classes),
        'feature_arch': args.feature_arch,
        'temporal_arch': args.temporal_arch,
        'use_ctx': args.use_ctx,
        'use_audio': args.use_audio,
        'clip_len': args.clip_len,
        'batch_size': args.batch_size,
        'crop_dim': args.crop_dim,
        'window': args.window,
        'stride': args.stride,
        'num_epochs': num_epochs,
        'warm_up_epochs': args.warm_up_epochs,
        'learning_rate': args.learning_rate,
        'start_val_epoch': args.start_val_epoch,
        'gpu_parallel': args.gpu_parallel,
        'epoch_num_frames': EPOCH_NUM_FRAMES,
        'dilate_len': args.dilate_len
    }
    store_json(file_path, config, pretty=True)


def get_num_train_workers(args):
    n = BASE_NUM_WORKERS * 2
    return min(os.cpu_count(), n)


def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    if args.num_workers is not None:
        global BASE_NUM_WORKERS
        BASE_NUM_WORKERS = args.num_workers

    assert args.batch_size % args.acc_grad_iter == 0
    if args.start_val_epoch is None:
        args.start_val_epoch = args.num_epochs - BASE_NUM_VAL_EPOCHS
    if args.crop_dim <= 0:
        args.crop_dim = None

    classes, train_data, val_data, train_data_frames, val_data_frames = get_datasets(args)

    def worker_init_fn(id):
        random.seed(id + epoch * 100)
    loader_batch_size = args.batch_size // args.acc_grad_iter
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=get_num_train_workers(args),
        prefetch_factor=1, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=BASE_NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    model = F3Set(len(classes), args.feature_arch, args.temporal_arch, clip_len=args.clip_len, step=args.stride, 
                  window=args.window, use_ctx=args.use_ctx, use_audio=args.use_audio, multi_gpu=args.gpu_parallel)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    # obtain best epoch from previous stage
    if STAGE == 2:
        # stage0_save_dir = args.save_dir.replace('stage2', 'stage0')
        # losses, best_epoch, best_criterion = get_best_epoch_and_history(stage0_save_dir, args.criterion)
        # print('Loading from STAGE 0 epoch {}'.format(best_epoch))
        # model_stage0 = torch.load(os.path.join(stage0_save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch)))
        stage1_save_dir = args.save_dir.replace('stage2', 'stage1')
        losses, best_epoch, best_criterion = get_best_epoch_and_history(stage1_save_dir, args.criterion)
        print('Loading from STAGE 1 epoch {}'.format(best_epoch))
        model_stage1 = torch.load(os.path.join(stage1_save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch)))
        # # update stage 1 flow modules with stage 0
        # for name, param in model_stage1['model_state_dict'].items():
        #     if 'flow' in name:
        #         model_stage1['model_state_dict'][name] = model_stage0['model_state_dict'][name]
        model.load(model_stage1)
    elif STAGE == 3:
        stage2_save_dir = args.save_dir.replace('stage3', 'stage2')
        losses, best_epoch, best_criterion = get_best_epoch_and_history(stage2_save_dir, args.criterion)
        print('Loading from STAGE 2 epoch {}'.format(best_epoch))
        model.load(torch.load(os.path.join(stage2_save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

    # Warmup schedule
    num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
    num_epochs, lr_scheduler = get_lr_scheduler(
        args, optimizer, num_steps_per_epoch)

    losses = []
    best_epoch = None
    best_criterion = 0 if args.criterion == 'edit' else float('inf')
    best_loss, stop_criterion = float('inf'), 0

    epoch = 0
    if args.resume:
        epoch, losses, best_epoch, best_criterion = load_from_save(args, model, optimizer, scaler, lr_scheduler)
        epoch += 1

    # Write it to console
    store_config('/dev/stdout', args, num_epochs, classes)

    for epoch in range(epoch, num_epochs):
        train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler, acc_grad_iter=args.acc_grad_iter, epoch=epoch)
        val_loss = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter, epoch=epoch)
        print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
            epoch, train_loss, val_loss))

        val_edit = 0
        if args.criterion == 'loss':
            if val_loss < best_criterion:
                best_criterion = val_loss
                best_epoch = epoch
                print('New best epoch!')
        elif args.criterion == 'edit':
            if epoch >= args.start_val_epoch:
                val_edit = evaluate(model, val_data_frames, classes, window=args.window)
                if args.criterion == 'edit' and val_edit > best_criterion:
                    best_criterion = val_edit
                    best_epoch = epoch
                    print('New best epoch!')
        else:
            print('Unknown criterion:', args.criterion)

        losses.append({
            'epoch': epoch, 'train': train_loss, 'val': val_loss, 'val_edit': val_edit})
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            store_json(os.path.join(args.save_dir, 'loss.json'), losses,
                        pretty=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir,
                    'checkpoint_{:03d}.pt'.format(epoch)))
            clear_files(args.save_dir, r'optim_\d+\.pt')
            torch.save(
                {'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'lr_state_dict': lr_scheduler.state_dict()},
                os.path.join(args.save_dir,
                                'optim_{:03d}.pt'.format(epoch)))
            store_config(os.path.join(args.save_dir, 'config.json'),
                            args, num_epochs, classes)

    print('Best epoch: {}\n'.format(best_epoch))

    if args.save_dir is not None:
        model.load(torch.load(os.path.join(
            args.save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))

        # Evaluate on hold out splits
        eval_splits = ['test']
        for split in eval_splits:
            split_path = os.path.join(
                'data', args.dataset, '{}.json'.format(split))
            if os.path.exists(split_path):
                if args.use_audio:
                    audio_dir = 'vid_audio'
                else:
                    audio_dir = None
                split_data = ActionSeqVideoDataset(classes, split_path, args.frame_dir, args.clip_len,
                                                   overlap_len=args.clip_len//2, crop_dim=args.crop_dim,
                                                   stride=args.stride, is_test=True, audio_dir=audio_dir)
                split_data.print_info()
                evaluate(model, split_data, classes, window=args.window)


if __name__ == '__main__':
    main(get_args())
