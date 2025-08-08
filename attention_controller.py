from __future__ import annotations
from typing import Optional, Union, Tuple, List, Dict
import abc

import cv2
from cv2 import dilate
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as nnf
import seq_aligner
import ptp_utils

T = torch.Tensor

def get_equalizer(text: str,
                  word_select: Union[int, Tuple[int, ...]],
                  values: Union[List[float], Tuple[float, ...]],
                  tokenizer):

    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)

    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer



class LocalBlend:

    def __init__(self, prompts, mask, device, NUM_DDIM_STEPS=50, start_blend=0.2):
        self.batch_size = len(prompts)
        self.device = device
        self.mask = mask
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0

    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
            mask = cv2.resize(self.mask, x_t.shape[2:])
            mask = mask / 255.0

            # mask = dilate(mask, np.ones((1, 1), np.uint8), iterations=1)
            # mask = torch.from_numpy(mask).float().cuda().view(1, 1, 64, 64)

            mask = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(mask), dim=0), dim=0)
            mask = torch.cat([mask] * self.batch_size).to(self.device)
            mask = mask.float()
            # print("mask", mask.shape)
            # print("x_t", x_t.shape)  # x_t torch.Size([2, 4, 64, 64])   x_t[:1] torch.Size([1, 4, 64, 64])

            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t


class AttentionStore(abc.ABC):
    def __init__(self, local_blend=None):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.cur_attention = {}

        self.local_blend=local_blend

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    @property
    def num_uncond_att_layers(self):
        return 0

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()


    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def get_cur_attention(self):
        return self.cur_attention

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}


    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        self.step_store[key].append(attn)
        self.cur_attention.update(self.step_store)
        return attn

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

        # print("cur_att_layer",  self.cur_att_layer, "cur_step", self.cur_step, "num_att_layers", self.num_att_layers, "num_uncond_att_layers", self.num_uncond_att_layers)
        return attn


class AttentionRefine(AttentionStore, abc.ABC):
    def __init__(self,
                 prompts,
                 tokenizer,
                 device,
                 num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 equalizer=None,
                 local_blend=None):
        super(AttentionRefine, self).__init__()

        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            tokenizer).to(device)
        # torch (num_steps+1, len(prompts), 1, 1, max_num_words)

        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps

        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

        self.device = device
        self.equalizer = equalizer
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    def replace_cross_attention(self, attn_base, att_replace):
        # if self.equalizer is not None:
        #     attn_base = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace


    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionRefine, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                            1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
