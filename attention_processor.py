from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from diffusers.models import attention_processor
from torch.nn import functional as F
import numpy as np

import cv2
import copy
from PIL import Image


T = torch.Tensor

def minmax_normalize(batch_maps):
    min_val = batch_maps.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_val = batch_maps.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    return (batch_maps - min_val) / (max_val - min_val + 1e-5)


class selfAttentionProcessorInit(torch.nn.Module):
    def __init__(self, name, controller):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.name = name
        self.controller = controller

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        is_cross = encoder_hidden_states is not None
        # print("name", self.name, "is_cross", is_cross)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # query, key, value
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # multi head
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)


        # attention controller
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        place_in_unet = self.name.split("_")[0]
        self.controller(attn=attention_probs, is_cross=is_cross, place_in_unet=place_in_unet)

        # feature output
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)



        # linear proj
        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # print("hidden_states dropout", hidden_states.shape)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class selfAttentionProcessorReplace(torch.nn.Module):
    def __init__(self, name, controller=None):
        super().__init__()
        self.name = name
        self.controller = controller

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        is_cross = encoder_hidden_states is not None
        # print("name", self.name, "is_cross", is_cross)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # query, key, value
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # replace
        query[1, :, :] = query[0, :, :]
        query[3, :, :] = query[2, :, :]
        key[1, :, :] = key[0, :, :]
        key[3, :, :] = key[2, :, :]

        # print(query.shape, key.shape)

        # multi head
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)


        # attention controller
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        place_in_unet = self.name.split("_")[0]
        self.controller(attn=attention_probs, is_cross=is_cross, place_in_unet=place_in_unet)

        # feature output
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # print("hidden_states dropout", hidden_states.shape)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



class crossAttentionProcessorInit(nn.Module):
    def __init__(self, name, controller):
        super().__init__()
        self.name = name
        self.controller = controller

    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        is_cross = encoder_hidden_states is not None
        # print("name", self.name, "is_cross", is_cross)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, seqadaptive_instance_normalizationuence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # query, key, value
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # multi-head
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        print("query", query.shape, "key", key.shape, "value", value.shape, )

        # attention controller
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        place_in_unet = self.name.split("_")[0]
        self.controller(attn=attention_probs, is_cross=is_cross, place_in_unet=place_in_unet)

        # feature output
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)


        # linear proj
        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # print("hidden_states dropout", hidden_states.shape)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states




class crossAttentionProcessorAmplify(nn.Module):
    def __init__(self, name, controller, idx_list, amplify_ratio=0.02):
        super().__init__()
        self.name = name
        self.controller = controller
        self.idx_list = idx_list
        self.amplify_ratio = amplify_ratio


    def __call__(self, attn: attention_processor.Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        cur_timestep = str(attn.timestep.detach().cpu().numpy() - 1)


        residual = hidden_states
        input_ndim = hidden_states.ndim
        is_cross = encoder_hidden_states is not None
        # print("name", self.name, "is_cross", is_cross)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, seqadaptive_instance_normalizationuence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # query, key, value
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        # print("query", query.shape, "value", value.shape, "key", key.shape)
        # print("cur_timestep", cur_timestep, type(cur_timestep))


        amplify = int(cur_timestep) * self.amplify_ratio
        if amplify < 1.0:
            amplify = 1.0

        for _ in self.idx_list:
            value[1, [_], :] = value[1, [_], :] * amplify
            value[3, [_], :] = value[3, [_], :] * amplify



        # multi-head
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)


        # attention controller
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        place_in_unet = self.name.split("_")[0]
        self.controller(attn=attention_probs, is_cross=is_cross, place_in_unet=place_in_unet)

        # feature output
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)


        # linear proj
        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        # print("hidden_states dropout", hidden_states.shape)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



def register_attention_processors_init(pipeline, controller):
    attn_procs = {}
    number_of_self_att_layers, number_of_cross_att_layers = 0, 0
    for i, name in enumerate(pipeline.unet.attn_processors.keys()):
        is_self_attention = True if 'attn1' in name else False
        if is_self_attention:
            number_of_self_att_layers += 1
            attention_processor_name = ".".join(name.split(".")[:-1])
            attn_procs[name] = selfAttentionProcessorInit(attention_processor_name,
                                                       controller,
                                                       )
        else:
            number_of_cross_att_layers += 1
            attention_processor_name = ".".join(name.split(".")[:-1])
            attn_procs[name] = crossAttentionProcessorInit(attention_processor_name,
                                                       controller,
                                                       )

    controller.num_att_layers = number_of_self_att_layers + number_of_cross_att_layers
    pipeline.unet.set_attn_processor(attn_procs)


def register_attention_processors_amplify(pipeline, controller, idx_list, ratio):
    attn_procs = {}
    number_of_self_att_layers, number_of_cross_att_layers = 0, 0
    for i, name in enumerate(pipeline.unet.attn_processors.keys()):
        is_self_attention = 'attn1' in name
        if is_self_attention:
            number_of_self_att_layers += 1
            attention_processor_name = ".".join(name.split(".")[:-1])
            attn_procs[name] = selfAttentionProcessorReplace(attention_processor_name,
                                                       controller,
                                                       )
        else:
            number_of_cross_att_layers += 1
            attention_processor_name = ".".join(name.split(".")[:-1])
            attn_procs[name] = crossAttentionProcessorAmplify(attention_processor_name,
                                                       controller,
                                                       idx_list,
                                                       ratio,
                                                       )

    controller.num_att_layers = number_of_self_att_layers + number_of_cross_att_layers
    pipeline.unet.set_attn_processor(attn_procs)