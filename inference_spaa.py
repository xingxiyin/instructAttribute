from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
from attention_processor import register_attention_processors_amplify, register_attention_processors_init
import seq_aligner
from PIL import Image
from tqdm import tqdm
import os
import cv2
import random
from skimage.metrics import structural_similarity as ssim
from skimage import metrics
from copy import deepcopy
import torch
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from torch.nn import functional as F
from transformers import ViTModel
import time
import argparse


from diffusers.utils.testing_utils import require_torch


class LocalBlend:
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3, MAX_NUM_WORDS = 77, device="cuda", tokenizer=None,  NUM_DDIM_STEPS=50, start_blend=0.2):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold
        self.device = device
        self.counter = 0
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)

    def __call__(self, x_t, attention_store, k=1, MAX_NUM_WORDS=77):
        self.counter += 1
        if self.counter > self.start_blend:
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            # for item in maps:
            #     print(item.shape)
            # maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]

            _, num_pixels, _ = maps[0].shape
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, int(num_pixels**0.5), int(num_pixels**0.5), MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            maps = (maps * self.alpha_layers).sum(-1).mean(1)
            mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
            mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
            mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
            mask = mask.gt(self.threshold)
            mask = (mask[:1] + mask[1:]).float()

            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t


class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return 0

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    @property
    def num_uncond_att_layers(self, LOW_RESOURCE=False):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, LOW_RESOURCE=False):
        # print("AttentionControl 1 ", is_cross, place_in_unet, attn.shape)
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        # print("AttentionControl 2 ", is_cross, place_in_unet, attn.shape)
        return attn



class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):
    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
                    # print("between_steps", key, i, self.step_store[key][i].shape)
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # print("AttentionStore", key, attn.shape)
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn


class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(self,
                 prompts,
                 tokenizer,
                 device,
                 num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            tokenizer).to(device)
        # torch (num_steps+1, len(prompts), 1, 1, max_num_words)

        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
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


class AttentionReplace(AttentionControlEdit):
    def __init__(self,
                 prompts,
                 tokenizer,
                 device,
                 num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, tokenizer, device, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)


class AttentionRefine(AttentionControlEdit):
    def __init__(self,
                 prompts,
                 tokenizer,
                 device,
                 num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, tokenizer, device, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace



class AttentionReweight(AttentionControlEdit):
    def __init__(self,
                 prompts,
                 tokenizer,
                 device,
                 num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, tokenizer, device, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace



def get_equalizer(text: str,
                  word_select: Union[int, Tuple[int, ...]],
                  values: Union[List[float], Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int,
                        ) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                # print(location, item.shape, item.reshape(1, -1, res, res, item.shape[-1]).shape)
                # up torch.Size([16, 256, 77]) torch.Size([1, 16, 16, 16, 77])

                # print(item.reshape(1, -1, res, res, item.shape[-1]).shape)  #  torch.Size([1, 8, 16, 16, 77])
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out


def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = cv2.resize(heatmap, img.shape[:2])
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    # print(type(image), image.shape)
    # image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    # image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)

    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis



def show_cross_attention(prompt: str,
                         attention_store: AttentionStore,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         from_where: List[str],
                         select: int = 0,
                         orig_image=None):
    tokens = tokenizer.encode(prompt)
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select).detach().cpu()
    images = []


    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        # print(image.shape)
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image))
            # image = text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    return images


def run_and_display(ldm_stable, prompts, controller, latent=None, run_baseline=False, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE)
    return images, x_t



def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn((1, model.unet.in_channels, height // 8, width // 8), generator=generator)
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        controller,
        idx_list,
        ratio=0.005,
        resolution=512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,
):
    register_attention_processors_amplify(model, controller, idx_list, ratio)
    # register_attention_processors_init(model, controller)

    height,width =resolution,resolution
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)

    image = latent2image(model.vae, latents)

    return image, latents

def main(args):
    LOW_RESOURCE = False
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.model).to(device)
    tokenizer = ldm_stable.tokenizer

    ####################################################################################################################

    # output path
    outputPath = args.output
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    prompt_template = args.prompt
    subject = args.subject
    attribute = args.attribute
    seed = args.seed
    resolution = args.resolution
    ratio = args.ratio

    # source prompt and target prompt
    prompts = ["{} {}".format(prompt_template, subject), "{} {} {}".format(prompt_template, attribute, subject)]
    idx_list = [4] # index of attribute in the target prompt

    # output path
    prompt_dir = "_".join(prompts[0].split(" "))
    outputPathTemp = os.path.join(outputPath, prompt_dir, str(seed))
    if not os.path.exists(outputPathTemp):
        os.makedirs(outputPathTemp)

    lb = LocalBlend(prompts, (subject, subject), device=device, tokenizer=tokenizer)
    g_cpu = torch.Generator().manual_seed(int(seed))
    controller = AttentionRefine(prompts, tokenizer, device, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=0.4, local_blend=lb)
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller,
                                      idx_list=idx_list,
                                      ratio=ratio,
                                      resolution=resolution,
                                      latent=None,
                                      num_inference_steps=NUM_DIFFUSION_STEPS,
                                      guidance_scale=GUIDANCE_SCALE,
                                      generator=g_cpu, low_resource=LOW_RESOURCE)

    image = Image.fromarray(images[0])
    image.save(os.path.join(outputPathTemp, "source.png"))

    image = Image.fromarray(images[1])
    image.save(os.path.join(outputPathTemp, "{}.png".format(attribute)))


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="A simple script demonstrating argparse.")

    # Add arguments
    parser.add_argument("--model", type=str, required=True, help="prompt template")
    parser.add_argument("--prompt", type=str, required=True, help="prompt template")
    parser.add_argument("--subject", type=str, required=True, help="subject name")
    parser.add_argument("--attribute", type=str, required=True,   help="attribute")
    parser.add_argument("--seed", type=int, default=66666, help = "seed")
    parser.add_argument("--output", type=str, default="./outputs", help = "subject name")
    parser.add_argument("--ratio", type=float, default=0.005, help="image resolution")
    parser.add_argument("--resolution", type=int, default=512, help="image resolution")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)