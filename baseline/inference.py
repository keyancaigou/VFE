"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random
import json 
from tqdm import tqdm
from enum import auto, Enum
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from BaseLine.common.registry import registry
from decord import VideoReader

from BaseLine.processors import transforms_video
from BaseLine.processors.base_processor import BaseProcessor
from BaseLine.processors.randaugment import VideoRandomAugment
from BaseLine.processors import functional_video as F
from omegaconf import OmegaConf
from torchvision import transforms
import random as rnd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
from BaseLine.common.registry import registry
from BaseLine.processors.video_processor import ToTHWC,ToUint8,load_video
from BaseLine.processors import Blip2ImageEvalProcessor
            
from BaseLine.models.ImageBind.data import load_and_transform_audio_data
from BaseLine.common.config import Config
from BaseLine.common.dist_utils import get_rank
from BaseLine.common.registry import registry
from BaseLine.conversation.conversation_video import Conversation, default_conversation,SeparatorStyle
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from BaseLine.datasets.builders import *
from BaseLine.models import *
from BaseLine.processors import *
from BaseLine.runners import *
from BaseLine.tasks import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/BaseLine_eval.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, filename, save_pretty=False, sort_keys=False):
            with open(filename, "w") as f:
                if save_pretty:
                    f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
                else:
                    json.dump(data, f)
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
# vid_h5 = h5py.File(h5_path, "r", driver=None)

def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))
def file_exists(file_path):
    return os.path.exists(file_path)


# ========================================
#             Model Initialization
# ========================================




# video_path = '/data1/lch/datasets/vlep/vlep_ytb_clips/_7ui0Pd8Bd0_subs_004_00:05:00_00:06:00_ep.mp4'

def load_anet_video(video_path, n_frms=8, height=-1, width=-1, timestamps=None,sampling="uniform", return_msg = False):
 
    vr = VideoReader(uri=video_path, height=height, width=width)
   
    fps = vr.get_avg_fps()
    start_time = timestamps[0]
    end_time = timestamps[1]
        # 计算开始和结束帧的索引
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    vlen = end_frame - start_frame
    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start_frame, end_frame, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices)
    # print(type(temp_frms))
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        self.image_vis_processor = Blip2ImageEvalProcessor()
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and ('</Video>' in conv.messages[-1][1] or '</Image>' in conv.messages[-1][1]):  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]
        
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy()
    
    def upload_video(self, video_path, conv, img_list):

        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            # print(video_path)
            
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
            video, msg = load_video(
                video_path=video_path,
                n_frms=8,
                height=224,
                width=224,
                # timestamps=timestamps,
                sampling ="uniform", return_msg = True
            )
            video = self.vis_processor.transform(video)
            video = video.unsqueeze(0).to(self.device)
            # print(image)
        else:
            raise NotImplementedError
        
        try:
            audio_flag = 1
            audio = load_and_transform_audio_data([video_path],"cpu",  clips_per_video=8)
            audio = audio.to(self.device)
        except :
            # print('no audio is found')
            audio_flag = 0
        finally:
            if audio_flag == 1:
                # image_emb, _ = self.model.encode_videoQformer_audiovideo(video,audio)
                image_emb, _ = self.model.encode_videoQformer_visual(video)
                audio_emb,_  = self.model.encode_audioQformer(audio)
                img_list.append(image_emb)
                img_list.append(audio_emb)
                conv.system = ""
                # conv.append_message(conv.roles[0], "The audio of this video is <Video><ImageHere></Video> ")
                conv.append_message(conv.roles[0], "Close your eyes, open your ears and you imagine only based on the sound that: <ImageHere>. \
                Close your ears, open your eyes and you see that <Video><ImageHere></Video>.  \
                Now answer my question based on what you have just seen and heard.")

            else:  # only vison no audio
                # conv.system = "You can understand the video that the user provides. Follow the instructions carefully and explain your answers in detail."
                image_emb, _ = self.model.encode_videoQformer_visual(video)
                img_list.append(image_emb)
                conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
            # return "Received."
            return img_list

    def upload_video_without_audio(self, video_path, conv, img_list):
        msg = ""
        if isinstance(video_path, str):  # is a video path
            ext = os.path.splitext(video_path)[-1].lower()
            # print(video_path)
            # image = self.vis_processor(image).unsqueeze(0).to(self.device)
            video, msg = load_video(
                video_path=video_path,
                n_frms=8,
                height=224,
                width=224,
                sampling ="uniform", return_msg = True
            )
            video = self.vis_processor.transform(video)
            video = video.unsqueeze(0).to(self.device)
            # print(image)
        else:
            raise NotImplementedError
        
        
        # conv.system = "You can understand the video that the user provides.  Follow the instructions carefully and explain your answers in detail."
        image_emb, _ = self.model.encode_videoQformer_visual(video)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Video><ImageHere></Video> "+ msg)
        return "Received."

    def upload_img(self, image, conv, img_list):

        msg = ""
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB') # 增加一个时间维度
            image = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.image_vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        else:
            raise NotImplementedError

        image_emb, _ = self.model.encode_videoQformer_visual(image)
        img_list.append(image_emb)
        # Todo msg=""
        conv.append_message(conv.roles[0], "<Image><ImageHere></Image> "+ msg)

        # return "Received."
        return img_list

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs



print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


data_path = "./data/VFE_val.json"
video_path = '/data/video_clips'

pre_data = []
raw_lines = load_json(data_path)
# print(len(raw_lines))

for line in tqdm(raw_lines):
    data = {}
    vid_name = line['video']
    ques = line['QA'][0]['q']
    timestamps = line['timestamps']
    vid_path = os.path.join(video_path,vid_name[:-4]+str(timestamps)+".mp4")


    chat_state = Conversation(
                system= "",
                roles=("Human", "Assistant"),
                messages=[],
                offset=0,
                sep_style=SeparatorStyle.SINGLE,
                sep="###",
            )


    imgs = []
    img_list = chat.upload_video(vid_path,chat_state,imgs)
    # llm_message = chat.upload_video_without_audio(video_path, chat_state, img_list)

    chat_state.messages.append([chat_state.roles[0], f"<Video><VideoHere></Video> \n"])
        
    # chat.ask(ques,chat)
    chat.ask(ques, chat_state)


    message = chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=2,
                                temperature=0.9,
                                max_new_tokens=300,
                                max_length=2000)[0]
    # print(message)
    data[vid_name]=message
    data['timestamps']=timestamps
    pre_data.append(data)
    # print(pre_data)
save_json(pre_data,'output.json')
