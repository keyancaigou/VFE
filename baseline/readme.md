
## Installation

### Environment Preparation

First, create a conda environment:

```bash
conda env create -f environment.yml
conda activate chat
```

### Prerequisites

Before using the repository, ensure you have obtained the following checkpoints:

#### Pre-trained Language Decoder

1. **LLaMA Weights:**
   - Obtain the original LLaMA weights in the Hugging Face format by following the instructions [here](https://huggingface.co/docs/transformers/model_doc/llama).

2. **Vicuna Delta Weights:**
   - Download Vicuna delta weights for the 7B model [here](#).  
     **Note:** We use v0 weights instead of v1.1 weights.
   
   - Use the following command to add delta weights to the original LLaMA weights to obtain the Vicuna weights:

     ```bash
     python apply_delta.py \
         --base ckpt/LLaMA/7B_hf \
         --target ckpt/Vicuna/7B \
         --delta ckpt/Vicuna/vicuna-7b-delta-v0
     ```

#### Pre-trained Visual Encoder for baseline

- Download the MiniGPT-4 model (trained linear layer) from [this link](https://github.com/Vision-CAIR/MiniGPT-4).

#### Download Pretrained Weights

- Download pretrained weights to run baseline with Vicuna-7B as the language decoder locally from [this link](#).

## How to Run Demo Locally

1. **Configuration:**
   - Set the `llama_model`, `llama_proj_model`, and `ckpt` in `eval_configs/baseline.yaml`.

2. **Run the Inference Script:**

   ```bash
   python inference.py \
       --cfg-path eval_configs/baseline.yaml \
       --gpu-id 0 \
       --num-beams 1 \
       --temperature 1.0 \
       --text-query "What is he doing?" \
       --video-path src/examples/Cooking_cake.mp4 \
       --fragment-video-path src/video_fragment/output.mp4 \
       --cur-min 1 \
       --cur-sec 1 \
       --middle-video 1
   ```

   **Note:** If you want to use the global mode (understanding and question-answering for the whole video), remember to change `middle-video` to `0`.

## Acknowledgement

We are grateful for the following awesome projects that our baseline arises from:

- [Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding](https://github.com/example/Video-LLaMA)
- [Token Merging: Your ViT but Faster](https://github.com/example/Token-Merging)
- [XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model](https://github.com/example/XMem)
- [MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models](https://github.com/example/MiniGPT-4)
- [FastChat: An Open Platform for Training, Serving, and Evaluating Large Language Model-based Chatbots](https://github.com/example/FastChat)
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://github.com/example/BLIP-2)
- [EVA-CLIP: Improved Training Techniques for CLIP at Scale](https://github.com/example/EVA-CLIP)
- [LLaMA: Open and Efficient Foundation Language Models](https://github.com/example/LLaMA)
- [VideoChat: Chat-Centric Video Understanding](https://github.com/example/VideoChat)
- [LLaVA: Large Language and Vision Assistant](https://github.com/example/LLaVA)

## 🔒 Terms of Use

Our baseline is a research preview intended for **non-commercial use only**. You **must NOT** use baseline for any illegal, harmful, violent, racist, or sexual purposes. You are strictly prohibited from engaging in any activity that may potentially violate these guidelines.

---

```
