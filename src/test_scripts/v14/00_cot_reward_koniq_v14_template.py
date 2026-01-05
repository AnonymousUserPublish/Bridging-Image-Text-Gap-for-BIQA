import os
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
#1. load the dataset
import argparse
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
# def custom_forward(
#         self,
#         hidden_states: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#         rotary_pos_emb: Optional[torch.Tensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     ) -> torch.Tensor:
#         seq_length = hidden_states.shape[0]
#         q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
#             cos = emb.cos().float()
#             sin = emb.sin().float()
#         else:
#             cos, sin = position_embeddings
#             # Add this
#             cos = cos.to(torch.float)
#             sin = sin.to(torch.float)
#         q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
#         q = q.squeeze(0)
#         k = k.squeeze(0)

#         max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
#         attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
#             seq_length, -1
#         )
#         attn_output = self.proj(attn_output)
#         return attn_output

# Qwen2_5_VLVisionFlashAttention2.forward = custom_forward
parser = argparse.ArgumentParser(description="Text Evaluation Script")

# # Add parameters
#parser.add_argument('--cuda', type=str, required=True, help='Specify the CUDA device (e.g., cuda:0)')
parser.add_argument('--po', type=int, required=True, help='Path to save the output annotations')
# parser.add_argument('--st', type=int, required=True, help='where to where')
# parser.add_argument('--end', type=int, required=True, help='where to where')
# parser.add_argument('--flag', type=bool, required=True, help='where to where')

# Parse the arguments
args = parser.parse_args()

# Use the parsed arguments
piece_order = int(args.po)
save_template = "./koniq_t1/v14_po.json"
text_inference = save_template.replace("po","p"+str(piece_order))
device = "cuda:"+str(piece_order)
piece_lens = 375
test_range_start = int(piece_order)*piece_lens
test_range_end = (int(piece_order)+1)*piece_lens
if piece_order ==7:
    test_range_end = None
#device = "cuda:0"
# start_point = args.st
# flag = False
# if flag:
#     end_point =None
# else:
#     end_point = args.end
# print(f"start and end point {start_point,end_point}")
#answer_tag_pattern = r'<answer>(.*?)</answer>'
# score_pattern = r'\"rating\"\s*:\s*([\d\.]+)'

def load_text_data(data_file:str="/home/ly/projects/14_q-insight/Q-Insight/model_anotation/koniq_normalized_test_set.json"):
    data_file = open(data_file,"r")
    sample_list  = json.load(data_file)

    return sample_list

dataset = load_text_data()


#2. test the text inference ability of the model
#2.1 load the model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch
#device = "cuda:0"
seed = 42
set_seed(seed)
# MODEL_PATH = "/mnt/data1/model_tensors"
# SUBFOLDER = "Qwen2.5-VL-7B-score"
# lora_model_path = "/mnt/data1/model_tensors/Koniq_2stage_7B_lora_normalize/checkpoint-1000"
MODEL_PATH = "/mnt/data3/trained_parameters/rl_markov/"
SUBFOLDER = "v14_autoencoder"
#lora_model_path = "/mnt/data3/trained_parameters/rl_markov/4_Koniq_cot_v9_lora"
image_folder = "/mnt/data3/datasets/ar_datasets/koniq-10k/512x384"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
    subfolder=SUBFOLDER
)

tokenizer = AutoTokenizer.from_pretrained((MODEL_PATH+SUBFOLDER))
'''
vision_attrs = ("visual", "vision_model", "vision_tower")
for attr in vision_attrs:
    if hasattr(model, attr):
        module = getattr(model, attr)
        print(module)
        for p in module.parameters():
            p.requires_grad_(False)

'''


#processor = AutoProcessor.from_pretrained(lora_model_path)
processor = AutoProcessor.from_pretrained(MODEL_PATH+SUBFOLDER)


PERCEPTION_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "follows a human thinking logics."
)

REASONING_SYSTEM_PRMOPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)



#PERCEPTION_QUESTION_PROMPT = 'Please give a detailed, objective description that is sufficient to judge overall image quality. Describe only what you can visually observe in this image. Focus exclusively on the concrete visual elements you see - objects, people, colors, shapes, text, layout, and spatial relationships. '# Do not include any interpretation, analysis, reasoning, assumptions, or inferences about meaning, purpose, context, or implications. Simply provide a direct, objective inventory of the visible contents as if you were creating a detailed visual catalog '

REASONING_QUESTION_PRMOPT = 'If there is an image described as image_place_holder, what is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. The final answer in JSON format with the following keys: "rating": The score.'

PERCEPTION_QUESTION_PROMPT  = ' With 1 representing very poor quality and 5 representing excellent quality, please explain why this image is rated as some score. Do not mention this rating score in your answer.'
#text_inference = "./2stage_prior_text_wo2_test_p4r4_ckpt1000.json"
saved_annotation = open(text_inference,"w",encoding="UTF-8")
saved_annotation.write("[\n")



gt_lists = []
perception_lists = []
reasoning_lists = []
caption_lists = []
#answer_tag_pattern = r'<answer>(.*?)</answer>'
description_pattern = re.compile(r"<description>(.*?)</description>", re.DOTALL)
# think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# score_pattern = re.compile(r'\"rating\"\s*:\s*([\d\.]+)',re.DOTALL)
#description_pattern = r"<description>(.*?)</description>"
think_pattern = r"<think>(.*?)</think>"
answer_pattern = r"<answer>(.*?)</answer>"
score_pattern = r'\"rating\"\s*:\s*([\d\.]+)'


gen_config = GenerationConfig(
do_sample=True, 
temperature=1.0,
top_k=50,     
top_p=0.95,
max_new_tokens=1024,
)

from tqdm import tqdm
for order,item in enumerate(tqdm(dataset[test_range_start:test_range_end])):
    with torch.no_grad():
        image_path = os.path.join(image_folder,item["image"])
        message_1 = [
            {"role": "system", "content": [{"type": "text", "text": PERCEPTION_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text":PERCEPTION_QUESTION_PROMPT 
                    },
                    {"type": "image", "image": f"file://{image_path}"}
                ]
            }
        ]
        #generate the description


        text = [processor.apply_chat_template(message_1, tokenize=False, add_generation_prompt=True)]
        image_inputs, video_inputs = process_vision_info([message_1])
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)


        generated_ids = model.generate(
        **inputs,
        generation_config=gen_config,
        use_cache=True,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        record_output_1 = output_text
        #print(output_text)

        caption_description = output_text

          
        caption_input = REASONING_QUESTION_PRMOPT.replace("image_place_holder", caption_description)
        message_3 = [
        {"role": "system", "content": [{"type": "text", "text": REASONING_SYSTEM_PRMOPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":caption_input
                },
            ]
        }
        ]
        text_3 = [processor.apply_chat_template(message_3, tokenize=False, add_generation_prompt=True)]
        inputs_3 = processor(
            text=text_3,
            padding=True,
            return_tensors="pt",
        )
        inputs_3 = inputs_3.to(device)
        generated_ids_3 = model.generate(
        **inputs_3,
        generation_config=gen_config,
        use_cache=True,
        )
        generated_ids_trimmed_3 = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_3.input_ids, generated_ids_3)
        ]
        output_text_3 = processor.batch_decode(
            generated_ids_trimmed_3, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        #print(output_text)
        reasoning_match_score_b_3 = re.search(score_pattern, output_text_3,re.DOTALL)
        if reasoning_match_score_b_3:
            reasoning_match_score_3 = float(reasoning_match_score_b_3.group(1))


        
        gt_lists.append(item["normalized_score"])
        caption_lists.append(reasoning_match_score_3)
        anotation = {
            "image": item["image"],
            "gt_score":item["normalized_score"],
 
            "caption_score":reasoning_match_score_3,
            "caption_answer_text":output_text_3,
            "perception_text":record_output_1,
        }
        saved_annotation.write(json.dumps(anotation, indent=4,ensure_ascii=False))
        if order<piece_lens-1:
            saved_annotation.write(",\n")
        else:
            saved_annotation.write("\n")

saved_annotation.write("]")
