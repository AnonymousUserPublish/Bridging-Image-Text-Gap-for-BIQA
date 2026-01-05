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

parser = argparse.ArgumentParser(description="Text Evaluation Script")
parser.add_argument('--po', type=int, required=True, help='GPU rank')
parser.add_argument('--v', type=int, required=True, help='version order from 0-3')
parser.add_argument('--d', type=int, required=True, help='dataset from 0-5')
args = parser.parse_args()


version = int(args.v)
dataset_order = int(args.d)
piece_order = int(args.po)
model_paths = ["v11_baseline_fullmodel_entire","v12_cot","v13_self_consistency","v14_autoencoder"]
#dataset_folder = ["agiqa","csiq","livew","kadid","koniq","spaq"]
dataset_folder = ["kadid"]
data_file = "merged.json"

input_item = "perception_text"
evaluate_item ="gt_score"
removed_tokens = ["good","moderate","average","poor","decent"]

def remove_processing(prompt:str,removed_tokens:list=removed_tokens):
    for token in removed_tokens:
        prompt = prompt.replace(token,"")
    return prompt

data_file_path = os.path.join("./",f"v{version+11}",f"{dataset_folder[dataset_order]}",data_file)
save_file_path = os.path.join("./",f"v{version+11}",f"txt_{dataset_folder[dataset_order]}",f"p{piece_order}.json")
SUBFOLDER = model_paths[version]
device = "cuda:"+str(piece_order)
datas = json.load(open(data_file_path,"r"))
piece_lens = len(datas)//8
test_range_start = int(piece_order)*piece_lens
test_range_end = (int(piece_order)+1)*piece_lens
if piece_order ==7:
    test_range_end = None

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
MODEL_PATH = "..."

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
    subfolder=SUBFOLDER
)

tokenizer = AutoTokenizer.from_pretrained((MODEL_PATH+SUBFOLDER))

processor = AutoProcessor.from_pretrained(MODEL_PATH+SUBFOLDER)


PERCEPTION_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "follows a human thinking logics."
    "First describes the low-level visual contents of the image, mainly focus on three aspects including the main subject of this image (why human takes this picture), the advantage of this image (what makes this image looks good) and the flaws (what makes the image looks bad). Second  thinks about the reasoning process in the mind and then provides the user with the answer. The description, reasoning"
    " and answer are enclosed within <caption></caption> <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<caption>description here</caption><think> reasoning process here </think><answer> answer here </answer>."
)

REASONING_SYSTEM_PRMOPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)



PERCEPTION_QUESTION_PROMPT = 'Please give a detailed, objective description that is sufficient to judge overall image quality. Describe only what you can visually observe in this image. Focus exclusively on the concrete visual elements you see - objects, people, colors, shapes, text, layout, and spatial relationships. '

REASONING_QUESTION_PRMOPT = 'If there is an image described as image_place_holder, what is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. The final answer in JSON format with the following keys: "rating": The score.'

PERCEPTION_QUESTION_PROMPT  = 'What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. The final answer in JSON format with the following keys: "rating": The score.'

#text_inference = "./2stage_prior_text_wo2_test_p4r4_ckpt1000.json"
saved_annotation = open(save_file_path,"w",encoding="UTF-8")
saved_annotation.write("[\n")



gt_lists = []
perception_lists = []
reasoning_lists = []
caption_lists = []
description_pattern = re.compile(r"<description>(.*?)</description>", re.DOTALL)

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
for order,item in enumerate(tqdm(datas[test_range_start:test_range_end])):
    with torch.no_grad():
        caption_description = item[input_item].split("<think>")
        if len(caption_description)>1:
            caption_description = remove_processing(caption_description[0])
            #generate the predictions
            caption_input = REASONING_QUESTION_PRMOPT.replace("image_place_holder", caption_description)
            message_2 = [
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
            text = [processor.apply_chat_template(message_2, tokenize=False, add_generation_prompt=True)]
            inputs = processor(
                text=text,
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
            #print(output_text)
            reasoning_match_score_b = re.search(score_pattern, output_text,re.DOTALL)
            if reasoning_match_score_b:
                reasoning_match_score = float(reasoning_match_score_b.group(1))
          
        reasoning_lists.append(reasoning_match_score)

        anotation = {
            "image": item["image"],
            "gt_score":float(item[evaluate_item]),
            "caption_score":reasoning_match_score,
            "reasoning_text":output_text
        }
        saved_annotation.write(json.dumps(anotation, indent=4,ensure_ascii=False))
        if order<piece_lens-1:
            saved_annotation.write(",\n")
        else:
            saved_annotation.write("\n")

saved_annotation.write("]")
