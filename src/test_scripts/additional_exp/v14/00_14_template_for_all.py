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

# # Add parameters
#parser.add_argument('--cuda', type=str, required=True, help='Specify the CUDA device (e.g., cuda:0)')
parser.add_argument('--po', type=int, required=True, help='Path to save the output annotations')
parser.add_argument('--d', type=int, required=True, help='from 0 to 4')

# Parse the arguments
args = parser.parse_args()

# Use the parsed arguments
datasets = ["agiqa","csiq","livew","kadid","spaq"]
datasets_files = ["../agiqa.json","../CSIQ.json","../livew.json","../KADID_test.json","../SPAQ_TEST.json"]
datasets_files=[os.path.join("../",dataset_f) for dataset_f in datasets_files]
datasets_folders = ["/mnt/data3/datasets/ar_datasets/agiqa", "/mnt/data3/datasets/ar_datasets/","/mnt/data3/datasets/ar_datasets/LIVE-w/ChallengeDB_release/Images","/mnt/data3/datasets/ar_datasets/KADID10K/images","/mnt/data3/datasets/ar_datasets/SPAQ/TestImage"]
datasets_gt_names =["score","score","score","score","score"]
piece_order = int(args.po)
dataseet_order = int(args.d)

test_dataset = datasets[dataseet_order]
test_file = datasets_files[dataseet_order]
test_folder = datasets_folders[dataseet_order]
gt_name = datasets_gt_names[dataseet_order]
text_save_folder = f"{test_dataset}_t1"
os.makedirs(text_save_folder,exist_ok=True)
save_path = os.path.join("./",text_save_folder,f"{piece_order}.json")

device = "cuda:"+str(piece_order)

def load_text_data(data_file:str):
    data_file = open(data_file,"r")
    sample_list  = json.load(data_file)

    return sample_list


datas = load_text_data(test_file)

piece_lens = len(datas)//8

test_range_start = int(piece_order)*piece_lens
test_range_end = (int(piece_order)+1)*piece_lens
if piece_order ==7:
    test_range_end = None


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
image_folder = test_folder

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
saved_annotation = open(save_path,"w",encoding="UTF-8")
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
for order,item in enumerate(tqdm(datas[test_range_start:test_range_end])):
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


        
        caption_lists.append(reasoning_match_score_3)
        anotation = {
            "image": item["image"],
            "gt_score":item[datasets_gt_names[dataseet_order]], 
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
