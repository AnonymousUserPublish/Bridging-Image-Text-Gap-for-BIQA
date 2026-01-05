import os
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
#1. load the dataset
import argparse
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2,Qwen2_5_VLVisionAttention,apply_rotary_pos_emb_flashatt, flash_attn_varlen_func,repeat_kv,apply_multimodal_rotary_pos_emb
from transformers.cache_utils import Cache,logging
from transformers.modeling_flash_attention_utils import _flash_attention_forward
import torch
from typing import Tuple
from typing import Optional
import math
from object_analysis import get_nested_shape_details

meaningless_tokens = ['.',  ',',':', '<', '>', '{', "}","</",'the', 'rating', 'a', 'answer', 'and', 'is', 'think','assistant','<th',"an","this","caption","'s","as","in","isn","based","on","with","which","of","to","which","-","be","is","are","doesn","about"]


def indices_match(sequences,processor,inputs):
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, sequences)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    matches = list(re.finditer(r"\d+\.\d{2}", output_text))
    if not matches:
        print(output_text)
        print("No decimals with two digits found.")

    # pick the last decimal match
    else:
        last = matches[-1]
        num_str = last.group()
        char_start, char_end = last.span()

        # 4. Convert token IDs to printable pieces

        tokens = tokenizer.convert_ids_to_tokens(generated_ids_trimmed[0])
        decoded_tokens = [tokenizer.convert_tokens_to_string([t]) for t in tokens]

        # 5. Build a list of character offsets for each token

        token_offsets = []
        cursor = 0

        for tok in decoded_tokens:
            start = cursor
            end = start + len(tok)
            token_offsets.append((start, end))
            cursor = end

        # 6. Find token indices overlapping with decimal span

        decimal_token_indices = [
            idx for idx, (s, e) in enumerate(token_offsets)
            if not (e <= char_start or s >= char_end)
        ]
        return decimal_token_indices


def top_k_attentions(attn_weights:torch.Tensor,token_indices:list,prompt_length:int=486,text_prompt_length:int=226,layer_select:int=-1,head_select:int=-1,top_k:int=50,image_count:bool=False):
    selected_attns = [attn_weights[token_index] for token_index in token_indices]
    #remove '.'
    selected_attns.pop(1)
    #select the layer
    selected_attns = [selected_attn[layer_select] for selected_attn in selected_attns]
    #return selected_attns
    values = []
    indices = []
    for selected_attn in selected_attns:
        if head_select==-1:
            averaged_attn_map = selected_attn.mean(dim=1)
            final_attn_map = averaged_attn_map
        else:
            final_attn_map = selected_attn[:,head_select,:,:]
        weights_vector = final_attn_map.squeeze()
        # only generated tokens
        if not image_count:
            weights_vector = weights_vector[prompt_length:]
        else:
            weights_vector = weights_vector[text_prompt_length:]
        top_k_results = torch.topk(weights_vector, k=top_k, dim=0)
        top_k_values = top_k_results.values
        top_k_indices_list = top_k_results.indices.tolist() # Convert indices to a standard Python list
        values.append(top_k_values)
        indices.append(top_k_indices_list)
    return values,indices

def valid_format(item:str):
    item = item.replace(" ","")
    item = item.replace("\n","")
    item = item.replace('"',"")
    item= item.lower()
    return item


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
save_template = "./koniq_v11/v11_ana_test.json"
text_inference = save_template.replace("po","p"+str(piece_order))
device = "cuda:"+str(piece_order)
piece_lens = 3 #375
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

def load_text_data(data_file:str="src/open-r1-multimodal/data_config/koniq_normalized_test_set.json"):
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
MODEL_PATH = ".."
SUBFOLDER = "v11_baseline_fullmodel_entire"
#lora_model_path = "/mnt/data3/trained_parameters/rl_markov/4_Koniq_cot_v9_lora"
image_folder = "../koniq-10k/512x384"

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
saved_annotation = open(text_inference,"w",encoding="UTF-8")




gt_lists = []
perception_lists = []
reasoning_lists = []
caption_lists = []
#answer_tag_pattern = r'<answer>(.*?)</answer>'
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

image_attn_dic = [{},{},{}]
image_text_attn_dic = [{},{},{}]
text_attn_dic = [{},{},{}]



from tqdm import tqdm
image_token_num = 0
total_prompt_token_num = 0
image_attn_dic[0]["image_token"]=0.0
image_attn_dic[1]["image_token"]=0.0
image_attn_dic[2]["image_token"]=0.0
for order,item in enumerate(tqdm(dataset)):
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
        output_attentions=True,
        return_dict_in_generate=True,
        use_cache=True,
        )
        seq,attn,past_key = generated_ids.values()
        full_seq = processor.batch_decode(inputs.input_ids[0],skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        txt_prompt_length = full_seq.index(" score")+2
        prompt_length = len(inputs.input_ids[0])

        decimal_token_indices = indices_match(sequences=seq,processor=processor,inputs=inputs)
        #FIND THE INDEXS
        ''' WITHOUT IMAGE ATTENTIONS'''
        vals,indices = top_k_attentions(attn_weights=attn,prompt_length=prompt_length,token_indices=decimal_token_indices,text_prompt_length=txt_prompt_length,image_count=False)
        for num,(val,ind) in enumerate(zip(vals,indices)):
            top_k_tokens = [seq[0][index+prompt_length] for index in ind]
            top_text = processor.batch_decode(
            top_k_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for order,item in enumerate(top_text):
                item = valid_format(item)
                if item in meaningless_tokens or not item or item.isspace():
                    continue
                else:
                    if item in image_text_attn_dic[num].keys():
                        image_text_attn_dic[num][item]+=val[order].item()
                    else:
                        image_text_attn_dic[num][item]=val[order].item()
                        


        ''' WITH IMAGE ATTENTIONS'''
        vals,indices = top_k_attentions(attn_weights=attn,prompt_length=prompt_length,token_indices=decimal_token_indices,text_prompt_length=txt_prompt_length,image_count=True)
        for num,(val,ind) in enumerate(zip(vals,indices)):
            total_prompt_token_num+=len(ind)

            top_k_tokens = [seq[0][index+txt_prompt_length] for index in ind]
            #print(top_k_tokens)
            top_text = processor.batch_decode(
            top_k_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for order,item in enumerate(top_text):
                item = valid_format(item)
                if ind[order]+txt_prompt_length<prompt_length:
                    image_attn_dic[num]["image_token"]+=val[order].item()
                    image_token_num+=1
                elif item in meaningless_tokens or not item or item.isspace():
                    total_prompt_token_num-=1
                    continue
                else:
                    if item in image_attn_dic[num].keys():
                        image_attn_dic[num][item]+=val[order].item()
                    else:
                        image_attn_dic[num][item]=val[order].item()
            #print(total_count)
            #print(f"image token ratio {image_token/(total_count+image_token)}")
            #print("with images top_tokens",top_text)
            

        ''' START FROM HERE'''
        stage_1 = processor.batch_decode(seq[:,prompt_length:],skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        caption_description = stage_1.split("<think>")
        #print(caption_description)
        # reasoning_match_score = -1.0
        if len(caption_description)>1:
            caption_description = caption_description[0]
            #generate the predictions
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
            output_attentions=True,
            return_dict_in_generate=True,
            use_cache=True,
            )
            seq,attn,past_key = generated_ids_3.values()
            full_seq = processor.batch_decode(inputs_3.input_ids[0],skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            txt_prompt_length = full_seq.index(" score")+2
            prompt_length = len(inputs_3.input_ids[0])

            decimal_token_indices = indices_match(sequences=seq,processor=processor,inputs=inputs_3)
            #FIND THE INDEXS
            ''' WITHOUT IMAGE ATTENTIONS'''
            vals,indices = top_k_attentions(attn_weights=attn,prompt_length=prompt_length,token_indices=decimal_token_indices,text_prompt_length=txt_prompt_length,image_count=False)
            for num,(val,ind) in enumerate(zip(vals,indices)):
                top_k_tokens = [seq[0][index+prompt_length] for index in ind]
                #print(top_k_tokens)
                top_text = processor.batch_decode(
                top_k_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for order,item in enumerate(top_text):
                    item = valid_format(item)
                    if item in meaningless_tokens or not item or item.isspace():
                        continue
                    else:
                        if item in text_attn_dic[num].keys():
                            text_attn_dic[num][item]+=val[order].item()
                        else:
                            text_attn_dic[num][item]=val[order].item()



prepared_data = {
    "image_attn":image_attn_dic,
    "image_text_attn": image_text_attn_dic,
    "text_attn":text_attn_dic,
    "image_token_num": image_token_num,
    "total_prompt_token_num": total_prompt_token_num
}
json.dump(prepared_data,saved_annotation,indent=4)





        