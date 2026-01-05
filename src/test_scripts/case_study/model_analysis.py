import os
import json
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

test_type = "txt"
model_version = '13'
file_path = f"./koniq_v{model_version}/v{model_version}_ana.json"
save_path_json = f"v{model_version}_{test_type}.json"
save_path = open(save_path_json,"w")
data = json.load(open(file_path,"r"))
# dict_keys(['image_attn', 'image_text_attn', 'text_attn', 'image_token_num', 'total_prompt_token_num'])

def softmax(x):
    """Numerically stable softmax implementation."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

from collections import Counter

def cumulative_merge_counter(dict1: dict, dict2: dict) -> dict:
    """Merges two dictionaries, adding values for common keys using Counter."""
    
    # Counter automatically handles non-existent keys (treating them as 0).
    # It converts the dictionary to a Counter object.
    counter1 = Counter(dict1)
    counter2 = Counter(dict2)
    
    # The addition operator (+) performs a cumulative merge of the two Counter objects.
    merged_counter = counter1 + counter2
    
    # Convert the resulting Counter back into a standard dictionary
    return dict(merged_counter)

def plot_top_k_figure(data_dic:dict,top_k:int=20,figure_path:str="softmax_ascending.png"):
    df = pd.DataFrame(data_dic.items(), columns=['Category', 'Score'])
    df_top_k = df.head(top_k).reset_index(drop=True)
    # 4. Apply Softmax
    scores = df_top_k['Score'].to_numpy()
    softmax_probabilities = softmax(scores)
    # 5. Recombine and Plot
    df_top_k['Softmax_Prob'] = softmax_probabilities
    df_asc = df_top_k.sort_values(by='Softmax_Prob', ascending=True)
    # Plotting the horizontal bar chart.
    plt.figure(figsize=(10, 8)) 
    plt.barh(df_asc['Category'], df_asc['Softmax_Prob'], color='darkgreen')
    plt.title(f'Top {top_k} Softmax Probabilities (Ascending Order)', fontsize=14)
    plt.xlabel('Softmax Probability', fontsize=12)
    plt.ylabel('Token/Category', fontsize=12)
    # ... (labeling and saving the plot)
    plt.tight_layout()
    plt.savefig(figure_path)




remove_meaningless_tokens=[ 'it', ').',  '-l', 'or', 'out', 'ex', 'those','its', '),',  'i', '**' , '(', "'t",  'for', 'there',  '-and', 'that', 'could', 'by', "'.", '’s',  'y', 'ia', 'sep', 'has',  ')',  'would', '’t',  'will',  "'ll",  'we', 'b','ab','bo', 'at',  'c', 'ed',  "'ve",  'itself',  '**:', 'ye',  'am',  ';', '-com',  'en', 'es', 'does', '—a',  'el',  't',  "',",  '-s', "'",  '.,', '/no', '/logo', '€', 'm', 'these', 'nearby', 'includes', '“', 'ly', 'seem',  'des', 'pl',  'd', '/gr',  'h',').','ny',')',"'—the'",'.a','!','/','—','.d','over','some','used','photo','image','but','however','john','another','iating','seems','likely','from','enta','popcorn','fully','assess','>.','ink','quality','score','shows','should','while',"description","described","provided","let","convey","overall",'not','subject',"assessment","picture",'rate','appears',"my","around","scale","rated",'given',"ness",'0','1','2','3','4','5','6','7','8','9']
merged_tokens = ["bl","urr","iness"]

image_token_ratio = data['image_token_num']/data['total_prompt_token_num']

image_attns = data['image_attn']
image_text_attns = data['text_attn']
text_attns = data['text_attn']
if test_type == "image": 
    image_attn_merge = image_attns[0]#image_attns[0]#cumulative_merge_counter(image_attns[0],image_attns[1])
else:
    image_attn_merge = text_attns[0]
static_keys =  list(image_attn_merge.keys())
image_attn_merge["blurriness"] = 0.0
for key in static_keys:
    if key in merged_tokens:
        if "blurry" in static_keys:
            image_attn_merge["blurry"]+= image_attn_merge[key]
        else:
            image_attn_merge["blurry"] = image_attn_merge[key]
        image_attn_merge.pop(key)
    elif key in  remove_meaningless_tokens:
        image_attn_merge.pop(key)
if test_type == "image":
    print( image_attn_merge["image_token"])
# 1. Convert the dictionary to a DataFrame and sort
TOP_K = 15
df = pd.DataFrame(image_attn_merge.items(), columns=['Category', 'Score'])
df_sorted_asc = df.sort_values(by='Score', ascending=False)
df_top_k = df_sorted_asc.head(TOP_K).reset_index(drop=True)
if test_type == "image":
    if "image_token" not in list(df_top_k['Category']):
        image_t_value= pd.DataFrame({'Category':["image_token"], 'Score':[float( image_attn_merge["image_token"])]})
        df_top_k = pd.concat([df_top_k, image_t_value], ignore_index=True)
# 4. Apply Softmax
scores = df_top_k['Score'].to_numpy()/3015
#normalize
mean_score = np.mean(scores)
max_score = np.max(scores)
scores = (scores-mean_score)/max_score

softmax_probabilities = softmax(scores)

# 5. Recombine and Plot
df_top_k['Softmax_Prob'] = softmax_probabilities

df_asc = df_top_k.sort_values(by='Softmax_Prob', ascending=False)


average_score = list(df_asc['Score'])
average_score = [i/3015 for i in average_score]
data_to_save = {
    "tokens":list(df_asc['Category']),
    "original_scores":average_score,
    "softmax":list(df_asc['Softmax_Prob'])
}

json.dump(data_to_save,save_path,indent=4)

print(image_token_ratio)


# Plotting the horizontal bar chart.
# plt.figure(figsize=(10, 8)) 
# plt.barh(df_asc['Category'], df_asc['Softmax_Prob'], color='darkgreen')

# plt.title(f'Top {TOP_K} Softmax Probabilities (Ascending Order)', fontsize=14)
# plt.xlabel('Softmax Probability', fontsize=12)
# plt.ylabel('Token/Category', fontsize=12)

# # ... (labeling and saving the plot)
# plt.tight_layout()
# plt.savefig(f'top_{TOP_K}_softmax_ascending_plot.png')
# # Plotting the horizontal bar chart
# plt.figure(figsize=(10, 8)) 
# plt.barh(df_top_k['Category'][::-1], df_top_k['Softmax_Prob'][::-1], color='coral')

# plt.title(f'Top {TOP_K} Scores after Softmax Normalization', fontsize=14)
# plt.xlabel('Softmax Probability', fontsize=12)
# plt.ylabel('Token/Category', fontsize=12)

# # ... (labeling and saving the plot)
# plt.tight_layout()
# plt.savefig('top_30_softmax_plot.png')
    #print(len(softmax_img_attn.keys()))
