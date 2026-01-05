import os
import json


model_types = ["v11","v12","v13","v14"]
datasets = ["koniq","spaq","kadid","agiqa","csiq","livew"]
datasets = ["txt_"+d for d in datasets]
for model_type in model_types:
    results_folders = [os.path.join(model_type,dataset) for dataset in datasets]
    if model_type=="v12":
        results_folders.pop(0)
    for result_folder in results_folders:
        print(f"-------------------Evaluating :{model_type} {result_folder}----------")
        #result_folder = "./v13_2/koniq"
        result_filepaths = os.listdir(result_folder)
        result_filepaths = [file_path for file_path in result_filepaths if file_path.endswith("fixed.json")]
        result_files = [os.path.join(result_folder,result_file) for result_file in result_filepaths]

        gt_list=[]
        caption_list=[]


        for result in result_files:
            print(result)
            current_list = json.load(open(result,"r"))
            for item in current_list:
                gt_list.append(float(item[ "gt_score"]))

                if "caption_score" in item.keys():
                    caption_list.append(float(item["caption_score"]))
                else:
                    caption_list.append(float(item["reasoning_score"]))

                #reasoning_list.append(float(item["reasoning_score"]))

        from scipy.stats import pearsonr, spearmanr

        caption_pearson_corr, pearson_p_value = pearsonr(caption_list, gt_list)
        print(f"Overall Pearson correlation : {caption_pearson_corr}, p-value: {pearson_p_value}")
        caption_spearman_corr, spearman_p_value = spearmanr(caption_list, gt_list)
        print(f"Overall Pearson correlation : {caption_spearman_corr}, p-value: {spearman_p_value}")
        print(f"-----------------Finished--------------")
# reasoning_pearson_corr, pearson_p_value = pearsonr(reasoning_list, gt_list)
# print(f"Overall Pearson correlation : {reasoning_pearson_corr}, p-value: {pearson_p_value}")
# reasoning_spearman_corr, spearman_p_value = spearmanr(reasoning_list, gt_list)
# print(f"Overall Pearson correlation : {reasoning_spearman_corr}, p-value: {spearman_p_value}")