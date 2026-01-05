import os
import json


model_type = "./v14_2"
datasets = ["koniq","spaq","kadid","agiqa","csiq","livew"]
results_folders = [os.path.join(model_type,dataset) for dataset in datasets]
for result_folder in results_folders:
    print(f"-------------------Evaluating : {result_folder}----------")
    #result_folder = "./v13_2/koniq"
    result_filepaths = os.listdir(result_folder)
    result_files = [os.path.join(result_folder,result_file) for result_file in result_filepaths]

    gt_list=[]
    oveall_list=[]
    caption_list=[]
    reasoning_list = []


    for result in result_files:
        print(result)
        current_list = json.load(open(result,"r"))
        for item in current_list:
            gt_list.append(float(item[ "gt_score"]))
            oveall_list.append(float(item["overall_score"]))
            if "caption_score" in item.keys():
                caption_list.append(float(item["caption_score"]))
            else:
                caption_list.append(float(item["reasoning_score"]))

            #reasoning_list.append(float(item["reasoning_score"]))

    from scipy.stats import pearsonr, spearmanr
    overall_pearson_corr, pearson_p_value = pearsonr(oveall_list, gt_list)
    print(f"Overall Pearson correlation : {overall_pearson_corr}, p-value: {pearson_p_value}")
    overall_spearman_corr, spearman_p_value = spearmanr(oveall_list, gt_list)
    print(f"Overall Pearson correlation : {overall_spearman_corr}, p-value: {spearman_p_value}")

    caption_pearson_corr, pearson_p_value = pearsonr(caption_list, gt_list)
    print(f"Overall Pearson correlation : {caption_pearson_corr}, p-value: {pearson_p_value}")
    caption_spearman_corr, spearman_p_value = spearmanr(caption_list, gt_list)
    print(f"Overall Pearson correlation : {caption_spearman_corr}, p-value: {spearman_p_value}")
    print(f"-----------------Finished--------------")
# reasoning_pearson_corr, pearson_p_value = pearsonr(reasoning_list, gt_list)
# print(f"Overall Pearson correlation : {reasoning_pearson_corr}, p-value: {pearson_p_value}")
# reasoning_spearman_corr, spearman_p_value = spearmanr(reasoning_list, gt_list)
# print(f"Overall Pearson correlation : {reasoning_spearman_corr}, p-value: {spearman_p_value}")