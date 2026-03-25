# Bridging Image-Text Gap for BIQA



## Introduction
Textual reasoning has recently been widely adopted in Blind Image Quality Assessment (BIQA). However, it remains unclear how textual information contributes to quality prediction and to what extent text can represent the score-related image contents. This work addresses these questions from an information-flow perspective by comparing existing BIQA models with three paradigms designed to learn the image-text-score relationship: Chain-of-Thought, Self-Consistency, and Autoencoder. Our experiments show that the score prediction performance of the existing model significantly drops when only textual information is used for prediction. Whereas the Chain-of-Thought paradigm introduces little improvement in BIQA performance, the Self-Consistency paradigm significantly reduces the gap between image- and text-conditioned predictions, narrowing the PLCC/SRCC difference to 0.02/0.03. The Autoencoder-like paradigm is less effective in closing the image-text gap, yet it reveals a direction for further optimization. These findings provide insights into how to improve the textual reasoning for BIQA and high-level vision tasks.


##  Dependencies and Installation
```bash
git clone git@github.com:AnonymousUserPublish/Bridging-Image-Text-Gap-for-BIQA.git
bash setup.sh
```

## Data Preparation 
Download meta files from [Data-DeQA-Score](https://huggingface.co/datasets/zhiyuanyou/Data-DeQA-Score/tree/main) and the source images from the [KONIQ](https://database.mmsp-kn.de/koniq-10k-database.html) dataset.



## Training

```
cd src/open-r1-multimodal/
bash several_training.sh
```
```
v11 -> baseline, v12 -> CoT, v13 -> Self-Consistency, v14 -> Auto-encoder
```


## Testing
```
cd src/test_scripts/
check the differnt scripts for evaluating the sevearl tasks
```
## Model Weights
Backbone: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

## File Path
```
PLease change the 1.training config file 2.training dataset folder path 3.default weights path to your own directory.
```

##  To Do List
- [x] Release training and inference code
- [x] Release the paper

## Acknowledgement
This work and repo is built based on [Q-Insight](https://github.com/bytedance/Q-Insight).
We appreciate the releasing codes and data of [Q-Insight](https://github.com/bytedance/Q-Insight),[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [VLM-R1](https://github.com/om-ai-lab/VLM-R1),  and [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score).



## Citation


If you find the code helpful in your research or work, please cite the following papers:
```
@misc{li2026understandingpuretextualreasoning,
      title={Understanding Pure Textual Reasoning for Blind Image Quality Assessment}, 
      author={Yuan Li and Shin'ya Nishida},
      year={2026},
      eprint={2601.02441},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.02441}, 
}
```








