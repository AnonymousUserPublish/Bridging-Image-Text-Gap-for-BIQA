<div align="center">
<h3>

Bridging Image-Text Gap for BIQA



## Introduction
Text-IQA minimize the PLCC/SRCC performance between image-conditioned and text-conditioned to 0.02/0.03 across  6 general BIQA datasets.


##  Dependencies and Installation
```bash
git clone  this repo
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


##  To Do List
- [] Release weights
- [x] Release training and inference code
- [x] Release the paper

## Acknowledgement
This work and repo is built based on [Q-Insight] (https://github.com/bytedance/Q-Insight).
We appreciate the releasing codes and data of [Q-Insight](https://github.com/bytedance/Q-Insight), [VLM-R1](https://github.com/om-ai-lab/VLM-R1),  and [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score).



## Citation


If you find the code helpful in your research or work, please cite the following papers:
```
@article{li2026,
  title={Bridging Image-Text Gap in BIQA},
  year={2026}
}
```
