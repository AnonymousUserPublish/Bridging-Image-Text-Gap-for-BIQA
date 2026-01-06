cd src/open-r1-multimodal 
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install qwen_vl_utils 
pip install flash-attn --no-build-isolation
pip install transformers==4.51.3
