from modelscope import snapshot_download
save_dir='model/llama'
model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct', local_dir=save_dir)