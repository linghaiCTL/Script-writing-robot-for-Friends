# Description: Inference script for llama model
# if cmd is true, the script will load user input from the command line, otherwise it will load from the test data path
# before running this script, make sure you have the model and the test data path
python train_llama.py \
    --mode inference \
    --cmd true \
    --version 2 \
    --test_data_path data/filt/test.json \
    --model_path model/llama \
    --planner_lora_path checkpoints/SWM2.0/planner \
    --writer_lora_path checkpoints/SWM2.0/writer 