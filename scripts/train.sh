# Description: Inference script for llama model
# task set to imaging, would training the planner
# task set to generating, would training the writer
# before running this script, make sure you have the model and the data path
python train_llama.py \
    --mode train \
    --task imaging \
    --data_path data/filt/train.json \
    --test_data_path data/filt/test.json \
    --model_path model/llama \
    --output_dir output