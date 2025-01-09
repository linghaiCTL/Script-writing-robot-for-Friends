# Script-writing-robot-for-Friends

## Quick Start
Clone the source code from GitHub:
```
git clone https://github.com/linghaiCTL/Script-writing-robot-for-Friends.git
cd Script-writing-robot-for-Friends
```

First prepare the environment by running the following command:
```
conda create -n SWM
conda activate SWM
pip install -r requirements.txt
```

Then you need to download the Llama-3.2-1B-Instruct model, you can simply done this by running (which utilizes the source of model scope):
```
python model/download_llama.py
```

The dataset and peft parameter is contained in this repo, therefore you don't need to download it from other resource.

To run training and inference, you can simply run the bash scripts. Please make sure the path of model and data correct before running.
```
# for training:
bash script/train.sh
# for inference:
bash script/inference.sh
```
