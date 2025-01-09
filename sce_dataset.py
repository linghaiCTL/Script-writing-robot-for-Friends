import torch
from torch.utils.data import Dataset
import json
from template import Scene_gen_prompt, Summarize_prompt, Imaging_prompt
import random

class line:
    def __init__(self, character, dialogue):
        self.character = character
        self.dialogue = dialogue

class scene:
    def __init__(self, lines, background):
        self.background = background
        self.lines = lines
        self.characters = set([line.character for line in lines])

    def analyze(self):
        for line in self.lines:
            analyze_scene(line)

    def __str__(self):
        return '\n'.join([str(line) for line in self.lines])
    
    def to_dict(self):
        return {'background': self.background, 'characters': list(self.characters), 'lines': [line.__dict__ for line in self.lines]}
    
    def view(self):
        for line in self.lines:
            print('['+line.character + ']: ' + line.dialogue)

def get_scene(lines, ptr):
    sce = []
    # Each scene should start with [Scene: ...]
    while(ptr < len(lines) and not lines[ptr].startswith('[')):
        # print(lines[ptr])
        ptr += 1
    if ptr >= len(lines):
        return None, ptr
    sce.append(lines[ptr])
    ptr += 1
    while ptr < len(lines) and lines[ptr].strip() != '':
        if lines[ptr].startswith('['):
            break
        if lines[ptr] == '':
            ptr += 1
            continue
        sce.append(lines[ptr])
        ptr += 1
    return sce, ptr

def analyze_scene(sce):
    if sce[0].startswith('[Scene:'):
        background = sce[0].split('[')[1].split(']')[0]
        lines=[]
        for l in sce[1:]:
            if l == '':
                continue
            try:
                character = l.split(':')[0]
                dialogue = l.split(':')[1]
                lines.append(line(character, dialogue))
            except:
                continue
        sce_ = scene(lines, background)
        #sce_.view()
        return sce_
    else:
        print(sce[0])
    return None

class scene_dataset(Dataset):
    def __init__(self, scenes):
        self.scenes = scenes
        
    def __len__(self):
        return len(self.scenes)
    
    def __getitem__(self, idx):
        return self.scenes[idx].to_dict()
    
    def collate_fn(self, batch):
        return batch
    
class line_dataset(Dataset):
    def __init__(self, file_path, process, typed='generate'):
        print('dataset type:', typed)
        self.template=None
        self.typed=typed
        if self.typed == 'generate':
            self.template=Scene_gen_prompt
        elif self.typed == 'imaging':
            print('imaging')
            self.template=Imaging_prompt
        self.data=None
        self.load(file_path)
        self.process=process
    
    def load(self, json_path):
        with open(json_path, 'r') as f:
            file_data = json.load(f)
        self.data=[]
        for scene in file_data:
            scene_prompt=self.template.format(background=scene['background'], characters=scene['characters'])
            raw_input_prompt=scene_prompt+'[[main dialogue]]\n'
            input_prompt=scene_prompt+'[[Main Content]]'+scene['summary']+'\n'+'[[main dialogue]]\n'
            train_prompt=input_prompt
            label=''
            for j in range(len(scene['lines'])):
                if j<=3:
                    train_prompt+=scene['lines'][j]['character']+' '+scene['lines'][j]['dialogue']+'\n'
                else:
                    line_=scene['lines'][j]
                    label+=line_['character']+' '+line_['dialogue']+'\n'
            label+='[End of Scene]'
            self.data.append({'input':train_prompt, 'label':label, 'test':input_prompt, 'summary':scene['summary'], 'scene_prompt':scene_prompt, 'raw_input':raw_input_prompt})
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def collate_fn(self, batch):
        return batch
    
class llama3_wrap_dataset(line_dataset):
    def __init__(self, file_path, process, tokenizer, typed='generate'):
        self.data=line_dataset(file_path, process, typed)
        self.process=process
        self.tokenizer=tokenizer
        self.typed=typed
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.typed=='generate':
            input_text={
                'conversations': [
                {
                    'from': 'user',
                    'value': self.data[idx]['input']
                },
                {
                    'from': 'assistant',
                    'value': self.data[idx]['label']
                },
                ],
                'test': self.data[idx]['test'],
                'raw_input': self.data[idx]['raw_input'],
            }
        elif self.typed=='imaging':
            input_text={
                'conversations': [
                {
                    'from': 'user',
                    'value': self.data[idx]['scene_prompt']
                },
                {
                    'from': 'assistant',
                    'value': self.data[idx]['summary']
                },
                ],
                'test': self.data[idx]['scene_prompt'],
                'response': self.data[idx]['label']
            }
        #print(input_text)
        #input()
        if self.process:
            input_text=self.process(input_text, self.tokenizer)
        return input_text
    
    def collate_fn(self, batch):
        return batch
    
from transformers import AutoTokenizer

if __name__ == "__main__":
    sce_dataset=line_dataset('data\\filt\\train.json', None)
    tokenizer = AutoTokenizer.from_pretrained('E:\\code\\NLP\\final_proj\\model\\llama', use_fast=False, trust_remote_code=True)
    length_bigger_than_512=0
    for i in range(len(sce_dataset)):
        if len(tokenizer(sce_dataset[i]['label'])['input_ids'])>1024:
            length_bigger_than_512+=1
    print(length_bigger_than_512)