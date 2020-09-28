from PIL import Image
import torch
import json
import os

class Dialog(torch.utils.data.Dataset):

    def __init__(self, image_loc, json_loc):
        self.json_objs = [json.loads(l) for l in 
            open(json_loc, 'r').readlines()]
        self.image_loc = image_loc
    
    def __len__(self):
        return len(self.json_objs)
    
    def __getitem__(self, i):
        obj = self.json_objs[i]
        img_path = os.path.join(self.image_loc, obj['image']['file_name'])
        img = Image.open(img_path)
