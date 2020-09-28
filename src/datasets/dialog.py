from PIL import Image
import torch
import json
import os

def unit_normalize(x, max_, min_):
    """
    Returns x normalized to the range -1,1
    """
    x_zero_one = (x - min_) / (max_ - min_)
    return 2 * x_zero_one - 1

class Dialog(torch.utils.data.Dataset):

    def __init__(self, image_loc, json_loc):
        self.json_objs = [json.loads(l) for l in 
            open(json_loc, 'r').readlines()]
        self.image_loc = image_loc
    
    def __len__(self):
        return len(self.json_objs)
    
    def __getitem__(self, i):
        x = self.json_objs[i]
        img_path = os.path.join(self.image_loc, x['image']['file_name'])
        img = Image.open(img_path)
        x['image']['raw'] = img

        # TODO:
        # encode questions and answers as integers?
        
        for i in range(len(x['objects'])):
            # [x,y,width,height]
            # x, y: the upper-left coordinates of the bounding box
            [x_tl, y_tl, w, h] = x['objects']['bbox']
            xmin = unit_normalize(x_tl, w, 0)
            xmax = unit_normalize(x_tl + w, w, 0)
            xcenter = unit_normalize(x_tl + w // 2, w, 0)
            ymin = unit_normalize(y_tl, h, 0)
            ymax = unit_normalize(y_tl + h, h, 0)
            ycenter = unit_normalize(y_tl + h // 2, h, 0)
            wbox = unit_normalize(w, w, 0)
            hbox = unit_normalize(h, h, 0)
            # spatial features used by https://arxiv.org/pdf/1703.05423.pdf
            x['objects']['spatial'] = [xmin, ymin, xmax, 
                ymax, xcenter, ycenter, wbox, hbox]
                
        return x
