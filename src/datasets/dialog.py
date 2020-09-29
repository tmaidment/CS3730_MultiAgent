from PIL import Image
import torch
import json
import os

from nltk.tokenize import TweetTokenizer

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

        # setting up vocab based on: 
        # https://github.com/GuessWhatGame/guesswhat/blob/ \
        # master/src/guesswhat/preprocess_data/create_dictionary.py
        self.vocab = {'<padding>': 0,
              '<start>': 1,
              '<stop>': 2,
              '<stop_dialogue>': 3,
              '<unk>': 4,
              '<yes>' : 5,
              '<no>': 6,
              '<n/a>': 7,
              }
        self.tknzr = TweetTokenizer(preserve_case=False)
        counts = dict()
        for data in self.json_objs:
            for qa in data['qas']:
                q = qa['question']
                tokens = self.tknzr.tokenize(q)
                for tok in tokens:
                    if tok not in counts:
                        counts[tok] = 0
                    counts[tok] += 1
        for word, c in counts.items():
            if c >= 3 and word.count('.') <= 1:
                self.vocab[word] = len(self.vocab)
        self.image_loc = image_loc
    
    def __len__(self):
        return len(self.json_objs)
    
    def tokenize(self, question):
        _vocab = lambda w: self.vocab[w] \
            if w in self.vocab \
            else 4
    
    def __getitem__(self, i):
        data = self.json_objs[i]
        img_path = os.path.join(self.image_loc, data['image']['file_name'])
        img = Image.open(img_path)
        data['image']['raw'] = img

        # TODO:
        # encode questions and answers as integers?
        
        for j in range(len(data['objects'])):
            # [x,y,width,height]
            # x, y: the upper-left coordinates of the bounding box
            [x_tl, y_tl, w, h] = data['objects'][j]['bbox']
            xmin = unit_normalize(x_tl, w, 0)
            xmax = unit_normalize(x_tl + w, w, 0)
            xcenter = unit_normalize(x_tl + w // 2, w, 0)
            ymin = unit_normalize(y_tl, h, 0)
            ymax = unit_normalize(y_tl + h, h, 0)
            ycenter = unit_normalize(y_tl + h // 2, h, 0)
            wbox = unit_normalize(w, w, 0)
            hbox = unit_normalize(h, h, 0)
            # spatial features used by https://arxiv.org/pdf/1703.05423.pdf
            data['objects'][j]['spatial'] = [xmin, ymin, xmax, 
                ymax, xcenter, ycenter, wbox, hbox]

        return data
