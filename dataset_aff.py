import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pickle
from collections import Counter
import torch

INFO = "/data/DATA/OCL_DATA/OCL_data/data/"
Ann_file = "/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_annot_test.pkl"

class affDataset(Dataset):
    def __init__(self, ann_file = Ann_file, root_dir = INFO, transform = None, OCL_class_aff_filename = "/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_class_affordance_word.txt"):
        self.ann_file = ann_file
        self.affordances = self.get_affordances()
        self.aff_to_idx = {
            aff_name: idx for idx, aff_name in enumerate(self.affordances)
        }
        self.root_dir = root_dir
        self.OCL_class_aff_filename = OCL_class_aff_filename
        self.img_aff = self.load_annotations()
        self.img_box = self.load_box()
        self.img = [os.path.join(root_dir, img) for img in list(self.img_aff.keys())]
        # self.classes = self.get_classes()
        self.label = [label for label in list(self.img_aff.values())]
        self.transform = transform
        
        
    def load_annotations(self): # <- by pickle module
        data_infos = {}
        with open(self.ann_file, "rb") as f:
            # index_objectclass_mapping_reference = self.get_obj_index_map()
            data = pickle.load(f)
            for img in data:
                
                specific_affs = img['objects'][0]['aff'] # 会是个List ex. [11, 14, 16, ...]
                
                data_infos[img['name']] = specific_affs
        return data_infos
    

    # 这个__getitem__()要修改，从attributes = torch.tensor(...)那一行开始改
    def __getitem__(self, idx):
        
        image = Image.open(self.img[idx]).convert("RGB") # <- RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224] 有些是灰度图所以要.convert("RGB")
        image = image.crop(self.img_box[ (self.img[idx])[len(INFO):] ])

        # attributes = torch.tensor(self.attribute_to_idx[self.label[idx]]) <- 这么写会报错：TypeError: unhashable type: 'list'
        affs = torch.zeros(170)
        # print(self.label[idx])
        # print(self.img[idx])
        affs[self.label[idx]] = 1
        if self.transform:
            image = self.transform(image)
        return image, affs # <- 返回这张image有哪些attributes

    def __len__(self):
        return len(self.img)
    def get_affordances(self):
        affordances = []
        with open("/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_class_affordance_word.txt", "r") as f:
            for line in f:
                aff = line.strip()
                affordances.append(aff)
        return affordances

    def load_box(self):
        data_infos = {}
        with open(self.ann_file, "rb") as f:
            # index_objectclass_mapping_reference = self.get_obj_index_map()
            data = pickle.load(f)
            for img in data:
                # data[0]['objects'][0]['obj']
                box = img['objects'][0]['box']
                # data_infos[img['name']] = np.array( index_objectclass_mapping_reference[specific_type] )
                data_infos[img['name']] = box
        return data_infos

