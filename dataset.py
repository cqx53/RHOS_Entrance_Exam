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
class topkDataset(Dataset):
    def __init__(self, ann_file = Ann_file, root_dir = INFO, transform = None, OCL_class_object_filename = "/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_class_object.json"):
        self.ann_file = ann_file
        self.classes = self.get_classes()
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }
        self.root_dir = root_dir
        self.OCL_class_object_filename = OCL_class_object_filename
        self.img_label = self.load_annotations()
        self.img_box = self.load_box()
        self.top10classes = [k for k, _ in Counter(list(self.img_label.values())).most_common(10)]
        self.img = [os.path.join(root_dir, img) for img in list(self.img_label.keys())]
        # self.classes = self.get_classes()
        self.label = [label for label in list(self.img_label.values())]
        self.transform = transform
        self.prompts = self.to_prompt(self.classes)
        
    def load_annotations(self): # <- by pickle module
        data_infos = {}
        with open(self.ann_file, "rb") as f:
            # index_objectclass_mapping_reference = self.get_obj_index_map()
            data = pickle.load(f)
            for img in data:
                # data[0]['objects'][0]['obj']
                specific_type = img['objects'][0]['obj']
                # data_infos[img['name']] = np.array( index_objectclass_mapping_reference[specific_type] )
                data_infos[img['name']] = specific_type
        return data_infos
    
    def to_prompt(self, words):
        return ["a photo of " + w for w in words]
    
    def __getitem__(self, idx):
        image = Image.open(self.img[idx]).convert("RGB") # <- RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224] 有些是灰度图所以要.convert("RGB")
        image = image.crop(self.img_box[ (self.img[idx])[len(INFO):] ])
        label = torch.tensor(self.class_to_idx[self.label[idx]])
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.img)
    def get_classes(self):
        with open("/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_class_object.json", "r") as f:
            return json.load(f)
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

