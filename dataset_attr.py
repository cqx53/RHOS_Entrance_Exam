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
prompts = [
    "a crumpled object",
    "a ceramic object",
    "a cold object",
    "a curled object",
    "a furry object",
    "a black object",
    "a wet object",
    "an orange object",
    "a brown object",
    "a yellow object",
    "a striped object",
    "a cool object",
    "a gray object",
    "a leather object",
    "a large object",
    "a wooden object",
    "a small object",
    "a soft object",
    "a round object",
    "an old object",
    "a portable object",
    "a fluffy object",
    "a hard object",
    "an object with horns",
    "a messy object",
    "a heavy object",
    "a blue object",
    "a purple object",
    "a closed object",
    "a new object",
    "a red object",
    "a thin object",
    "a full object",
    "a vertical object",
    "a strong object",
    "a dry object",
    "a spotted object",
    "a quadruped animal",
    "a whole object",
    "a sharp object",
    "a long object",
    "a fake object",
    "an open object",
    "a toy object",
    "a plastic object",
    "a white object",
    "a columnar object",
    "an empty object",
    "a flat object",
    "a cloth object",
    "a warm object",
    "a leashed animal",
    "a solid object",
    "a smooth object",
    "a worn object",
    "a rectangular object",
    "a bipedal animal",
    "a tasty food",
    "a curved object",
    "a pink object",
    "a hot object",
    "a digital object",
    "an electric object",
    "a fresh object",
    "a horizontal object",
    "a short object",
    "a natural object",
    "a metal object",
    "a cooked food",
    "a green object",
    "a folded object",
    "a broken object",
    "a bent object",
    "a sliced object",
    "a thick object",
    "a wide object",
    "a narrow object",
    "an arched object",
    "a puffy object",
    "a cream-colored object",
    "a stone object",
    "a cement object",
    "a marble object",
    "a floral object",
    "a glass object",
    "water",
    "a rubber object",
    "a brick object",
    "a sandy surface",
    "a plaid pattern",
    "a paper object",
    "a checkered pattern",
    "a parked vehicle",
    "a moving object",
    "a melted object",
    "a lit object",
    "an object wearing something",
    "a framed object",
    "stacked objects",
    "a tiled surface",
    "a standing object",
    "a hanging object",
    "a sitting person or animal",
    "a walking person or animal",
    "a sleeping person or animal",
    "a flying object",
    "a dead object",
    "a ripe fruit",
    "in the picture",
    "a reflective surface",
    "a grassy area",
    "a leafy plant",
    "a painted object",
    "a rusty object"
]
class attrDataset(Dataset):
    def __init__(self, ann_file = Ann_file, root_dir = INFO, transform = None, OCL_class_attribute_filename = "/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_class_attribute.json"):
        self.ann_file = ann_file
        self.attributes = self.get_attributes()
        self.attribute_to_idx = {
            attr_name: idx for idx, attr_name in enumerate(self.attributes)
        }
        self.root_dir = root_dir
        self.OCL_class_attribute_filename = OCL_class_attribute_filename
        self.img_attr = self.load_annotations()
        self.img_box = self.load_box()
        self.img = [os.path.join(root_dir, img) for img in list(self.img_attr.keys())]
        # self.classes = self.get_classes()
        self.label = [label for label in list(self.img_attr.values())]
        self.transform = transform
        self.prompts = prompts
        
    def load_annotations(self): # <- by pickle module
        data_infos = {}
        with open(self.ann_file, "rb") as f:
            # index_objectclass_mapping_reference = self.get_obj_index_map()
            data = pickle.load(f)
            for img in data:
                
                specific_attributes = img['objects'][0]['attr'] # 会是个List ex. [11, 14, 16, ...]
                
                data_infos[img['name']] = specific_attributes
        return data_infos
    

    # 这个__getitem__()要修改，从attributes = torch.tensor(...)那一行开始改
    def __getitem__(self, idx):
        
        image = Image.open(self.img[idx]).convert("RGB") # <- RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224] 有些是灰度图所以要.convert("RGB")
        image = image.crop(self.img_box[ (self.img[idx])[len(INFO):] ])

        # attributes = torch.tensor(self.attribute_to_idx[self.label[idx]]) <- 这么写会报错：TypeError: unhashable type: 'list'
        attributes = torch.zeros(114)
        # print(self.label[idx])
        # print(self.img[idx])
        attributes[self.label[idx]] = 1
        if self.transform:
            image = self.transform(image)
        return image, attributes # <- 返回这张image有哪些attributes

    def __len__(self):
        return len(self.img)
    def get_attributes(self):
        with open("/data/DATA/OCL_DATA/OCL_data/data/resources/OCL_class_attribute.json", "r") as f:
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

