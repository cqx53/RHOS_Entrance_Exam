import os
# import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchmetrics.functional.classification import multilabel_average_precision
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import Dataset, DataLoader
import imageio
import time
import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes*")
import random
import sys
import copy
import json
from PIL import Image
import pickle
from dataset import INFO, topkDataset
from dataset_attr import attrDataset
from dataset_aff import affDataset
import clip
from tqdm import trange
# from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
# import transformers
def test_model(model, test_dl, text_tokens):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corrects = torch.tensor(0)
    model.to(device)

    model.eval()
    for i, (imgs, labels) in enumerate(test_dl):
        with torch.no_grad():
            imgs = imgs.to(device)
            labels = labels.to(device)
            image_features = model.encode_image(imgs).float().to(device)
            # print(image_features.shape)
            text_features = model.encode_text(text_tokens).float().to(device)
            # print(text_features.shape)

            # return 
            image_features /= image_features.norm(dim = -1, keepdim = True)
            text_features /= text_features.norm(dim = -1, keepdim = True)

            similarity = image_features @ text_features.T
            # print(similarity[0])
            # print(len(similarity[0].cpu().numpy()))


            # 取k = 10
            # k = 5 -> Accuracy: 0.8113394132512911
            # k = 10 -> Accuracy: 0.8683661136138886
            # k = 100 -> Accuracy: 0.9748379298978135
            for j, sim in enumerate(similarity):
                # values, indices = torch.topk(sim, 5)
                values, indices = torch.topk(sim, 5)
                # a = indices.cpu().tolist()
                # print(a)
                # a.sort()
                # print(a)
                # print(labels[k].item())
                # print(labels[k].item() in set(a))

                # return 

                # print(sim)
                # print(values)
                # print(indices)
                # print(labels[0])
                # print(labels[0].item())
                # print(set(indices.cpu().tolist()))
                if labels[j].item() in set(indices.cpu().tolist()):
                    corrects += 1
            # return
    # print(f"corrects: {corrects}")
    print(f"Accuracy: {corrects.double() / len(test_dl.dataset)}")

def compute_AP(rank):
    rank = sorted(rank, key = lambda x: x[0], reverse = True)
    num_pos = sum(1 for _, y in rank if y == 1)
    if num_pos == 0:
        return 0.0
    tp = 0
    ap_sum = 0.0
    for i, (_, y) in enumerate(rank, start = 1):
        if y == 1:
            tp += 1
            precision_at_i = tp/i
            ap_sum += precision_at_i
    return ap_sum / num_pos

def test_attr_model(model, test_attr_dl):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    # 一共有哪些attributes先取出来
    attribute_list = test_attr_dl.dataset.attributes
    # print(test_attr_dl.dataset.attribute_to_idx['pink'])
    # data = test_attr_dl.dataset[4707]

    prompts = test_attr_dl.dataset.prompts
    # print(test_attr_dl.dataset.attribute_to_idx[attribute_list[0]])
    mAP = [0 for _ in range(114)]

    ap = 0
    text_tokens = clip.tokenize(prompts).to(device)
    preds = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    # for i, prompt in enumerate(prompts):
    for i, (imgs, attrs) in enumerate(test_attr_dl):
        with torch.no_grad():
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            image_features = model.encode_image(imgs).float().to(device)
            text_features = model.encode_text(text_tokens).float().to(device)

            image_features /= image_features.norm(dim = -1, keepdim = True)
            text_features /= text_features.norm(dim = -1, keepdim = True)

            # similarity = text_features @ image_features.T
            similarity = image_features @ text_features.T
            # print(similarity[0])
            # preds = preds + similarity
            preds = torch.cat([preds, similarity])
            targets = torch.cat([targets, attrs])
            # print(preds.shape)
            # print(similarity.shape)
            # print(similarity == preds)
    myAPlist = multilabel_average_precision(preds, targets.int(), num_labels = len(attribute_list), average = None, thresholds = None)
    # print(targets)
    # print(targets.int())
    # print(myAPlist.shape)   
    # print(myAPlist)
    mAP = torch.sum(myAPlist).item() / len(attribute_list)
    print(f"mAP(attibute): {mAP}")
    return
    for i in trange(len(prompts)):
        prompt = prompts[i]
        rank = []
        text_tokens = clip.tokenize(prompt).to(device)
        for j, (imgs, attributes) in enumerate(test_attr_dl):
            with torch.no_grad():
                imgs = imgs.to(device)
                # labels = labels.to(device)
                image_features = model.encode_image(imgs).float().to(device)
                # print(image_features.shape)
                text_features = model.encode_text(text_tokens).float().to(device)

                image_features /= image_features.norm(dim = -1, keepdim = True)
                text_features /= text_features.norm(dim = -1, keepdim = True)

                # similarity = image_features @ text_features.T
                similarity = text_features @ image_features.T
                similarity = similarity.cpu().tolist()[0]
                for k, sim in enumerate(similarity):
                    # attribute_list[i]是一个str ex. 'crumpled'
                    # test_attr_dl.dataset.attribute_to_idx[attribute_list[i]] 就是现在在处理的attribute的代号
                    idx = test_attr_dl.dataset.attribute_to_idx[attribute_list[i]]

                    T_F_value = 1 if attributes[k][idx].int().item() == 1 else 0
        
                    # print((sim, T_F_value))
                    rank.append((sim, T_F_value))
                
        # print(len(rank))
        ap += compute_AP(rank)
        # print(ap)
    mAP = ap / (len(attribute_list))
                
    print(f"mAP(attribute): {mAP}")

def test_aff_model(model = None, test_aff_dl = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    # 一共有哪些attributes先取出来
    affordance_list = test_aff_dl.dataset.affordances
    # print(test_attr_dl.dataset.attribute_to_idx['pink'])
    # data = test_attr_dl.dataset[4707]

    prompts = test_aff_dl.dataset.affordances
    # print(prompts)
    # print(prompts == affordance_list)
    # return 
    # print(test_attr_dl.dataset.attribute_to_idx[attribute_list[0]])

    
    mAP = [0 for _ in range(len(affordance_list))]
    text_tokens = clip.tokenize(prompts).to(device)
    preds = torch.tensor([]).to(device)
    targets = torch.tensor([]).to(device)
    for i, (imgs, affs) in enumerate(test_aff_dl):
        with torch.no_grad():
            imgs = imgs.to(device)
            affs = affs.to(device)
            image_features = model.encode_image(imgs).float().to(device)
            text_features = model.encode_text(text_tokens).float().to(device)

            image_features /= image_features.norm(dim = -1, keepdim = True)
            text_features /= text_features.norm(dim = -1, keepdim = True)

            # similarity = text_features @ image_features.T
            similarity = image_features @ text_features.T
            # print(similarity[0])
            # preds = preds + similarity
            preds = torch.cat([preds, similarity])
            targets = torch.cat([targets, affs])
            # print(preds.shape)
            # print(similarity.shape)
            # print(similarity == preds)
    myAPlist = multilabel_average_precision(preds, targets.int(), num_labels = len(affordance_list), average = None, thresholds = None)
    # print(targets)
    # print(targets.int())
    # print(myAPlist.shape)   
    # print(myAPlist)
    mAP = torch.sum(myAPlist).item() / len(affordance_list)
    print(f"mAP(affordance): {mAP}")
    return
    ap = 0
    # for i, prompt in enumerate(prompts):
    for i in trange(len(prompts)):
        prompt = prompts[i]
        rank = []
        text_tokens = clip.tokenize(prompt).to(device)
        for j, (imgs, affs) in enumerate(test_aff_dl):
            with torch.no_grad():
                imgs = imgs.to(device)
                # labels = labels.to(device)
                image_features = model.encode_image(imgs).float().to(device)
                # print(image_features.shape)
                text_features = model.encode_text(text_tokens).float().to(device)

                image_features /= image_features.norm(dim = -1, keepdim = True)
                text_features /= text_features.norm(dim = -1, keepdim = True)

                # similarity = image_features @ text_features.T
                similarity = text_features @ image_features.T
                similarity = similarity.cpu().tolist()[0]
                for k, sim in enumerate(similarity):
                    # attribute_list[i]是一个str ex. 'crumpled'
                    # test_attr_dl.dataset.attribute_to_idx[attribute_list[i]] 就是现在在处理的attribute的代号
                    idx = test_aff_dl.dataset.aff_to_idx[affordance_list[i]]

                    T_F_value = 1 if affs[k][idx].int().item() == 1 else 0
        
                    # print((sim, T_F_value))
                    rank.append((sim, T_F_value))
                
        # print(len(rank))
        ap += compute_AP(rank)
        # print(ap)
    mAP = ap / (len(affordance_list))
                
    print(f"mAP(affordance): {mAP}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_transforms = {
        'test':
            transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 用ImageNet的“三围”来标准化
            ]),
        'correct':
            transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }
    test_dataset = topkDataset(transform = data_transforms['correct'])
    test_dl = DataLoader(test_dataset, batch_size = 4)

    
    #for i, batch in enumerate(test_dl):
    #    print(f"Batch {i} with shape: {batch.shape}")
    #    return
    model, preprocess = clip.load("ViT-B/32", device=device)
    for param in model.parameters():
        param.requires_grad = False
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)

    # print(test_dataset.class_to_idx)
    # print(test_dataset.prompts)
    # return 



    text_tokens = clip.tokenize(test_dataset.prompts).to(device)

    test_model(model = model, text_tokens = text_tokens, test_dl = test_dl, ann_file = './resources/OCL_annot_test.pkl', root_dir = '.')

    # --------- 计算attribute(s)的mean Average Precision
    test_attr_dataset = attrDataset(transform = data_transforms['correct'], ann_file = './resources/OCL_annot_test.pkl', root_dir = '.')
    test_attr_dl = DataLoader(test_attr_dataset, batch_size = 4)
    test_attr_model(model = model, test_attr_dl = test_attr_dl)

    test_aff_dataset = affDataset(transform=data_transforms['correct'], ann_file = './resources/OCL_annot_test.pkl', root_dir = '.')
    test_aff_dl = DataLoader(test_aff_dataset, batch_size = 4)
    test_aff_model(model = model, test_aff_dl = test_aff_dl)
    
        
if __name__ == '__main__':
    main()