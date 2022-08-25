import models
from benchmarks import Accuracy, shape_texture_statistics
from datasets import get_imagenet_val, cueconflict_dataloader, CueConflictDataset, imagenet_transformation
from torch.utils.data import random_split
from torchvision.models import *

resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()


'''
transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
dataset = get_imagenet_val(transform=transform)

import torch
torch.tensor(3) in torch.tensor([1,2,3])
percentage = 0.1
dataset, percentage = random_split(dataset,[int(len(dataset)*percentage),len(dataset)-int(len(dataset)*percentage)])
acc = Accuracy("imagenet_accuracy",dataset,8,(1,5))
wut = acc(resnet)

print(wut)
vit = vit_b_32(pretrained=True)
dicto = shape_texture_statistics(vit,8,4)
print("hey")
'''
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(list_1, list_2):
  cos_sim = dot(list_1, list_2) / (norm(list_1) * norm(list_2))
  return cos_sim

import numpy as np
import torch
dataset = CueConflictDataset(transform=imagenet_transformation)


""" """
import re
data_t = []
def texture(string):
    return re.sub(".*-","",string).replace(".png","")

def shape(string):
    return re.sub("-.*","",re.sub(".*/","",string))

airplane1 = []
for d in dataset:
    if texture(d[-1]) == "airplane1":
        data_t.append(d[0])

def all_pairs(data):
    paired_data = []
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            assert i != j
            paired_data.append((data[i],data[j]))
    return paired_data
            
            

import random
from tqdm import tqdm
def correlate_pairs(fun,string,neuron):
    pattern = re.compile(string)
    data = []
    for d in dataset:
        if pattern.match(fun(d[-1])):
            data.append(d[0])
    print("size of data:" + str(len(data)))
    xs = []
    ys = []
    filtred_data = all_pairs(data)
    with torch.no_grad():
        for x,y in tqdm(filtred_data):
            lx = resnet(x.unsqueeze(0))
            ly = resnet(y.unsqueeze(0))
            xs.append(lx[0,neuron])
            ys.append(ly[0,neuron])
    return cosine_similarity(xs,ys)




image = "airplane."
'''
print("TEXTURE")
result = correlate_pairs(texture,image,0)
print(result)
result = correlate_pairs(texture,image,1)
print(result)
result = correlate_pairs(texture,image,2)
print(result)
result = correlate_pairs(texture,image,3)
print(result)
result = correlate_pairs(texture,image,4)
print(result)


print("SHAPE")
result = correlate_pairs(shape,image,0)
print(result)
result = correlate_pairs(shape,image,1)
print(result)
result = correlate_pairs(shape,image,2)
print(result)
result = correlate_pairs(shape,image,3)
print(result)
result = correlate_pairs(shape,image,4)
print(result)
'''




from benchmarks.neuronal_dimensionality import *

def paired_data(data):
    X,Y = [],[]
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            assert i != j
            X.append(data[i])
            Y.append(data[j])
    return torch.stack(X),torch.stack(Y)


dataset = CueConflictDataset(transform=imagenet_transformation)
def get_data(fun,string,dataset):
    pattern = re.compile(string)
    data = []
    for d in dataset:
        if pattern.match(fun(d[-1])):
            data.append(d[0])
    return data

shape_data = get_data(shape,image,dataset)
X_shape, Y_shape = paired_data(shape_data)
texture_data = get_data(texture,image,dataset)
X_texture, Y_texture = paired_data(texture_data)

print("hey")
get_dimensionality(resnet,X_shape,Y_shape,["avgpool"])