from turtle import forward
from torch.utils.data import DataLoader
from benchmarks import TwoFactor_NeuronalDimensionality, ShapeBiasBenchmark
from datasets.neural_dim_datasets import Paired_StylizedVoc2012
from torchvision.models.feature_extraction import get_graph_node_names
from models import rwightman_model_pool
from datasets import imagenet_transformation, cropped_imagenet_transform
import timm
import torch
from torchvision import models
import numpy as np
import pandas as pd
import traceback
import os
from tools import Timeout
from torchvision import transforms, datasets
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def get_timm_transform(model):
    return  create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


def image_size(net,default_image_size):
    try:
        return (net.pretrained_cfg["input_size"][1], net.pretrained_cfg["input_size"][2])
    except:
        return default_image_size



if __name__ == '__main__':
    #model_names = timm.list_models(pretrained=True)
    model_names = timm.list_models("tf_efficientnet*",pretrained=True)
    batch_size = 64
    num_workers = 16
    device = "cuda:2" #!
    default_image_size = 513
    max_batches = None
    shape_file = None
    dim_file = "2NOV_effnets_fulltransform_pretrained.csv"
    run_shape_benchmark = False
    run_dim_benchmark = True
    timeout = 60    
    pretrained = True
    never_skip = False

    if dim_file is not None and os.path.exists(dim_file):
        df_dim = pd.read_csv(dim_file)
    else:
        df_dim = pd.DataFrame()
        
    if shape_file is not None and os.path.exists(shape_file):
        df_shape = pd.read_csv(shape_file)
    else:
        df_shape = pd.DataFrame()
        
    for name in model_names:
        try:
            print("Testing Name:" + str(name))
                
            #SHAPE BIAS 
            if run_shape_benchmark:
                if df_shape.empty or never_skip or (name not in df_shape["name"].values):
                    with Timeout(timeout):
                        net = timm.create_model(name,pretrained=pretrained).eval()
                    size = image_size(net,224)
                    shapebias_stat, _ = ShapeBiasBenchmark(batch_size,num_workers,device=device,size=size)(net)
                    shape_results = {"name":name,
                            "shape_bias":shapebias_stat["shape_bias"],
                            "shape_match":shapebias_stat["shape_match"],
                            "texture_match":shapebias_stat["texture_match"],
                            "img_size":size}
                    df_shape = pd.concat([df_shape,pd.DataFrame([shape_results])]).sort_values(by="name")
                    if shape_file is not None: df_shape.to_csv(shape_file)
                    del(net)
                else:
                    print("skipping shapebias benchmark on " + name + "...")
                    
            #NEURONAL DIM
            if run_dim_benchmark:
                if df_dim.empty or never_skip or (name not in df_dim["name"].values):
                    with Timeout(timeout):
                        net = timm.create_model(name,pretrained=pretrained,num_classes=0).eval()
                    img_size = image_size(net,default_image_size)
                    dim_results = {"name": name, "img_size":img_size}
                    dataloader = DataLoader(Paired_StylizedVoc2012(
                                                    transform=get_timm_transform(net)),
                                                    batch_size=batch_size,shuffle=True)
                    dim_results_ = TwoFactor_NeuronalDimensionality(dataloader,max_batches=max_batches,device=device)(net)
                    dim_results.update(dim_results_)
                    description_max_batches = np.inf if max_batches is None else max_batches
                    
                    df_dim = pd.concat([df_dim,pd.DataFrame([dim_results])]).sort_values(by="name")
                    if dim_file is not None: df_dim.to_csv(dim_file)
                    del(net)
                else:
                    print("skipping dim benchmark on model " + name + "...")

        except Exception as e:
            print("problem with model:" + str(name))
            print("error:")
            traceback.print_exc()
            print("\n\n")

