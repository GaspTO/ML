from socket import NETLINK_DNRTMSG
from benchmarks import TwoFactor_NeuronalDimensionality, ShapeBiasBenchmark
from torchvision import models
from torchvision.models.feature_extraction import get_graph_node_names
import torch
from datasets.neural_dim_datasets import Paired_StylizedVoc2012
from datasets import imagenet_transformation, cropped_imagenet_transform
from torch.utils.data import DataLoader
import pandas as pd

''' models '''
sixteen_nets = [
    ("resnet18",lambda :models.resnet18(pretrained=True),69.758),
    ("vit_l_32",lambda :models.vit_l_32(pretrained=True),76.972),
    ("efficientnetB7",lambda :models.efficientnet_b7(pretrained=True),84.122),
    ("vit_b_16",lambda :models.vit_b_16(pretrained=True),81.072),
    ("vit_b_32",lambda :models.vit_b_32(pretrained=True),75.912),
    ("vit_l_16",lambda :models.vit_l_16(pretrained=True),79.662),
    ("alexnet",lambda :models.alexnet(pretrained=True),56.522),
    ("inceptionv3",lambda :models.inception_v3(pretrained=True),77.294),
    ("resnet50",lambda :models.resnet50(pretrained=True),76.130),
    ("resnet101",lambda :models.resnet101(pretrained=True),77.374),
    ("densenet121",lambda :models.densenet121(pretrained=True),74.434),
    ("densenet169",lambda :models.densenet169(pretrained=True),75.600),
    ("resnext-50-32x4d",lambda :models.resnext50_32x4d(pretrained=True),77.618),
    ("wide_resnet-50-2",lambda :models.wide_resnet50_2(pretrained=True),78.468),
    ("wide_resnet-101-2",lambda :models.wide_resnet101_2(pretrained=True),78.848),
    ("efficientnetB4",lambda :models.efficientnet_b4(pretrained=True),83.384),
]

## 11 models
efficient_nets = [ 
    ("efficientnetV2_l",lambda: models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1),85.808),
    ("efficientnetV2_m",lambda: models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1),85.112),
    ("efficientnetV2_s",lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),84.228),
    ("efficientnetB7",lambda: models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1),84.122),
    ("efficientnetB6",lambda: models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1),84.008),
    ("efficientnetB5",lambda: models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1),83.444),
    ("efficientnetB4",lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1),83.384),
    ("efficientnetB3",lambda: models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1),82.008),
    ("efficientnetB2",lambda: models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1),80.608),
    ("efficientnetB1",lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1),78.642),
    ("efficientnetB0",lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),77.692),
]

## 4 models
dense_nets = [
    ("densenet121", lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),74.434),
    ("densenet161", lambda: models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1),77.138),
    ("densenet169", lambda: models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1),75.600),
    ("densenet201", lambda: models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1),76.896)
]

## 5 models
res_nets = [
    ("resnet18",lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),69.758),
    ("resnet34",lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1),73.314),
    ("resnet50",lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),76.130),
    ("resnet101",lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1),77.374),
    ("resnet152",lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1),78.374),
]

## 4 models
mnas_nets = [
    ("mnasnet0_5",lambda: models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1),67.734),
    ("mnasnet0_75",lambda: models.mnasnet0_75(weights=models.MNASNet0_75_Weights.IMAGENET1K_V1),71.180),
    ("mnasnet1_0",lambda: models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1),73.456),
    ("mnasnet1_3",lambda: models.mnasnet1_3(weights=models.MNASNet1_3_Weights.IMAGENET1K_V1),76.506),
]

## 4 models
vits = [
    ("vit_b_16",lambda :models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1),81.072),
    ("vit_b_32",lambda :models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1),75.912),
    ("vit_l_16",lambda :models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1),79.662),
    ("vit_l_32",lambda :models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1),76.972),
]

## 2 models
inception_nets = [
    ("googlenet", lambda: models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1),69.778),
    ("inceptionv3", lambda: models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1),77.294)
]


all_nets = [
    *efficient_nets, *dense_nets, *res_nets, *mnas_nets, *vits, *inception_nets
]

for n in all_nets:
    n[1]()

exit(-1)

batch_size = 128
max_batches = None #!
num_workers = 8
device = "cuda:3"
data = []
for name, net_lambda, acc in efficient_nets:
    print("running " + str(name) + " ...")
    net = net_lambda()
    net = net.eval()
    names = get_graph_node_names(net)
    try:
        image_size = getattr(net,"image_size")
    except:
        image_size = 513
    dataloader = DataLoader(Paired_StylizedVoc2012(
                            transform=cropped_imagenet_transform(image_size)),
                            batch_size=batch_size,shuffle=True)
    shapebias_stat, _ = ShapeBiasBenchmark(batch_size,num_workers,device=device)(net)
    
    neuronal_dim = TwoFactor_NeuronalDimensionality(dataloader,max_batches=max_batches,device=device)(net,layers=[names[1][-2]])

    result = {"name":name, "acc_in":acc, "shape_bias":shapebias_stat["shape_bias"], "shape_match":shapebias_stat["shape_match"], "texture_match":shapebias_stat["texture_match"]}
    result.update(neuronal_dim)
    
    data.append(result)

    dataframe = pd.DataFrame(data).sort_values(by="name")
    dataframe.to_csv("results_efficientnets_logits.txt")

    del(net)