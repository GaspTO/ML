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
nets = [
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

batch_size = 128
max_batches = None #!
num_workers = 8
device = "cuda:3"
data = []
for name, net_lambda, acc in nets:
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
    dataframe.to_csv("results3_logits.txt")

    del(net)