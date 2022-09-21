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
    ("resnet18",lambda :models.resnet18(pretrained=True)),
    ("efficientnetB7",lambda :models.efficientnet_b7(pretrained=True)),
    ("vit_l_32",lambda :models.vit_l_32(pretrained=True)),
    ("vit_b_16",lambda :models.vit_b_16(pretrained=True)),
    ("vit_b_32",lambda :models.vit_b_32(pretrained=True)),
    ("vit_l_16",lambda :models.vit_l_16(pretrained=True)),
    ("alexnet",lambda :models.alexnet(pretrained=True)),
    ("inceptionv3",lambda :models.inception_v3(pretrained=True)),
    ("resnet50",lambda :models.resnet50(pretrained=True)),
    ("resnet101",lambda :models.resnet101(pretrained=True)),
    ("densenet121",lambda :models.densenet121(pretrained=True)),
    ("densenet169",lambda :models.densenet169(pretrained=True)),
    ("resnext-50-32x4d",lambda :models.resnext50_32x4d(pretrained=True)),
    ("wide_resnet-50-2",lambda :models.wide_resnet50_2(pretrained=True)),
    ("wide_resnet-101-2",lambda :models.wide_resnet101_2(pretrained=True)),
    ("efficientnetB4",lambda :models.efficientnet_b4(pretrained=True)),
]

batch_size = 128
max_batches = 4 #!
num_workers = 8
device = "cuda:2"
data = []
for name, net_lambda in nets:
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
    shapebias_stat, _ = ShapeBiasBenchmark(batch_size,4,device=device)(net)
    neuronal_dim = TwoFactor_NeuronalDimensionality(dataloader,max_batches=max_batches,device=device)(net,layers=[names[1][-1]])
    data.append((name,
        shapebias_stat["shape_bias"],
        shapebias_stat["shape_match"],
        shapebias_stat["texture_match"],
        neuronal_dim[0],
        neuronal_dim[1],
        neuronal_dim[2],
        neuronal_dim[3],
        neuronal_dim[4]))

    dataframe = pd.DataFrame(data,columns=["name","shape_bias","shape_match","texture_match","dim_3","dim_2","dim_1","dim_0_softmax","dim_0_normalize"])
    dataframe.to_csv("results2.txt")

    del(net)