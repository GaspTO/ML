from benchmarks import TwoFactor_NeuronalDimensionality, ShapeBiasBenchmark
from torchvision.models import *
from torchvision.models.feature_extraction import get_graph_node_names
import torch
from datasets.neural_dim_datasets import Paired_StylizedVoc2012
from datasets import imagenet_transformation, cropped_imagenet_transform
from torch.utils.data import DataLoader

''' models '''
resnet_50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
resnet_101 = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2).eval()
vit = vit_l_32(pretrained=True)
efficientnetb4 = efficientnet_b4(pretrained=True)

''' chosen model '''
model = resnet_101
model = model.eval()
names = get_graph_node_names(model)


try:
    image_size = model.__getattr__("image_size")
except:
    image_size = 513




benchmark = ShapeBiasBenchmark(100,4)
print(benchmark(model))
print("\n\n\n")

dataloader = DataLoader(Paired_StylizedVoc2012(
                            transform=cropped_imagenet_transform(image_size)),
                            batch_size=8,shuffle=True)
benchmark = TwoFactor_NeuronalDimensionality(dataloader,max_batches=5)
benchmark(model,layers=[names[1][-1]])

