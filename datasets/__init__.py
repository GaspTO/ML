from cgitb import text
from torchvision import transforms, datasets
import torch
import os

imagenet_transformation = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ])

cropped_imagenet_transform = lambda size: transforms.Compose([
                                transforms.ToTensor(),
                                transforms.CenterCrop(size),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

"""                             
...................................
..............IMAGEBET.............
...................................
"""
def get_imagenet_val(transform=None):
    root = "datasets/imagenet/val"
    dataset = datasets.ImageFolder(root,transform=transform,target_transform=None)
    original_idx_to_true_indx, true_index = {}, 0
    for line in open("datasets/imagenet/classes.txt"):
        class_name = line.replace('\n','')
        if class_name in dataset.class_to_idx:
            original_index = dataset.class_to_idx[class_name]
            original_idx_to_true_indx[original_index] = true_index
            true_index += 1
    dataset.target_transform = lambda idx: original_idx_to_true_indx[idx] 
    return dataset



"""                             
...................................
............SHAPE BIAS.............
...................................
"""

class CueConflict(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""

    def __init__(self, root_path="datasets/cue_conflict",transform=None):
        super(CueConflict, self).__init__(root_path,transform=transform)

    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        sample, _ = super().__getitem__(index)

        path = self.imgs[index][0]
        file_name = path.split("/")[-1][:-4]
        raw_shape = file_name.split("-")[0]
        shape = ''.join([i for i in raw_shape if not i.isdigit()])
        raw_texture = file_name.split("-")[1][:-1]
        texture = ''.join([i for i in raw_texture if not i.isdigit()])

        assert shape in self.class_to_idx
        assert texture in self.class_to_idx
        
        return sample,shape,texture,path

def cueconflict_dataloader(root_path="datasets/cue_conflict", batch_size=32, num_workers=4):
    dataset = CueConflict(root_path, imagenet_transformation)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    return loader

    
"""                             
...................................
............StylizedVOC2012........
...................................
"""

class StylizedVOC2012(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""

    def __init__(self, root_path="datasets/styized_voc2012",transform=None):
        super(StylizedVOC2012, self).__init__(root_path,transform=transform)
        self.textures = ["woven","cracked","marbled","pleated","potholed"]
        self.shapes = ['person', 'pottedplant', 'boat', 'sofa', 'bicycle', 'horse',
                       'chair', 'tvmonitor', 'bottle', 'dog', 'motorbike', 'bird',
                       'train', 'car', 'diningtable', 'cat', 'aeroplane', 'sheep', 'bus', 'cow']
        self._texture_map, self._shape_map, self._texture_shape_map, self._shape_texture_map = None,None,None,None

    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        sample, _ = super().__getitem__(index)

        path = self.imgs[index][0]
        filename, texture, shape = self.break_path(path)
        
        assert shape in self.shapes
        assert texture in self.textures
        
        return sample, shape, texture, path

    def break_path(self,path):
        filename = path.split("/")[-1][:-4]
        texture, shape = filename.split("_")[0:2]
        return filename, texture, shape
    
    def _set_texture_shape_maps(self):
        self._texture_map, self._shape_map, self._texture_shape_map, self._shape_texture_map = {}, {}, {}, {}
        for texture in self.textures:
            self._texture_map[texture] = []
            self._texture_shape_map[texture] = {}
        for shape in self.shapes:
            self._shape_map[shape] = []
            self._shape_texture_map[shape] = {} 
        for texture in self.textures:
            for shape in self.shapes:
                self._shape_texture_map[shape][texture] = []
                self._texture_shape_map[texture][shape] = []
                
        for path, _ in self.imgs:
            _, texture, shape = self.break_path(path)
            self._texture_map[texture].append(path)
            self._shape_map[shape].append(path)
            self._texture_shape_map[texture][shape].append(path)
            self._shape_texture_map[shape][texture].append(path)
        
    @property
    def texture_map(self):
        if self._texture_map is None:
            self._set_texture_shape_maps()
        return self._texture_map
    
    @property
    def shape_map(self):
        if self._shape_map is None:
            self._set_texture_shape_maps()
        return self._shape_map
 
    @property
    def texture_shape_map(self):
        if self._texture_shape_map is None:
            self._set_texture_shape_maps()
        return self._texture_shape_map
            
    @property
    def shape_texture_map(self):
        if self._shape_texture_map is None:
            self._set_texture_shape_maps()
        return self._shape_texture_map
    
    
        
            
