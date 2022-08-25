from torchvision import transforms, datasets
import torch

imagenet_transformation = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ])

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

class CueConflictDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""

    def __init__(self, root_path="datasets/cue_conflict",transform=None):
        super(CueConflictDataset, self).__init__(root_path,transform=transform)

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
    dataset = CueConflictDataset(root_path, imagenet_transformation)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    return loader

    


