from torchvision import transforms, datasets


imagenet_transformation = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                        ])

class CueConflictDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""

    def __init__(self, root_path="datasets/cue_conflict",transform=None):
        if transform is None: transform = imagenet_transformation
        super(CueConflictDataset, self).__init__(root_path,transform=transform)
        self._get_shape_and_texture_data()
        self._shape_data, self._texture_data = None, None
    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        sample, _ = super().__getitem__(index)
        path = self.imgs[index][0]
        shape,texture, _, _ = self._extract_concepts_from_path(path)
        assert shape in self.class_to_idx
        assert texture in self.class_to_idx
        return sample,shape,texture,path

    @property
    def shape_data(self):
        if self._shape_data is None:
            self._get_shape_and_texture_data()
        return self._shape_data

    @property
    def texture_data(self):
        if self._texture_data is None:
            self._get_shape_and_texture_data()
        return self._texture_data

    def _extract_concepts_from_path(self,path):
        file_name = path.split("/")[-1][:-4]
        raw_shape = file_name.split("-")[0]
        shape = ''.join([i for i in raw_shape if not i.isdigit()])
        raw_texture = file_name.split("-")[1][:-1]
        texture = ''.join([i for i in raw_texture if not i.isdigit()])
        return shape, texture, raw_shape, raw_texture

    def _get_shape_and_texture_data(self):
        self._shape_data = {}
        self._texture_data = {}
        for i in tqdm(range(len(self.imgs))):
            path = self.imgs[i][0]
            sample, _ = super().__getitem__(i)
            shape, texture, _, _ = self._extract_concepts_from_path(path)
            if shape not in self._shape_data:
                self._shape_data[shape] = [sample]
            else:
                self._shape_data[shape].append(sample)
            
            if texture not in self.texture_data:
                self._texture_data[texture] = [sample]
            else:
                self._texture_data[texture].append(sample)

        for shape,texture in zip(self._shape_data,self._texture_data):
            self._shape_data[shape] = torch.stack(self._shape_data[shape])
            self._texture_data[texture] = torch.stack(self._texture_data[texture])


def cueconflict_dataloader(root_path="datasets/cue_conflict", batch_size=32, num_workers=4):
    dataset = CueConflictDataset(root_path, imagenet_transformation)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    return loader




