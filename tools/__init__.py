class VisionList:
    def __init__(self, paths,transform=None):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index: int):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def loader(self,path: str):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

