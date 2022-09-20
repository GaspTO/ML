from datasets import CueConflict ,StylizedVOC2012
import random
import re


class Paired_StylizedVoc2012(StylizedVOC2012):
    def __init__(self, path="datasets/stylized_voc2012",transform=None):
        super(Paired_StylizedVoc2012,self).__init__(path,transform)
        self.n_factors = 3

    def __getitem__(self, i):  # shape and texture
        # use text file and open example1 and example2 based on self.data txt file
        sample1, shape1, texture1, _ = super().__getitem__(i)
        assert type(texture1) == str and type(shape1) == str
        factor = random.randint(0, self.n_factors-2)

        # select random factor (0 is shape, 1 is texture)
        if factor == 0:
            # same shape, different texture
            list_possible_textures = self.textures.copy()
            list_possible_textures.remove(texture1)
            texture2 = random.choice(list_possible_textures)
            choose_new_file_list = self.texture_shape_map[texture2][shape1]
            path2 = random.choice(choose_new_file_list)
            shape2 = shape1
        else:
            # different shape (class), same texture
            list_possible_shapes = self.shapes.copy()
            list_possible_shapes.remove(shape1)
            # select different image with same texture
            shape2 = random.choice(list_possible_shapes)
            choose_new_file_list = self.shape_texture_map[shape2][texture1]
            path2 = random.choice(choose_new_file_list)
            texture2 = texture1
            
        sample2 = self.loader(path2)
        if self.transform is not None:
            sample2 = self.transform(sample2)
        
        
        return factor, sample1, sample2, shape1, shape2, texture1, texture2

    
    

class Paired_CueConflict(CueConflict):
    """
    Problem, when this compares to images with same shape and different textures, it doesn't need
    to be the same plane, just two planes with different textures.
    """
    def __init__(self, path="datasets/stylized_voc2012",transform=None):
        super(Paired_CueConflict,self).__init__(path,transform)
        self.n_factors = 3

    def __getitem__(self, i):  # shape and texture
        # use text file and open example1 and example2 based on self.data txt file
        sample1, shape1, texture1, path1 = super().__getitem__(i)
        assert type(texture1) == str and type(shape1) == str
        factor = random.randint(0, self.n_factors-2)

        # select random factor (0 is shape, 1 is texture)
        if factor == 0:
            # same shape, different texture
            list_possible_textures = self.textures.copy()
            list_possible_textures.remove(texture1)
            texture2 = random.choice(list_possible_textures)
            id2 = texture2 + path1.split('/')[-1]
            id2 = re.sub("^.*?_",texture2+"_",path1.split('/')[-1])
            
            path2 = path1.split('/')[:-1]
            path2.append(id2)
            path2 = '/'.join(path2)
            shape2 = shape1
        else:
            # different shape (class), same texture
            list_possible_shapes = self.shapes.copy()
            list_possible_shapes.remove(shape1)
            # select different image with same texture
            shape2 = random.choice(list_possible_shapes)
            choose_new_file_list = self.shape_texture_map[shape2][texture1]
            path2 = random.choice(choose_new_file_list)
            texture2 = texture1
        sample2 = self.loader(path2)
        
        return factor, sample1, sample2, shape1, shape2, texture1, texture2

    def __len__(self):
        return len(self.data)
