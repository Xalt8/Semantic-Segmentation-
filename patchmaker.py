import numpy as np
from dataclasses import dataclass
from dataclasses import field


@dataclass
class PatchMaker:
    pic2split:np.ndarray = field(init=False)
    patch:tuple = field(init=False)
    step:int = field(init=False)
    pic_height:int = field(init=False)
    pic_width:int = field(init=False)
    split_indicies:list = field(init=False)
    

    def split(self, pic2split:np.ndarray, patch:tuple, step:int) -> np.ndarray:
        ''' Splits an image array into patches and returns an array 
        ================================================================ '''
        self.pic2split = pic2split
        self.patch = patch
        self.step = step
        self.pic_height, self.pic_width, _ = self.pic2split.shape

        assert len(patch) == 3, "Patch should be tuple -> (100,100,3)"
        assert (self.pic_height%patch[0] == 0) and (self.pic_width%patch[1] == 0), "pic_arr not equally divisible by patch"            
        
        self.split_indicies = [(i, i+step, j, j+step) \
                            for i in range(0, self.pic_height, self.step) \
                            for j in range(0, self.pic_width, self.step)]

        arr5d = np.array([self.pic2split[h:h_step, w:w_step] for h, h_step, w, w_step in self.split_indicies])
        # Convert (63,25,100,100,3) -> (63*25, 100,100,3)
        # return arr5d.reshape(arr5d.shape[0] * arr5d.shape[1], *self.patch)
        return arr5d
         

    def join(self, split_arr:np.ndarray):
        ''' Joins an array into the previous shape passed to method split()
            =============================================================== '''
        z_arr = np.zeros(self.pic2split.shape, dtype=np.int64)
        for arr, (i, i_offset, j, j_offset) in zip(split_arr, self.split_indicies):
            z_arr[i:i_offset, j:j_offset] = arr
        assert self.pic2split.shape == z_arr.shape, "The array split and array joined are not the same shape"
        return z_arr        


if __name__ == "__main__":
    pass