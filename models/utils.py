from math import ceil


def get_same_padding(kernel_size,stride,in_size=0):
    out_size = ceil(in_size/stride)
    return ceil(((out_size-1)*stride+ kernel_size-in_size)/2)#(k-1)//2 for same padding


def squeeze_batch_frames(target):
    batch_size,num_frames,*extra = target.shape
    return target.reshape(batch_size*num_frames,*extra)