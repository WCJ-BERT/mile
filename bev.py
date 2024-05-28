import pickle
import pandas as pd
import numpy as np
from PIL import Image
import torch


BIRDVIEW_COLOURS = np.array([[255, 255, 255],          # Background
                             [225, 225, 225],       # Road
                             [160, 160, 160],      # Lane marking
                             [0, 83, 138],        # Vehicle
                             [127, 255, 212],      # Pedestrian
                             [50, 205, 50],        # Green light
                             [255, 215, 0],      # Yellow light
                             [220, 20, 60],        # Red light and stop sign
                             ], dtype=np.uint8)
#print(BIRDVIEW_COLOURS.shape)

def calculate_birdview_labels(birdview, n_classes, has_time_dimension=False):
    """
    Parameters
    ----------
        birdview: torch.Tensor<float> (C, H, W)
        n_classes: int
            number of total classes
        has_time_dimension: bool

    Returns
    -------
        birdview_label: (H, W)
    """
    # When a pixel contains two labels, argmax will output the first one that is encountered.
    # By reversing the order, we prioritise traffic lights over road.
    dim = 0
    if has_time_dimension:
        dim = 1
    birdview_label = torch.argmax(birdview.flip(dims=[dim]), dim=dim) ##通过这个flip函数，将多标签问题的重要性，排序了 red light > green light > pedestrian > vehicle > lane marking > road > background
    # We then re-normalise the classes in the normal order.
    birdview_label = (n_classes - 1) - birdview_label
    return birdview_label

def integer_to_binary(integer_array, n_bits):
    """
    Parameters
    ----------
        integer_array: np.ndarray<int32> (n,)
        n_bits: int

    Returns
    -------
        binary_array: np.ndarray<float32> (n, n_bits)

    """
    return (((integer_array[:, None] & (1 << np.arange(n_bits)))) > 0).astype(np.float32) ##将整数数组中的每个元素与二进制位的每一位进行位与操作


if __name__ == '__main__':
    # data = pickle.load(open('mile-data/pd_dataframe.pkl','rb'))
    # df = pd.DataFrame(data)
    # df.to_excel('data.xlsx',index=False)

    # print(data.columns)

    n_classes = 8 ##类别总数

    birdview = Image.open('outputs/2024-05-24/16-17-26/dataset/Town02/0004/birdview/birdview_000000000.png')
    birdview = np.asarray(birdview)
    #bugui print(birdview[:,:,0])
    h, w= birdview.shape
    print(birdview.shape)
    birdview = integer_to_binary(birdview.reshape(-1), n_classes).reshape(h, w, n_classes)
    #print(birdview[:,:,0])
    #img = Image.fromarray(np.uint8(birdview[:,:,0]))
    #print(img.shape)
    #img.show()
    print(birdview.shape)
    birdview = birdview.transpose((2, 0, 1))
    print('birdview.shape',birdview.shape)
    #print(birdview.max())
    birdview_label = calculate_birdview_labels(torch.from_numpy(birdview), n_classes).numpy()
    #print(birdview_label.max())
    print('birdview_label.shape:',birdview_label.shape)
    #print('birdview_label_none.shape:',birdview_label[None].shape)
    img = BIRDVIEW_COLOURS[birdview_label] ### 正如issue中所说，这里的birdview_label代表每一个像素点对应的类别，通过这个类别来找到对应的颜色
    # print(img)
    print('img_shape',img.shape)
    img = Image.fromarray(np.uint8(img))

    img.show()
    # img = Image.fromarray(np.uint8(birdview_label))
    # img.show()
    # print(birdview.shape)
    # for i in range(n_classes):
    #     img = Image.fromarray(np.uint8(birdview[i]))
    #     img.show()