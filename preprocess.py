from skimage import io
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.transform import resize
import os
import numpy as np

path_to_data = os.path.join('.', 'data')
path_to_jpg = os.path.join(path_to_data, 'raw_jpg')
# path_to_preprocessed = os.path.join(path_to_data, 'preprocessed')
# image_height = 72
# image_width = 112
path_to_preprocessed = os.path.join(path_to_data, 'preprocessed')
image_height = 128
image_width = 128
binary_thresh = 3
gaussian_sigma = 5


def preprocess(image):
    image = image[20:-25, 25:-25, :]#image->numpy数组[长，宽，通道] 裁剪
    assert isinstance(image, np.ndarray)#判断image是否是np.ndarray类型。断点，测试语句。
    grayscale = rgb2gray(image) #使用scikit-image工具将rgb转化成gray图
    grayscale = gaussian(grayscale, sigma=gaussian_sigma)#高斯平滑：可以去噪

    binary = grayscale * 255 > binary_thresh# 二值化：非黑即白

    labelled = label(binary)#labelled打标签之后的数据
    regions = regionprops(labelled)#二值图中各个连通区域
    areas = [region.area for region in regions]
    arg = int(np.argmax(areas))
    h1, w1, h2, w2 = regions[arg].bboxq
    extracted = image[h1: h2, w1: w2, :]
    return (resize(extracted, (image_height, image_width)) * 255).astype(np.uint8)#使用resize函数进行在extracted上缩放


if __name__ == '__main__':
    filenames = os.listdir(path_to_jpg)
    for filename in filenames:
        im = io.imread(os.path.join(path_to_jpg, filename), as_gray=False)
        processed = preprocess(im)
        io.imsave(os.path.join(path_to_preprocessed, filename), processed)
        print('Successfully processed %s' % filename)
