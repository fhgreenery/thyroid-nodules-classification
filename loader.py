import xml.etree.ElementTree as et
import os
import random
from skimage import io
import numpy as np
# from tensorflow import keras
import keras

path_to_data = os.path.join('.', 'data')#把目录和文件名合成一个路径
path_to_xml = os.path.join(path_to_data, 'xml')
path_to_raw_jpg = os.path.join(path_to_data, 'raw_jpg')
path_to_preprocessed = os.path.join(path_to_data, 'preprocessed')
path_to_inception = os.path.join(path_to_data, 'inception')

fields = ['number', 'age', 'sex', 'composition',
          'echogenicity', 'margins', 'calcifications', 'tirads']


def load(mode='preprocessed'):
    path_to_images = path_to_preprocessed
    if mode == 'raw':
        path_to_images = path_to_raw_jpg
    if mode == 'inception':
        path_to_images = path_to_inception
    xml_filenames = os.listdir(path_to_xml) #返回path指定的文件夹包含的文件或文件夹的名字的列表
    image_filenames = os.listdir(path_to_images)
    cases = []#列表
    for filename in xml_filenames:
        tree = et.parse(os.path.join(path_to_xml, filename))#指定的xml文件
        root = tree.getroot()
        case = {}#字典类型 9个键值对+{label:true/false}
        for field in fields:
            case[field] = root.find(field).text
        case_image_filenames = list(filter(lambda x: x.startswith(str(case['number']) + '_'), image_filenames))
        for image_filename in case_image_filenames:
            image = io.imread(os.path.join(path_to_images, image_filename))#numpy数组格式，imread读取的数据类型为uint8类型，范围为[0,255]
            case['image'] = image
            cases.append(case)
    return cases


def label(data): #为每一条数据打标签 false or true
    data = list(filter(lambda x: x['tirads'] is not None, data)) #得到tirads不空的数据
    for item in data:
        item['label'] = False if item['tirads'] == '2' or item['tirads'] == '3' else True
    return data


def make_dataset(data):
    data = label(data)
    benign_cases = list(filter(lambda x: not x['label'], data))#
    malign_cases = list(filter(lambda x: x['label'], data))
    random.shuffle(benign_cases) #shuffle()方法将序列的所有元素随机排序。
    random.shuffle(malign_cases)
    #划分训练集和测试集
    benign_limit = int(len(benign_cases) * 0.1)
    malign_limit = int(len(malign_cases) * 0.1)
    train_benign = benign_cases[0: -benign_limit]#benign_cases:90%的数据用作训练；10%的数据用作测试。
    test_benign = benign_cases[-benign_limit:]
    train_malign = malign_cases[0: -malign_limit]
    test_malign = malign_cases[-malign_limit:]
    train = train_benign + train_malign#保证训练集和测试集中的良恶性比例与原数据相同
    test = test_benign + test_malign
    random.shuffle(train)
    random.shuffle(test)
    #train和test是一个包含9个键值对的字典,模型训练只需要images和labels
    train_labels = keras.utils.to_categorical([x['label'] for x in train], 2)#将类别向量转化成0，1值
    train_images = np.array([x['image'] for x in train]).astype('float32') / 255 #将数据转化成浮点型，并归一化，将像素值转化到[0,1]之间
    test_labels = keras.utils.to_categorical([x['label'] for x in test], 2)
    test_images = np.array([x['image'] for x in test]).astype('float32') / 255

    return (train_images, train_labels), (test_images, test_labels)
