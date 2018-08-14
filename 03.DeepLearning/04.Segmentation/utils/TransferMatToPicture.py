'''
将Benchmark中的mat图像转化成png图片
'''
import os
import numpy as np

from scipy.io import loadmat
from scipy.misc import toimage, imsave, imread
from tqdm import tqdm
from shutil import move,copy

VOC_2012_ROOT_PATH = os.path.join("/input", "VOC2012")
BENCHMARK_ROOT_PATH = os.path.join("/input", "benchmark_RELEASE")
PHASE = ["train.txt", "val.txt"]

def encode_segmap(mask):
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask
def get_pascal_labels():
    return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0],
                       [0,0,128], [128,0,128], [0,128,128], [128,128,128],
                       [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                       [64,0,128], [192,0,128], [64,128,128], [192,128,128],
                       [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],
                       [0,64,128]])

TARGET_SEGMETATION_FOLDER = os.path.join(VOC_2012_ROOT_PATH, "SegmentationImages")

if os.path.exists(TARGET_SEGMETATION_FOLDER) == True:
    os.rmdir(TARGET_SEGMETATION_FOLDER)

os.mkdir(TARGET_SEGMETATION_FOLDER)

'''
encode VOC 2012 Segmentation
'''
VOC_2012_SEGMETATION_INFO_FILE_FOLDER = os.path.join(VOC_2012_ROOT_PATH, "ImageSets", "Segmentation")
for phase in PHASE:
    file_path = os.path.join(VOC_2012_SEGMETATION_INFO_FILE_FOLDER, phase)
    lines = tuple(open(file_path, 'r'))
    lines = [id_.rstrip() for id_ in lines]

    for ii in tqdm(lines):
        fname = ii + '.png'
        lbl_path = os.path.join(VOC_2012_ROOT_PATH, 'SegmentationClass', fname)
        lbl = encode_segmap(imread(lbl_path))
        lbl = toimage(lbl, high=lbl.max(), low=lbl.min())
        imsave(os.path.join(TARGET_SEGMETATION_FOLDER, fname), lbl)

'''
Trasfer Benchmark mat Segmentation To VOC 2012 Folder
'''
BENCHMARK_DATASET_ROOT_PATH = os.path.join(BENCHMARK_ROOT_PATH, "dataset")
BENCHMARK_DATASET_SEG_PATH = os.path.join(BENCHMARK_DATASET_ROOT_PATH, "cls")
for phrase in PHASE:
    file_path = os.path.join(BENCHMARK_DATASET_ROOT_PATH, phrase)
    lines = tuple(open(file_path, 'r'))
    lines = [id_.rstrip() for id_ in lines]

    for line in tqdm(lines):
        lbl_path = os.path.join(BENCHMARK_DATASET_SEG_PATH, line + '.mat')
        data = loadmat(lbl_path)
        lbl = data['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl = toimage(lbl, high=lbl.max(), low=lbl.min())
        imsave(os.path.join(TARGET_SEGMETATION_FOLDER, line + '.png'), lbl)

'''
Move Benchmark two information files to VOC 2012 Folder
'''
for phase in PHASE:
    file_path = os.path.join(BENCHMARK_DATASET_ROOT_PATH, phase)
    move(file_path, os.path.join(VOC_2012_SEGMETATION_INFO_FILE_FOLDER, "benchmark_%s" % phase))


