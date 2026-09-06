import io
import os

import cv2
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """


def get_datalist(data_dir, data_path, max_len):
    """
    Get training and validation data list
    :param data_dir: Dataset root directory
    :param data_path: Dataset file list, each file contains 'path/to/img\tlabel'
    :return:
    """
    train_data = []
    if isinstance(data_path, list):
        for p in data_path:
            train_data.extend(get_datalist(data_dir, p, max_len))
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(),
                             desc=f'load data from {data_path}'):
                line = (line.strip('\n').replace('.jpg ', '.jpg\t').replace(
                    '.png ', '.png\t').split('\t'))
                if len(line) > 1:
                    img_path = os.path.join(data_dir, line[0].strip(' '))
                    label = line[1]
                    if len(label) > max_len:
                        continue
                    if os.path.exists(
                            img_path) and os.path.getsize(img_path) > 0:
                        train_data.append([str(img_path), label])
    return train_data


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    return imgH * imgW != 0


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(data_list, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=2147483648)
    cache = {}
    cnt = 1
    for imagePath, label in tqdm(data_list,
                                 desc=f'make dataset, save to {outputPath}'):
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            buf = io.BytesIO(imageBin)
            w, h = Image.open(buf).size
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print(f'{imagePath} is not a valid image')
                    continue
            except:
                continue

        imageKey = b'image-%09d' % cnt
        labelKey = b'label-%09d' % cnt
        whKey = b'wh-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[whKey] = (str(w) + '_' + str(h)).encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1
    nSamples = cnt - 1
    cache[b'num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    data_dir = './dataset/rec'

    label_file_list = [
        os.path.join(data_dir, 'train_labels.txt'),
        os.path.join(data_dir, 'val_labels.txt'),
        os.path.join(data_dir, 'test_labels.txt')
    ]
    save_path_root = os.path.join(data_dir, 'lmdb_data')

    for data_list in label_file_list:
        file_name = os.path.basename(data_list).split('.')[0].replace('_labels', '')
        save_path = os.path.join(save_path_root, file_name)

        # Check if the text file exists
        if not os.path.exists(data_list):
            print(f"File not found: {data_list}. Skipping...")
            continue

        os.makedirs(save_path, exist_ok=True)
        print(f"Creating LMDB dataset at: {save_path}")

        train_data_list = get_datalist(data_dir, data_list, 800)

        createDataset(train_data_list, save_path)
