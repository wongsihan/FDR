import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
from scipy.io import loadmat
from scipy.fftpack import fft
import scipy.signal as signal

def STFT(fl):
    f, t, Zxx = signal.stft(fl, nperseg=64)
    img = np.abs(Zxx) / len(Zxx)
    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.title('CWT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return img

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def noise_rw(x, snr):
    snr1 = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2, axis=0) / len(x)
    npower = xpower / snr1
    noise = np.random.normal(0, np.sqrt(npower), x.shape)
    noise_data = x + noise
    return noise_data

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x


def pu_folders(CLASS_NUM):
    #root = "D:/Github/Few-shot-Transfer-Learning/data"
    root = "/root/envy/wsh/Few-shot-Transfer-Learning/data"
    # labels = ['KA01', 'KA03', 'KA05', 'KA07', 'KA08', 'KI01', 'KI03', 'KI07'] + ['K001'] + ['KA04', 'KB23',
    #                                                                                         'KB27', 'KI04']

    # #原始A2N
    labels = ['KA01', 'KA03', 'KA05', 'KA07', 'KA08', 'KI01', 'KI03', 'KI07'] + ['K001'] + ['KA04', 'KB23',
                                                                                            'KB27', 'KI04']
    random.seed(1)
    # random.shuffle(labels)

    print("finetune generator process executed___________________________________")
    folds = [os.path.join(root, label, label).replace('\\','/') for label in labels]

    samples = dict()
    train_files = []
    test_files = []
    train_labels = []
    test_labels = []
    for c in folds[:-CLASS_NUM]:

        name0 = 'N09_M07_F10_'
        name1 = c.split('/')[-1] + '_'
        temps = [os.path.join(c, name0 + name1 + str(x)).replace('\\','/') for x in range(1, 21)]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]

        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('\\')[-1]][0][0][2][0][6][2][0]
            data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
            if i == 0:
                file = data1
            else:
                file = np.vstack(
                    [file, data1])

        train_labels.append(get_class(temp))
        train_files.append(file)
    for c in folds[-CLASS_NUM:]:

        name0 = 'N09_M07_F10_'
        name1 = c.split('\\')[-1] + '_'
        temps = [os.path.join(c, name0 + name1 + str(x)) for x in range(1, 21)]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]
        # self.train_roots += train_part
        # self.test_roots += test_part

        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('\\')[-1]][0][0][2][0][6][2][0]
            data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
            if i == 0:
                file = data1
            else:
                file = np.vstack(
                    [file, data1])

        test_labels.append(get_class(temp))
        test_files.append(file)

    metatrain_folders = list(zip(train_files, train_labels))
    metatest_folders = list(zip(test_files, test_labels))
    return metatrain_folders, metatest_folders


def get_class(sample):
    return sample.split('\\')[-2]


class trainTask(object):
    def __init__(self, character_folders):
        self.character_folders = character_folders
        self.train_files = []
        self.train_labels = []
        class_folders = self.character_folders
        index = 0
        for class_folder in class_folders:
            (file, label) = class_folder
            np.random.shuffle(file)
            self.train_files += list(file)
            self.train_labels += [index for i in range(file.shape[0])]
            index += 1


class finetuneTask(object):
    def __init__(self, character_folders, train_num):
        self.character_folders = character_folders
        self.train_num = train_num
        self.train_files = []
        self.test_files = []
        self.train_labels = []
        self.test_labels = []
        class_folders = self.character_folders
        index = 0
        for class_folder in class_folders:
            (file, label) = class_folder
            np.random.shuffle(file)
            self.train_files += list(file[:self.train_num, :])
            self.test_files += list(file[self.train_num:, :])
            self.train_labels += [index for i in range(train_num)]
            self.test_labels += [index for i in range(file.shape[0] - train_num)]
            index += 1


class pu_train(Dataset):

    def __init__(self, character_folders,class_num,snr):
        self.character_folders = character_folders
        self.train_files = []
        self.train_labels = []
        self.snr = snr
        class_folders = self.character_folders
        index = 0
        for class_folder in class_folders[:class_num]:
            (file, label) = class_folder
            np.random.shuffle(file)
            self.train_files += list(file)
            self.train_labels += [index for i in range(file.shape[0])]
            index += 1
        np.array(self.train_labels)

    def __getitem__(self, idx):
        image = self.train_files[idx]
        if self.snr == -100:
            image = abs(fft(image - np.mean(image)))[0:1024]
        else:
            image = noise_rw(image, self.snr)
            image = abs(fft(image - np.mean(image)))[0:1024]
        image = image[0:1024].reshape([1, 1024])
        label = self.train_labels[idx]
        return image, np.int64(label)

    def __len__(self):
        return len(self.train_files)


class pu_test(Dataset):

    def __init__(self, task, split, snr):
        self.split = split
        self.task = task
        self.image_files = self.task.train_files if self.split == 'support' else self.task.test_files
        self.labels = np.array(self.task.train_labels if self.split == 'support' else self.task.test_labels).reshape(-1)
        self.snr = snr
    def __getitem__(self, idx):
        image = self.image_files[idx]
        if self.snr == -100:
            image = abs(fft(image - np.mean(image)))[0:1024]
        else:
            image = noise_rw(image, self.snr)
            image = abs(fft(image - np.mean(image)))[0:1024]
        image = image[0:1024].reshape([1, 1024])
        label = self.labels[idx]
        return image, np.int64(label)

    def __len__(self):
        return len(self.image_files)


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_train_loader(metatrain_character_folders, batchsize, class_num,snr):
    dataset = pu_train(metatrain_character_folders, class_num,snr)
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    return loader


def get_test_loader(task, num_per_class=1, split='support', num_classes=1,snr = -100):
    dataset = pu_test(task, split, snr)
    if split == 'support':
        loader = DataLoader(dataset, batch_size=num_classes, shuffle=True)
    elif split == 'test':
        loader = DataLoader(dataset, batch_size=num_per_class, shuffle=True)
    return loader
