import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

class ImageDatasetHpMask(Dataset):

    def __init__(self, dataset, height=256, width=128, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.h = height
        self.w = width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            img_path, pid, camid, msk_path = self.dataset[index]
            img = Image.open(img_path).convert('RGB')
            msk = Image.open(msk_path).convert('P')
            msk = msk.resize((128, 256))
            msk = np.array(msk)
            msk = torch.from_numpy(msk)
            msk = msk.unsqueeze(dim=0)

            if self.transform is not None:
                img = self.transform(img)


        except:
            print(index)

        return img, msk, pid, camid, img_path



class PRCC(object):

    rgb_dir = 'rgb'
    msk_dir = 'hp'
    cam2label = {'A': 0, 'B': 1, 'C': 2}

    def __init__(self, root='../datasets/prcc'):
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.rgb_dir)
        self.train_dir = os.path.join(self.dataset_dir,'train_sr')
        self.mask_dir = os.path.join(self.root,self.msk_dir)
        self.test_a_dir = os.path.join(self.dataset_dir,'test','A')
        self.test_b_dir = os.path.join(self.dataset_dir, 'test', 'B')
        self.test_c_dir = os.path.join(self.dataset_dir, 'test', 'C')

        train_data, train_data_ids = self._process_data(self.train_dir, os.path.join(self.mask_dir, 'train'), select=['A','B','C'], relabel=True)
        test_a_data, test_a_data_ids = self._process_data_test(self.test_a_dir, os.path.join(self.mask_dir, 'test', 'A'), camid='A', relabel=False)
        test_b_data, test_b_data_ids = self._process_data_test(self.test_b_dir, os.path.join(self.mask_dir, 'test', 'B'), camid='B', relabel=False)
        test_c_data, test_c_data_ids = self._process_data_test(self.test_c_dir, os.path.join(self.mask_dir, 'test', 'C'), camid='C', relabel=False)

        self.test_a_data = test_a_data
        self.test_b_data = test_a_data
        self.test_c_data = test_a_data
        self.test_a_data_ids = test_a_data_ids
        self.test_b_data_ids = test_b_data_ids
        self.test_c_data_ids = test_c_data_ids

        self.train_data = train_data
        self.train_data_ids = train_data_ids

        self.query_data = test_c_data           #query_data
        self.query_data_ids = test_c_data_ids

        self.gallery_data = test_a_data          #gallery_data
        self.gallery_data_ids = test_a_data_ids

        test_id_container = set()
        self.test_data = self.query_data + self.gallery_data
        for data in self.test_data:
            test_id_container.add(data[1])
        self.test_data_ids = len(test_id_container)

        print("PRCC loaded")
        print("dataset  |   ids |     imgs")
        print("train    | {:5d} | {:8d}".format(self.train_data_ids,len(self.train_data)))
        print("query    | {:5d} | {:8d}".format(self.query_data_ids, len(self.query_data)))
        print("gallery  | {:5d} | {:8d}".format(self.gallery_data_ids, len(self.gallery_data)))
        print("----------------------------------------")

    def _process_data(self, dir_path, msk_dir_path, select=['A','B','C'], relabel=False):

        persons = os.listdir(dir_path)
        pid2label = {pid: label for label, pid in enumerate(persons)}
        dataset = []
        for person in persons:
            person_path = os.path.join(dir_path, person)
            pics = os.listdir(person_path)
            for pic in pics:
                camid = pic.split('_')[0]
                if camid not in select:
                    continue
                camid = self.cam2label[camid]
                if relabel: pid = pid2label[person]
                else: pid = int(person)
                img_path = os.path.join(dir_path, person, pic)
                name = pic.split('.')[0] + '.png'
                msk_path = os.path.join(msk_dir_path, person, name)
                if not os.path.exists(msk_path):
                    continue

                dataset.append((img_path, pid, camid, msk_path))
        return dataset, len(persons)

    def _process_data_test(self, dir_path, msk_dir_path, camid, relabel=False):
        camid = self.cam2label[camid]
        persons = os.listdir(dir_path)
        pid2label = {pid: label for label, pid in enumerate(persons)}
        dataset = []
        for person in persons:
            person_path = os.path.join(dir_path, person)
            pics = os.listdir(person_path)
            for pic in pics:

                if relabel: pid = pid2label[person]
                else: pid = int(person)
                img_path = os.path.join(dir_path, person, pic)
                name = pic.split('.')[0] + '.png'
                msk_path = os.path.join(msk_dir_path, person, name)
                if not os.path.exists(msk_path):
                    continue

                dataset.append((img_path, pid, camid, msk_path))
        return dataset, len(persons)

class Celeb(object):

    msk_train_dir = 'train_hp'
    msk_query_dir = 'query_hp'
    msk_gallery_dir = 'gallery_hp'

    def __init__(self, root='../datasets/Celeb-reID'):
        self.root = root
        self.train_dir = os.path.join(self.root, 'train')
        self.query_dir = os.path.join(self.root, 'query')
        self.gallery_dir = os.path.join(self.root, 'gallery')
        self.mask_train_dir = os.path.join(self.root, self.msk_train_dir)
        self.mask_query_dir = os.path.join(self.root, self.msk_query_dir)
        self.mask_gallery_dir = os.path.join(self.root, self.msk_gallery_dir)

        train_data, train_data_ids = self._process_data(self.train_dir, self.mask_train_dir)
        query_data, query_data_ids = self._process_data(self.query_dir, self.mask_query_dir)
        gallery_data, gallery_data_ids = self._process_data(self.gallery_dir, self.mask_gallery_dir)

        self.train_data = train_data
        self.train_data_ids = train_data_ids

        self.query_data = query_data           #query_data
        self.query_data_ids = query_data_ids

        self.gallery_data = gallery_data          #gallery_data
        self.gallery_data_ids = gallery_data_ids

        test_id_container = set()
        self.test_data = self.query_data + self.gallery_data
        for data in self.test_data:
            test_id_container.add(data[1])
        self.test_data_ids = len(test_id_container)

        print("Celeb loaded")
        print("dataset  |   ids |     imgs")
        print("train    | {:5d} | {:8d}".format(self.train_data_ids,len(self.train_data)))
        print("query    | {:5d} | {:8d}".format(self.query_data_ids, len(self.query_data)))
        print("gallery  | {:5d} | {:8d}".format(self.gallery_data_ids, len(self.gallery_data)))
        print("----------------------------------------")

    def _process_data(self, dir_path, msk_dir_path):

        pics = os.listdir(dir_path)
        dataset = []
        pid = 0
        for pic in pics:
            img_path = os.path.join(dir_path, pic)
            pid = int(pic.split('_')[0]) - 1
            camid = 0
            name = pic.split('.')[0] + '.png'
            msk_path = os.path.join(msk_dir_path, name)
            dataset.append((img_path, pid, camid, msk_path))

        return dataset, pid + 1


__factory = {
    'prcc': PRCC,
    'celeb': Celeb,
}

def init_dataset(name,**kwargs):
    if name not in __factory.keys():
        raise KeyError("Invalid dataset, expected to be one of {}".format(__factory.keys()))
    return __factory[name](**kwargs)

