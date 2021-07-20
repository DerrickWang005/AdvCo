import os.path as osp
import lmdb
import numpy as np
import pyarrow as pa
import six
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, file_dir, transform, ignore_label=True, linear=False):
        self.db_path = file_dir
        self.linear = linear
        self.env = None
        if 'train' in self.db_path:
            self.length = 1281167
        elif 'val' in self.db_path:
            self.length = 50000
        else:
            raise NotImplementedError
        self.transform = transform
        self.ignore_label = ignore_label

    def _init_db(self):
        self.env = lmdb.open(self.db_path,
                             subdir=osp.isdir(self.db_path),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()

        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = loads_pyarrow(byteflow)

        # load img.
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        # load label.
        target = unpacked[1]
        # transform
        if self.linear:
            img = self.transform(img)
        else:
            img_q, img_k = self.transform(img)
            img = [img_q, img_k]

        if self.ignore_label:
            return img
        else:
            return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

    def get_length(self):
        return self.length

    def get_sample(self, idx):
        return self.__getitem__(idx)
