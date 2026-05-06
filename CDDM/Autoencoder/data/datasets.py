# import cv2
import os
import sys

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms, datasets

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class FlatImageFolder(data.Dataset):
    """
    目录下直接放若干图片（无类别子文件夹），与标准 DIV2K_train_HR / DIV2K_valid_HR 布局一致。
    torchvision ImageFolder 要求 root/class_name/xxx.png，故单独实现。
    """

    def __init__(self, root, transform=None):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.transform = transform
        self.paths = []
        if not os.path.isdir(self.root):
            raise FileNotFoundError(self.root)
        for name in sorted(os.listdir(self.root)):
            p = os.path.join(self.root, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in _IMG_EXTS:
                self.paths.append(p)
        if not self.paths:
            raise FileNotFoundError(
                "目录中未找到图片（支持 {}）: {}".format(", ".join(sorted(_IMG_EXTS)), self.root)
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

NUM_DATASET_WORKERS = 16  # 未在 config 中指定 num_workers 时的默认
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


def get_loader(config):
    load_val = bool(getattr(config, "load_val_data", True))
    if config.dataset == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        transform_test = transforms.Compose([
            transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root=config.train_data_dir,
                                         train=True,
                                         transform=transform_train,
                                         download=False)

        test_dataset = None
        if load_val:
            test_dataset = datasets.CIFAR10(
                root=config.test_data_dir,
                train=False,
                transform=transform_test,
                download=False,
            )
    elif config.dataset == "DIV2K":
        transform_train = transforms.Compose([
            transforms.RandomCrop((config.image_dims[1], config.image_dims[2])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])

        transform_test = transforms.Compose([
            transforms.CenterCrop((config.image_dims[1], config.image_dims[2])),
            transforms.ToTensor()])

        train_dataset = FlatImageFolder(
            root=config.train_data_dir,
            transform=transform_train,
        )
        test_dataset = None
        if load_val:
            test_dataset = FlatImageFolder(
                root=config.test_data_dir,
                transform=transform_test,
            )
    elif config.dataset == "CelebA":
        transform_train = transforms.Compose([
            transforms.RandomCrop((config.image_dims[1], config.image_dims[2])),
            transforms.ToTensor()])

        transform_test = transforms.Compose([
            transforms.CenterCrop((config.image_dims[1], config.image_dims[2])),
            transforms.ToTensor()])
        train_dataset = datasets.ImageFolder(root=config.train_data_dir,
                                             transform=transform_train)

        test_dataset = None
        if load_val:
            test_dataset = datasets.ImageFolder(
                root=config.test_data_dir,
                transform=transform_test,
            )

    nw = int(getattr(config, "num_workers", NUM_DATASET_WORKERS))
    pin = bool(getattr(config, "pin_memory", True))
    pw = bool(getattr(config, "persistent_workers", False)) and nw > 0
    pf = 2 if nw > 0 else None
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=pw,
        prefetch_factor=pf,
        batch_size=config.batch_size,
        worker_init_fn=worker_init_fn_seed,
        shuffle=True,
        drop_last=True,
    )
    if test_dataset is not None:
        vnw = int(getattr(config, "val_num_workers", max(1, nw // 2)))
        vpw = bool(getattr(config, "persistent_workers", False)) and vnw > 0
        vpf = 2 if vnw > 0 else None
        test_loader = data.DataLoader(
            dataset=test_dataset,
            batch_size=config.test_batch,
            shuffle=False,
            num_workers=vnw,
            pin_memory=pin,
            persistent_workers=vpw,
            prefetch_factor=vpf,
        )
    else:
        test_loader = None

    return train_loader, test_loader


class config():
    dataset = "CelebA"
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    device_ids = [0]
    if_sample = False
    # logger
    print_step = 39
    plot_step = 10000
    # filename = datetime.now().__str__()[:-16]
    models = 'E:\code\DDPM\SemDiffusion\Autoencoder\history'
    logger = None
    equ = "MMSE"
    # training details
    normalize = False
    learning_rate = 0.0001
    epoch = 20

    save_model_freq = 20
    if dataset == "CIFAR10":
        image_dims = (3, 32, 32)
        train_data_dir = r"E:\code\DDPM\DenoisingDiffusionProbabilityModel-ddpm--main\DenoisingDiffusionProbabilityModel-ddpm--main\CIFAR10"
        test_data_dir = r"E:\code\DDPM\DenoisingDiffusionProbabilityModel-ddpm--main\DenoisingDiffusionProbabilityModel-ddpm--main\CIFAR10"
    elif dataset == "DIV2K":
        image_dims = (3, 256, 256)
        train_data_dir = r"D:\dateset\DIV2K\DIV2K_train_HR"
        test_data_dir = r"D:\dateset\DIV2K\DIV2K_valid_HR"
    elif dataset == "CelebA":
        image_dims = (3, 128, 128)
        train_data_dir = r"D:\dateset\CelebA\Img\trainset"
        test_data_dir = r"D:\dateset\CelebA\Img\validset"
    batch_size = 1
    # batch_size = 100
    downsample = 4


if __name__ == "__main__":
    train_loader, test_loader = get_loader(config)
    image = next(iter(train_loader))[0]
    print(image)
