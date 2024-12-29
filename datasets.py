# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import numpy as np

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.utils.data import Dataset

from augment import three_augment, MultiCrop


class INatDataset(ImageFolder):
    def __init__(
        self,
        root,
        train=True,
        year=2018,
        transform=None,
        target_transform=None,
        category="name",
        loader=default_loader,
    ):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, "categories.json")) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter["annotations"]:
            king = []
            king.append(data_catg[int(elem["category_id"])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data["images"]:
            cut = elem["file_name"].split("/")
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

# Code modified for DeiT-LT
class IMBALANCECIFAR10(datasets.CIFAR10):
    cls_num = 10

    def __init__(
        self,
        root,
        imb_type="exp",
        imb_factor=0.01,
        rand_number=0,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super(IMBALANCECIFAR10, self).__init__(
            root, train, transform, target_transform, download
        )
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend(
                [
                    the_class,
                ]
                * the_img_num
            )
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

# Code modified for DeiT-LT
class KD_IMBALANCECIFAR10(datasets.CIFAR10):
    cls_num = 10

    def __init__(
        self,
        root,
        imb_type="exp",
        imb_factor=0.01,
        rand_number=0,
        train=True,
        student_transform=None,
        teacher_transform=None,
        target_transform=None,
        download=False,
    ):
        super(KD_IMBALANCECIFAR10, self).__init__(
            root, train, target_transform=target_transform, download=download
        )
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend(
                [
                    the_class,
                ]
                * the_img_num
            )
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.student_transform is not None:
            student_img = self.student_transform(img)
            # teacher_img = self.teacher_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return student_img, target

# Code modified for DeiT-LT
class KD_IMBALANCECIFAR100(KD_IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    cls_num = 100

# Code modified for DeiT-LT
class KD_IMAGENETLT(Dataset):
    num_classes = 1000

    def __init__(self, root, txt, student_transform=None, teacher_transform=None):
        self.img_path = []
        self.targets = []
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        cls_num_list_old = [
            np.sum(np.array(self.targets) == i) for i in range(self.num_classes)
        ]  # 1000

        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i

        self.reverse_class_map = [0 for i in range(self.num_classes)]
        for i in range(len(self.class_map)):
            self.reverse_class_map[self.class_map[i]] = i

        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [
            np.sum(np.array(self.targets) == i) for i in range(self.num_classes)
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")
        if self.student_transform is not None:
            student_sample = self.student_transform(sample)
            # teacher_sample = self.teacher_transform(sample)
        return student_sample, target

    def get_cls_num_list(self):
        return self.cls_num_list

# Code modified for DeiT-LT
class IMAGENETLT_EVAL(Dataset):
    num_classes = 1000

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.targets = np.array(self.class_map)[self.targets].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target 

# Code modified for DeiT-LT
class LT_Dataset_CMO(Dataset):
    def __init__(self, root, txt, transform=None, use_randaug=False):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.use_randaug = use_randaug
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.use_randaug:
            r = random.random()
            if r < 0.5:
                sample = self.transform[0](sample)
            else:
                sample = self.transform[1](sample)
        else:
            if self.transform is not None:
                sample = self.transform(sample)

        # return sample, label, path
        return sample, label

# Code modified for DeiT-LT
class KD_INAT2018(Dataset):
    num_classes = 8142

    def __init__(
        self, root, txt, class_map, student_transform=None, teacher_transform=None
    ):
        self.img_path = []
        self.targets = []
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        # print("Preparing the cls_num_list_old")
        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]
        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i

        
        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [
            np.sum(np.array(self.targets) == i) for i in range(self.num_classes)
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")
        if self.student_transform is not None:
            student_sample = self.student_transform(sample)
        return student_sample, target

    def get_cls_num_list(self):
        return self.cls_num_list

# Code modified for DeiT-LT
class INAT2018_EVAL(Dataset):
    num_classes = 8142

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.targets = np.array(self.class_map)[self.targets].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]
        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def build_dataset(is_train, args, class_map=None):
    if args.student_transform == 0:
        student_transform = build_transform_deit(is_train, args)
    elif args.student_transform == 1:
        student_transform = build_transform_ldam(is_train, args)
    elif args.student_transform == 2:
        student_transform = build_transform_val(args)

    if args.teacher_transform == 0:
        teacher_transform = build_transform_deit(is_train, args)
    elif args.teacher_transform == 1:
        teacher_transform = build_transform_ldam(is_train, args)
    elif args.teacher_transform == 2:
        teacher_transform = build_transform_val(args)

    if args.data_set == "CIFAR100":
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform, download=True
        )
        nb_classes = 100
    elif args.data_set == "CIFAR10":
        dataset = datasets.CIFAR10(
            args.data_path, train=is_train, transform=transform, download=True
        )
        nb_classes = 10
    elif args.data_set == "IMNET":
        root = os.path.join(args.data_path, "train" if is_train else "val")
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "INAT":
        dataset = INatDataset(
            args.data_path,
            train=is_train,
            year=2018,
            category=args.inat_category,
            transform=transform,
        )
        nb_classes = dataset.nb_classes
    elif args.data_set == "INAT19":
        dataset = INatDataset(
            args.data_path,
            train=is_train,
            year=2019,
            category=args.inat_category,
            transform=transform,
        )
        nb_classes = dataset.nb_classes

    elif args.data_set == "CIFAR10LT":
        if is_train:
            dataset = KD_IMBALANCECIFAR10(
                root=args.data_path,
                imb_type=args.imb_type,
                imb_factor=args.imb_factor,
                rand_number=0,
                train=True,
                download=True,
                student_transform=student_transform,
                teacher_transform=teacher_transform,
            )
        else:
            dataset = datasets.CIFAR10(
                args.data_path,
                train=is_train,
                transform=student_transform,
                download=True,
            )
        nb_classes = 10

    elif args.data_set == "CIFAR100LT":
        if is_train:
            dataset = KD_IMBALANCECIFAR100(
                root=args.data_path,
                imb_type=args.imb_type,
                imb_factor=args.imb_factor,
                rand_number=0,
                train=True,
                download=True,
                student_transform=student_transform,
                teacher_transform=teacher_transform,
            )
        else:
            dataset = datasets.CIFAR100(
                args.data_path,
                train=is_train,
                transform=student_transform,
                download=True,
            )
        nb_classes = 100

    elif args.data_set == "IMAGENETLT":
        if is_train:
            dataset = KD_IMAGENETLT(
                root=args.data_path,
                txt="./data_txt/Imagenet_LT_train.txt",
                student_transform=student_transform,
                teacher_transform=teacher_transform,
            )
        else:
            # train_instance = KD_IMAGENETLT(root = args.data_path, txt = './data_txt/Imagenet_LT_train.txt', student_transform = student_transform, teacher_transform = teacher_transform)
            dataset = IMAGENETLT_EVAL(
                root=args.data_path,
                txt="./data_txt/Imagenet_LT_test.txt",
                class_map=class_map,
                transform=student_transform,
            )
            # dataset = LT_Dataset_CMO(args.data_path, './data_txt/Imagenet_LT_test.txt', student_transform)
        nb_classes = 1000

    elif args.data_set == "INAT18":
        if is_train:
            dataset = KD_INAT2018(
                root=args.data_path,
                txt="./data_txt/iNaturalist18_train.txt",
                class_map=class_map,
                student_transform=student_transform,
                teacher_transform=teacher_transform,
            )
        else:
            # train_instance = KD_INAT2018(root = args.data_path, txt = './data_txt/iNaturalist18_train.txt', student_transform = student_transform, teacher_transform = teacher_transform)
            dataset = INAT2018_EVAL(
                root=args.data_path,
                txt="./data_txt/iNaturalist18_val.txt",
                class_map=class_map,
                transform=student_transform,
            )
        nb_classes = 8142
    return dataset, nb_classes

# Code modified for DeiT-LT
def build_transform_deit(is_train, args):
    size = args.input_size
    resize_im = size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train

        # Three-augment from DeiT-3
        if args.ThreeAugment:
            transform = three_augment(args)

        # MultiCrop from DINO
        elif args.multi_crop:
            transform = MultiCrop(
                args.global_crops_scale,
                args.local_crops_scale,
                args.local_crops_number,
                args.rand_aug,
            )
        else:
            transform = create_transform(
                input_size=size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(size, padding=4)
        return transform

    print("Deit val")
    t = []
    if resize_im:
        new_size = int((256 / 224) * size)
        t.append(
            transforms.Resize(
                new_size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    # t.append(transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]))

    return transforms.Compose(t)

# Code modified for DeiT-LT
def build_transform_ldam(is_train, args):
    size = args.input_size
    if not is_train:
        transform = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
        return transform

    transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=3),
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    return transform

# Code modified for DeiT-LT
def build_transform_val(args):
    print("This trnasform")
    size = args.input_size
    transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    # transforms.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262]),])
    return transform
