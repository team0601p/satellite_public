import torch
import pandas as pd
import cv2
import albumentations
import numpy as np
import argparse
from albumentations.pytorch import ToTensorV2
from utils import rle_decode

# These were deprecated because of data loading speed

"""
class SatelliteTrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, csv_file, start = 0, end = None, transform=albumentations.Compose([albumentations.RandomCrop(224, 224), albumentations.Normalize(), ToTensorV2()])):
        print('loading train dataset')
        self.dataset_path = dataset_path
        self.data_csv = pd.read_csv(csv_file)
        self.start = start
        self.end = end if end != None else len(self.data_csv)
        self.data = []
        self.transform = transform
        self.load_data()
    
    def load_data(self):
        for i in range(self.start, self.end):
            if i % 100 == 0:
                print('loading', i - self.start, '/', self.end - self.start)
            img_path = self.data_csv.iloc[i, 1]
            image = cv2.imread(self.dataset_path + img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = rle_decode(self.data_csv.iloc[i, 2], (image.shape[0], image.shape[1]))
            self.data.append([image, mask])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        augmented = self.transform(image=self.data[idx][0], mask=self.data[idx][1])
        image = augmented['image']
        mask = augmented['mask']
        return image, mask

class SatelliteValidDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, csv_file, start = 0, end = None, transform=[]):
        print('loading valid dataset')
        self.dataset_path = dataset_path
        self.data_csv = pd.read_csv(csv_file)
        self.start = start
        self.end = end if end != None else len(self.data_csv)
        self.data = []
        self.transform = transform
        self.load_data()
    
    def load_data(self):
        for i in range(self.start, self.end):
            if i % 100 == 0:
                print('loading', i - self.start, '/', self.end - self.start)
            img_path = self.data_csv.iloc[i, 1]
            image = cv2.imread(self.dataset_path + img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = rle_decode(self.data_csv.iloc[i, 2], (image.shape[0], image.shape[1]))
            for transform in self.transform:
                augmented = transform(image=image, mask=mask)
                self.data.append([augmented['image'], augmented['mask']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

"""

class SatelliteTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, csv_file, start = 0, end = None, transform=albumentations.Compose([albumentations.Normalize(), ToTensorV2()])):
        print('loading test dataset')
        self.dataset_path = dataset_path
        self.data_csv = pd.read_csv(csv_file)
        self.start = start
        self.end = end if end != None else len(self.data_csv)
        self.data = []
        self.transform = transform
        self.load_data()
    
    def load_data(self):
        for i in range(self.start, self.end):
            if i % 100 == 0:
                print('loading', i - self.start, '/', self.end - self.start)
            img_path = self.data_csv.iloc[i, 1]
            image = cv2.imread(self.dataset_path + img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.data.append(self.transform(image=image)['image'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SatelliteTestDataset_No_Ram(torch.utils.data.Dataset):
    def __init__(self, dataset_path, csv_file, start = 0, end = None, transform=albumentations.Compose([albumentations.Normalize(), ToTensorV2()])):
        print('loading test dataset')
        self.dataset_path = dataset_path
        self.data_csv = pd.read_csv(dataset_path + csv_file)
        self.start = start
        self.end = end if end != None else len(self.data_csv)
        self.data = []
        self.transform = transform
#         self.load_data()
    
#     def load_data(self):
#         for i in range(self.start, self.end):
#             if i % 100 == 0:
#                 print('loading', i - self.start, '/', self.end - self.start)
#             img_path = self.data_csv.iloc[i, 1]
#             image = cv2.imread(self.dataset_path + img_path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             self.data.append(self.transform(image=image)['image'])

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        img_path = self.data_csv.iloc[idx, 1]
        image = cv2.imread(self.dataset_path + img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)['image']

class SatelliteTrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform=albumentations.Compose([albumentations.RandomCrop(224, 224), albumentations.Normalize(), ToTensorV2()])):
        print('loading train dataset')
        self.images = np.load(dataset_path + 'train_images.npy')
        self.masks = np.load(dataset_path + 'train_masks.npy')
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        augmented = self.transform(image = self.images[idx], mask = self.masks[idx])
        return augmented["image"], augmented["mask"]
    
class SatelliteTrainDataset_Aug(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform1, transform2):
        print('loading train dataset')
        self.images = np.load(dataset_path + 'train_images.npy')
        self.masks = np.load(dataset_path + 'train_masks.npy')
        self.transform1 = transform1
        self.transform2 = transform2
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        choose = np.random.rand(1)[0]
        if choose <= 0.5:
            augmented = self.transform1(image = self.images[idx], mask = self.masks[idx])
        else:
            augmented = self.transform2(image = self.images[idx], mask = self.masks[idx])
        return augmented["image"], augmented["mask"]

class SatelliteValidDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform=ToTensorV2()):
        print('loading valid dataset')
        self.images = np.load(dataset_path + 'valid_images.npy')
        self.masks = np.load(dataset_path + 'valid_masks.npy')
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        augmented = self.transform(image = self.images[idx], mask = self.masks[idx])
        return augmented["image"], augmented["mask"]

# class SatelliteTestDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset_path, transform=ToTensorV2()):
#         print('loading test dataset')
#         self.images = np.load(dataset_path + 'test_images.npy')
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         augmented = self.transform(image = self.images[idx])
#         return augmented["image"]

def get_dataloaders(args):
    # args <- dataset_path, batch_size, num_workers are required
    # train len : 7140
    # valid len : 60640

    train_split = int(7140 * 0.95)

    # basic
    

    if args.augment:
        train_transform = albumentations.Compose([albumentations.augmentations.geometric.resize.RandomScale(scale_limit=0.2, interpolation=1, always_apply=False, p=0.5),
                                                  albumentations.augmentations.geometric.rotate.RandomRotate90(),
                                                  albumentations.RandomCrop(224, 224),
                                                  albumentations.VerticalFlip(p=0.5),
                                                  albumentations.HorizontalFlip(p=0.5),
                                                  albumentations.Normalize(),
                                                  ToTensorV2()])
    
    else:
        train_transform = albumentations.Compose([albumentations.RandomCrop(224, 224), albumentations.Normalize(), ToTensorV2()])
        

    valid_transform = []
    
    for i in range(0, 1024, 128):
        for j in range(0, 1024, 128):
            x_start = 800 if i + 224 > 1024 else i
            y_start = 800 if j + 224 > 1024 else j
            valid_transform.append(albumentations.Compose([albumentations.Crop(x_start, y_start, x_start + 224, y_start + 224), albumentations.Normalize(), ToTensorV2()]))

    
    validdataset = SatelliteValidDataset(dataset_path=args.dataset_path, transform=valid_transform)
    testdataset = SatelliteTestDataset(dataset_path=args.dataset_path)
    traindataset = SatelliteTrainDataset(dataset_path=args.dataset_path, transform=train_transform)

    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return trainloader, validloader, testloader

def get_train_valid_loader(args):
    # args <- dataset_path, batch_size, num_workers are required
    # train len : 7140
    # valid len : 60640

    train_split = int(7140 * 0.95)
    if args.augment:
        train_transform = albumentations.Compose([albumentations.augmentations.geometric.resize.RandomScale(scale_limit=0.2, interpolation=1, always_apply=False, p=0.5),
                                                   albumentations.RandomCrop(224, 224),
                                                   albumentations.VerticalFlip(p=0.5),
                                                   albumentations.HorizontalFlip(p=0.5),
                                                   albumentations.Normalize(),
                                                   ToTensorV2()])
    
    else:
        train_transform = albumentations.Compose([albumentations.RandomCrop(224, 224), albumentations.Normalize(), ToTensorV2()])
    valid_transform = []
    
    for i in range(0, 1024, 128):
        for j in range(0, 1024, 128):
            x_start = 800 if i + 224 > 1024 else i
            y_start = 800 if j + 224 > 1024 else j
            valid_transform.append(albumentations.Compose([albumentations.Crop(x_start, y_start, x_start + 224, y_start + 224), albumentations.Normalize(), ToTensorV2()]))

    traindataset = SatelliteTrainDataset(dataset_path=args.dataset_path, transform=train_transform)
    validdataset = SatelliteValidDataset(dataset_path=args.dataset_path)
    
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validloader = torch.utils.data.DataLoader(validdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return trainloader, validloader

def get_test_loader(args):
    testdataset = SatelliteTestDataset(dataset_path=args.dataset_path, csv_file=args.dataset_path+'test.csv')
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return testloader

def make_npy(mode = 'train', dataset_path = './dataset/'):
    # save as npy of size (image_num, x_size, y_size)
    
    # train : valid = 0.95 : 0.05
    train_split = int(7140 * 0.95)

    if mode == 'train':
        # read train data
        print('reading train data')
        train_images = []
        train_masks = []
        train_csv = pd.read_csv(dataset_path + 'train.csv')
        for i in range(train_split):
            if i % 100 == 0:
                print('reading', i, '/', train_split)
            img_path = train_csv.iloc[i, 1]
            image = cv2.imread(dataset_path + img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = rle_decode(train_csv.iloc[i, 2], (image.shape[0], image.shape[1]))
            train_images.append(image)
            train_masks.append(mask)
        
        train_images_np = np.stack(train_images, axis = 0)
        train_masks_np = np.stack(train_masks, axis = 0)

        np.save(dataset_path + 'train_images.npy', train_images_np)
        np.save(dataset_path + 'train_masks.npy', train_masks_np)
    
    elif mode == 'valid':
        # read valid data
        print('reading valid data')
        valid_images = []
        valid_masks = []

        valid_transform = []
        train_csv = pd.read_csv(dataset_path + 'train.csv')
        for i in range(0, 1024, 128):
            for j in range(0, 1024, 128):
                x_start = 800 if i + 224 > 1024 else i
                y_start = 800 if j + 224 > 1024 else j
                valid_transform.append(albumentations.Compose([albumentations.Crop(x_start, y_start, x_start + 224, y_start + 224), albumentations.Normalize()]))

        for i in range(train_split, 7140):
            if (i - train_split) % 100 == 0:
                print('reading', i - train_split, '/', 7140 - train_split)
            img_path = train_csv.iloc[i, 1]
            image = cv2.imread(dataset_path + img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = rle_decode(train_csv.iloc[i, 2], (image.shape[0], image.shape[1]))
            for transform in valid_transform:
                augmented = transform(image=image, mask=mask)
                valid_images.append(augmented['image'])
                valid_masks.append(augmented['mask'])
        
        valid_images_np = np.stack(valid_images, axis = 0)
        valid_masks_np = np.stack(valid_masks, axis = 0)

        np.save(dataset_path + 'valid_images.npy', valid_images_np)
        np.save(dataset_path + 'valid_masks.npy', valid_masks_np)

    elif mode == 'test':
        # read test data
        print('reading test data')

        test_images = []
        transform = albumentations.Normalize()
        valid_csv = pd.read_csv(dataset_path + 'valid.csv')
        for i in range(60640):
            if i % 100 == 0:
                print('reading', i, '/', 61640)
            img_path = valid_csv.iloc[i, 1]
            image = cv2.imread(dataset_path + img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            augmented = transform(image = image)['image']
            test_images.append(augmented)
        
        test_images_np = np.stack(test_images, axis = 0)
        np.save(dataset_path + 'test_images.npy', test_images_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Satellite make npy")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='/input')
    args = parser.parse_args()
    make_npy(mode = args.mode, dataset_path = args.dataset_path)