import os

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import TopoDataset, PairedDatasetWithSharedLabels, ImgDataset


def delete_sparse_features(data, zero_count_threshold):
    """
    according to the number of zeros in each column, delete the sparse features
    :param data: feature data (DataFrame)
    :param zero_count_threshold: the number of zeros threshold (int)
    :return: filtered data (DataFrame)
    """
    columns_to_keep = np.sum(data == 0, axis=0) <= zero_count_threshold
    filtered_data = data.loc[:, columns_to_keep]
    return filtered_data


def prepare_data(config):
    """
    prepare data
    :param config: configuration information (dict)
    # :param zero_count_threshold: the number of zeros threshold (int)
    # :param train_size: the proportion of training set (float)
    # :param feature_info_path: the path of feature information (str)
    # :param data_dir: the directory of data (str)
    :return: train data and test data (DataFrame)
    """
    feature_info_path = config['feature_info']
    data_info = pd.read_csv(feature_info_path)
    train_size = config['train_size']  # 0.8

    data_info = delete_sparse_features(data_info, config['zero_count_threshold'])

    # change the data type of features to float32
    for col in data_info.columns[1:-1]:
        data_info[col] = data_info[col].astype('float32')

    train_data, test_data = None, None
    for label, group in data_info.groupby('label'):
        group_size = len(group)
        # random shuffle
        group = group.sample(frac=1).reset_index(drop=True)
        # split the data into training set and test set
        train_group = group[:int(train_size * group_size)]
        test_group = group[int(train_size * group_size):]

        if label == 0:
            train_data = train_group
            test_data = test_group
        else:
            train_data = pd.concat([train_data, train_group])
            test_data = pd.concat([test_data, test_group])

    # standardize the features
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    n_train = len(train_data)
    train_data.iloc[:, 1:-1] = all_features[:n_train]
    test_data.iloc[:, 1:-1] = all_features[n_train:]

    # label encoding
    label_encoder = LabelEncoder()
    train_data['label'] = label_encoder.fit_transform(train_data['label'])
    test_data['label'] = label_encoder.transform(test_data['label'])

    return train_data, test_data


def get_dataloader(data, config, datatype):
    """
    get data loader
    :param data: data (DataFrame)
    :param config: configuration information (dict)
    :param datatype: data type (str)
    :return: data loader
    """
    data_dir = config['data_dir']
    image_paths = data['image_path']
    topo_features = data.drop(['image_path', 'label'], axis=1).values
    y = data['label'].values

    # read image data
    images = []
    for img_path in image_paths:
        img = cv2.imread(str(os.path.join(data_dir, img_path)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    # data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),  # convert to PIL image
        transforms.RandomHorizontalFlip(),  # random horizontal flip
        transforms.RandomVerticalFlip(),  # random vertical flip
        transforms.RandomRotation(15),  # random rotation
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # random resized crop
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # color jitter
        transforms.ToTensor(),  # convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))  # random erasing
    ])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    topo_train_batch_size = config['batch_size']['topo_train']
    img_train_batch_size = config['batch_size']['img_train']
    test_batch_size = config['batch_size']['test']
    if datatype == 'train':
        topo_dataset = TopoDataset(topo_features, y)
        image_dataset = ImgDataset(images, y, transform=train_transform)
        topo_loader = DataLoader(topo_dataset, batch_size=topo_train_batch_size, shuffle=True)
        image_loader = DataLoader(image_dataset, batch_size=img_train_batch_size, shuffle=True)
        return topo_loader, image_loader
    elif datatype == 'test':
        combined_dataset = PairedDatasetWithSharedLabels(topo_features, images, y, transform=transform)
        combined_loader = DataLoader(combined_dataset, batch_size=test_batch_size, shuffle=False)
        return combined_loader


def read_data(datatype):
    """
    read data
    :param datatype: data type (str)
    :return: data loader
    """
    config = OmegaConf.load('config/config.yaml')['data']
    train_data, test_data = prepare_data(config)

    if datatype == 'train':
        topo_train_loader, img_train_loader = get_dataloader(train_data, config, 'train')
        return topo_train_loader, img_train_loader
    elif datatype == 'test':
        combined_loader = get_dataloader(test_data, config, 'test')
        return combined_loader
