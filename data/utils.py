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
    根据特征列零的个数删除稀疏特征
    :param data: 特征数据 (DataFrame)
    :param zero_count_threshold: 零的个数阈值 (int)
    :return: 删除稀疏特征后的特征数据 (DataFrame)
    """
    columns_to_keep = np.sum(data == 0, axis=0) <= zero_count_threshold
    filtered_data = data.loc[:, columns_to_keep]
    return filtered_data


def prepare_data(config):
    """
    准备数据
    :param config: 配置信息 (dict)
    # :param zero_count_threshold: 零的个数阈值 (int)
    # :param train_size: 训练集占比 (float)
    # :param feature_info_path: 特征信息文件路径 (str)
    # :param data_dir: 数据目录路径 (str)
    :return: 训练集和测试集 (DataFrame)
    """
    feature_info_path = config['feature_info']
    data_info = pd.read_csv(feature_info_path)
    train_size = config['train_size']  # 0.8

    data_info = delete_sparse_features(data_info, config['zero_count_threshold'])

    # 把特征数据类型转化成float
    for col in data_info.columns[1:-1]:
        data_info[col] = data_info[col].astype('float32')

    train_data, test_data = None, None
    for label, group in data_info.groupby('label'):
        group_size = len(group)
        # 打乱数据
        group = group.sample(frac=1).reset_index(drop=True)
        # 按比例划分训练集和测试集
        train_group = group[:int(train_size * group_size)]
        test_group = group[int(train_size * group_size):]

        if label == 0:
            train_data = train_group
            test_data = test_group
        else:
            train_data = pd.concat([train_data, train_group])
            test_data = pd.concat([test_data, test_group])

    # 对特征进行标准化
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    n_train = len(train_data)
    train_data.iloc[:, 1:-1] = all_features[:n_train]
    test_data.iloc[:, 1:-1] = all_features[n_train:]

    # 对标签进行编码
    label_encoder = LabelEncoder()
    train_data['label'] = label_encoder.fit_transform(train_data['label'])
    test_data['label'] = label_encoder.transform(test_data['label'])

    # # 查看训练集和测试集各类别的数量
    # print(f'Train set: \n{train_data["label"].value_counts()}')
    # print(f'Test set: \n{test_data["label"].value_counts()}')

    return train_data, test_data


def get_dataloader(data, config, datatype):
    """
    获取数据加载器
    :param data: 数据 (DataFrame)
    :param config: 配置信息 (dict)
    :param datatype: 数据类型 (str)
    :return: 数据加载器
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

    # 定义图像增强变换
    train_transform = transforms.Compose([
        transforms.ToPILImage(),  # 转换为PIL图像
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),  # 随机垂直翻转
        transforms.RandomRotation(15),  # 随机旋转15度
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并调整大小
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))  # 随机擦除
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
    读取数据
    :param datatype: 数据类型 (str)
    :return: 训练集和测试集的数据加载器
    """
    config = OmegaConf.load('config/config.yaml')['data']
    train_data, test_data = prepare_data(config)

    if datatype == 'train':
        topo_train_loader, img_train_loader = get_dataloader(train_data, config, 'train')
        return topo_train_loader, img_train_loader
    elif datatype == 'test':
        combined_loader = get_dataloader(test_data, config, 'test')
        return combined_loader
