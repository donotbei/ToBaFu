from torch.utils.data import Dataset


class TopoDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImgDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.y[idx]


class PairedDatasetWithSharedLabels(Dataset):
    def __init__(self, data1, data2, labels, transform=None):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels
        self.transform = transform

        # 确保两个数据集的长度和标签的长度相同
        assert len(self.data1) == len(self.data2) == len(self.labels), "数据集和标签的长度必须相同"

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.labels)

    def __getitem__(self, index):
        data1 = self.data1[index]
        data2 = self.data2[index]
        if self.transform:
            data2 = self.transform(data2)
        label = self.labels[index]
        return (data1, data2), label
