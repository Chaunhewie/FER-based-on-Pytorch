# coding=utf-8
import torch.utils.data as data


class RawDataSet(data.Dataset):
    """
    一个空数据集。可以通过add来写入数据。用于训练时的loop训练（识别错误的重新训练）
    """

    def __init__(self):
        self.data = []
        self.data_num = 0
        self.targets = []
        self.targets_num = 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (data, target)
        """
        if index >= self.__len__():
            return None, None
        return self.data[index], self.targets[index]

    def __len__(self):
        """
        Returns:
            int: data num.
        """
        if self.data_num == self.targets_num:
            return self.data_num
        else:
            return 0

    def add(self, d, target):
        self.data.append(d)
        self.targets.append(target)
        self.data_num += 1
        self.targets_num += 1


if __name__ == "__main__":
    c1 = RawDataSet()
    print(c1.__len__())
    c1.add(1, 2)
    print(len(c1))
    print(c1.__getitem__(0))
