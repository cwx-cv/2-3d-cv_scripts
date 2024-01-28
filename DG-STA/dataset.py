import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import random
import numpy as np


class Action_Dataset(Dataset):
    """Action recognition dataset."""

    def __init__(self, data):
        """
        Args:
            data: a list of video and it's label
            time_len: length of input video
            use_data_aug: flag for using data augmentation
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        # print("ind: ", ind)
        data_ele = self.data[ind]

        skeleton = data_ele['x']

        # sample = {'skeleton': skeleton, "label": data_ele['y']}

        # return sample
        return torch.from_numpy(np.array(skeleton).astype(np.float32)), torch.tensor(int(data_ele['y']))


def data_preprocessing():
    with open('total.pkl', 'rb') as f:
        total_data = pickle.load(f)
    random.seed(100)
    train_data = []
    test_data = []
    labels = list(total_data.keys())
    labels.sort()

    print(labels)
    label_dict_nums = {}
    for label in labels:
        label_dict_nums[label] = 0
        datas = total_data[label]
        for ddd_idx, data in enumerate(datas):
            tmp_list = []
            for i in tqdm(range(data.shape[0])):
                tmp_list.append({'x': data[i], 'y': label})
                random.shuffle(tmp_list)
            # train_len = int(data.shape[0] * 0.8)
            tmp_list_len = len(tmp_list)
            label_dict_nums[label] += tmp_list_len
            if label <= 5:
                if ddd_idx >= 1:
                    train_data.extend(tmp_list)
                else:
                    test_data.extend(tmp_list)       
            else:

                train_len = int(tmp_list_len * 0.8)

                train_data.extend(tmp_list[:train_len])

                test_data.extend(tmp_list[train_len:])

    weights_train = []
    for label in labels:
        weights_train.append(label_dict_nums[label])

    weights_train = [max(weights_train) / x for x in weights_train]

    return train_data, test_data, weights_train


if __name__ == '__main__':
    # class_names = ['bad', 'medium', 'good']
    pass
