import numpy as np
import os
import random

from numpy.core.defchararray import split
random.seed(100)
min_seq = 8

dataset_fold ="./handpose_x_gesture_v1_mediapipe_np"
dict_label={
    '000-one':1,
    '005-yearh':2,
    '009-Iloveyou':3,
    '010-gun':4,
    '012-nine':5
    }

def split_train_test(data_len):
    pass
    split_rate = 0.8
    name_classes = os.listdir(dataset_fold)
    train_data = []
    test_data = []
    for name_class in name_classes:
        label = dict_label[name_class]


        name_class_path = os.path.join(dataset_fold,name_class)

        npy_list = os.listdir(name_class_path)
        random.shuffle(npy_list)
        # npy_list = npy_list[:data_len]

        train_len = int(len(npy_list)*split_rate)

        for  idx,npy_name in enumerate(npy_list):
            image_np = np.load(os.path.join(name_class_path,npy_name))
            image_np = np.array([image_np])
            # print('image_np',image_np.shape)

            # print(image_np.shape)

            # exec()
            data_ele = {}
            data_ele["skeleton"] = image_np
            data_ele["label"] = label
            if idx <train_len:
                train_data.append(data_ele)

            else:
                test_data.append(data_ele)

    return train_data, test_data
# def split_train_test(data_cfg):
#     def parse_file(data_file,data_cfg):
#         #parse train / test file

#         label_list = []
#         all_data = []
#         for line in data_file:
#             data_ele = {}
#             data = line.split() #【id_gesture， id_finger， id_subject， id_essai， 14_labels， 28_labels size_sequence】
#             #video label
#             if data_cfg == 0:
#                 label = int(data[4])
#             elif data_cfg == 1:
#                 label = int(data[5])
#             label_list.append(label) #add label to label list
#             data_ele["label"] = label
#             #video
#             video = []
#             joint_path = dataset_fold + "/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt".format(data[0],data[1],data[2],data[3])
#             joint_file = open(joint_path)
#             for joint_line in joint_file:
#                 joint_data = joint_line.split()
#                 joint_data = [float(ele) for ele in joint_data]#convert to float
#                 joint_data = np.array(joint_data).reshape(22,3)#[[x1,y1,z1], [x2,y2,z2],.....]
#                 video.append(joint_data)
                
#             while len(video) < min_seq:
#                 video.append(video[-1])
#             # print(np.array(video).shape)
#             data_ele["skeleton"] = video
#             data_ele["name"] = line
#             all_data.append(data_ele)
#             joint_file.close()

#         print(np.array(all_data[1]["skeleton"]).shape )
#         print(np.array(all_data).shape)
#         print(np.array(label_list).shape)
#         exec()
#         return all_data, label_list



#     print("loading training data........")
#     train_path = dataset_fold + "/train_gestures.txt"
#     train_file = open(train_path)
#     train_data, train_label = parse_file(train_file,data_cfg)
#     assert len(train_data) == len(train_label)

#     print("training data num {}".format(len(train_data)))

#     print("loading testing data........")
#     test_path = dataset_fold + "/test_gestures.txt"
#     test_file = open(test_path)
#     test_data, test_label = parse_file(test_file, data_cfg)
#     assert len(test_data) == len(test_label)

#     print("testing data num {}".format(len(test_data)))

#     return train_data, test_data

if __name__ == "__main__":
    split_train_test()
