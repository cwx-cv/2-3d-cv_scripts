import os
import random
import numpy as np
import pickle
##############创建自己的训练数据集


def create_random_dict(seed, b_list=range(0, 100), create_nums=10, sample_len=10):
    total_strings = []
    i = 0
    while len(total_strings) < create_nums:
        random.seed(seed+i)
        blist_webId = random.sample(b_list, sample_len)
        blist_webId.sort()
        tmp_string = ','.join([str(x) for x in blist_webId])

        if not tmp_string in total_strings:
            total_strings.append(tmp_string)
        i += 1
        # print(total_strings)
    return total_strings

# print(len() )
# exec()


def get_label(labe_path):
    # labe_path = './video/exp/label.txt'
    labe_info = {}
    with open(labe_path, 'r') as f:
        data = f.readlines()
        data = [x.strip('\n') for x in data]
        for info in data:
            des, v = info.split('=')
            if v == ' None':
                labe_info[des] = None
            else:
                labe_info[des] = int(v)

    if labe_info['b_id'] is None:
        labe_info['b_id'] = labe_info['a_id']
    if labe_info['a_id'] is None:
        labe_info['a_id'] = labe_info['b_id']
    return labe_info


def get_track(text_path):
    with open(text_path, 'r') as f:
        data = f.readlines()
        data = [[float(y) for y in x.strip('\n').split(',')] for x in data]

    track_dict = {}
    for lines in data:
        frame_idx, id, cls, conf, bbox_left, bbox_top, bbox_w, bbox_h, _, _, _, _ = lines
        frame_idx_int = int(frame_idx)
        if not frame_idx_int in list(track_dict.keys()):
            track_dict[frame_idx_int] = {}
        track_dict[frame_idx_int][int(id)] = [
            cls, conf, bbox_left, bbox_top, bbox_w, bbox_h]

    return track_dict


def init_ids_data(start_frame, action_frame, width, height, a_id, b_id, track_dict):
    id_before = {}

    for i in range(start_frame, action_frame):
        try:
            a_cls, a_conf, a_bbox_left, a_bbox_top, a_bbox_w, a_bbox_h = track_dict[i][a_id]

        except Exception as e:
            a_cls, a_conf, a_bbox_left, a_bbox_top, a_bbox_w, a_bbox_h = -1, 0, 0, 0, 0, 0

        a_center_x = a_bbox_left + a_bbox_w / 2.0
        a_center_y = a_bbox_top + a_bbox_h / 2.0

        try:
            b_cls, b_conf, b_bbox_left, b_bbox_top, b_bbox_w, b_bbox_h = track_dict[i][b_id]
        except Exception as e:
            b_cls, b_conf, b_bbox_left, b_bbox_top, b_bbox_w, b_bbox_h = -1, 0, 0, 0, 0, 0
        b_center_x = b_bbox_left + b_bbox_w / 2.0
        b_center_y = b_bbox_top + b_bbox_h / 2.0
        tmp_a_b_dict = {'a': [a_center_x / width, a_center_y / height, a_bbox_w / width, a_bbox_h / height, a_cls], ########
                        'b': [b_center_x / width, b_center_y / height, b_bbox_w / width, b_bbox_h / height, b_cls]  ########
                        }
        dict_v = sum(tmp_a_b_dict['a']) + sum(tmp_a_b_dict['b'])

        # print(10*'>',dict_v)
        if dict_v:
            id_before[i] = tmp_a_b_dict

    return id_before


def track_data_select(label_dict, track_dict, a_id, b_id):
    ####获取到 起始  动作 结束的数据
    width = label_dict['width'] * 1.0
    height = label_dict['height'] * 1.0
    start_frame = label_dict['start_frame']
    end_frame = label_dict['end_frame']
    action_frame = label_dict['action']
    start_dict = init_ids_data(
        start_frame, action_frame, width, height, a_id, b_id, track_dict)
    action_dict = init_ids_data(
        action_frame, action_frame+1, width, height, a_id, b_id, track_dict)
    end_dict = init_ids_data(action_frame+1, end_frame,
                             width, height, a_id, b_id, track_dict)

    return start_dict, action_dict, end_dict


def get_data_by_random_key(data_dict, random_keys_list):
    samples = []
    for key_string in random_keys_list:
        key_list = [int(x) for x in key_string.split(',')]
        single_sample = []
        for key in key_list:
            # tmp_data = []
            a = data_dict[key]['a']
            b = data_dict[key]['b']
            tmp_data = [a, b]
            # tmp_data.extend(a)
            # tmp_data.extend(b)
            single_sample.append(tmp_data)
        samples.append(single_sample)
    return np.array(samples)


def data_enhancement(seed, start_dict, action_dict, end_dict, create_nums=400):
    ####数据增强

    # create_nums = 400
    sample_len = 20
    action_key = list(action_dict.keys())[0]
    tmp = [action_dict[action_key]['a'], action_dict[action_key]['b']]
    # tmp.extend()
    # tmp.extend()
    action_np = []
    for i in range(create_nums):
        action_np.append([tmp])
    action_np = np.array(action_np)

    start_keys = list(start_dict.keys())
    # print(start_keys,type(start_keys[0]),start_dict[15])
    random_keys_list = create_random_dict(
        seed, b_list=start_keys, create_nums=create_nums, sample_len=sample_len)
    start_np = get_data_by_random_key(start_dict, random_keys_list)

    end_keys = list(end_dict.keys())

    random_keys_list = create_random_dict(
        seed, b_list=end_keys, create_nums=create_nums, sample_len=sample_len)
    end_np = get_data_by_random_key(end_dict, random_keys_list)
    # print('action_np', start_np.shape, 'end_np',
    #       end_np.shape, 'action_np', action_np.shape)

    total_data = np.concatenate((start_np, action_np, end_np), axis=1)
    # print(total_data.shape)

    return total_data




if __name__ == '__main__':
    root_path = './dataset_add' # 原始标注数据集的根目录
    files = os.listdir(root_path)

    total_data_dict = {}
    total_data_dict[6] = []

    for idx, file_name in enumerate(files):
        print(idx, file_name)
        # if idx!=2:
        #     continue
        
        file_path = os.path.join(root_path, file_name)
        total_data_dict[idx] = []
        datas = os.listdir(file_path)
        for data_name in datas:

            label_path = os.path.join(file_path, data_name, 'label.txt')
            track_path = os.path.join(file_path, data_name, data_name+'.txt')
            if not (os.path.exists(label_path) and os.path.exists(track_path)):
                print(file_name, 'not exists')

            label_dict = get_label(label_path)

            track_dict = get_track(track_path)

            print(label_dict)
            a_id = label_dict['a_id']
            b_id = label_dict['b_id']
            print(10*'>', 'a_id', str(a_id), 'b_id', str(b_id))
            ###########    原始的分类标签数据
            start_dict, action_dict, end_dict = track_data_select(
                label_dict, track_dict, a_id, b_id)
            total_data = data_enhancement(
                idx+100, start_dict, action_dict, end_dict, create_nums=1000)
            total_data_dict[idx].append(total_data)
            ########### {0: 'aeroplane', 1: 'elevator car', 2: 'plane refueller', 3: 'tractor', 4: 'catering truck'}
            ###########    获取到飞机的id
            aeroplane_list = []
            other_list = []
            action_frame_info = track_dict[label_dict['action']]
            for lable_id_key in list(action_frame_info.keys()):
                class_int = int(action_frame_info[lable_id_key][0])
                if class_int == 0:
                    aeroplane_list.append(str(lable_id_key))
                else:
                    if not class_int in [2, 3]:
                        other_list.append(str(lable_id_key))

            exist_a_b = str(a_id) + "_" + str(b_id)
            print(exist_a_b)

            normal_action_list = []    
                 
            for aeroplane_label in aeroplane_list:
                #####其它飞机与其它场面车辆的运动关系
                for other_label in other_list:
                    desa = aeroplane_label+"_"+other_label
                    desb = other_label +"_"+aeroplane_label
                    if desa != exist_a_b and desb != exist_a_b:
                        normal_action_list.append(desa)

                #####飞机本身的位置关系
                for other_label in aeroplane_list:
                    desa = aeroplane_label+"_"+other_label
                    desb = other_label +"_"+aeroplane_label
                    if desa!=exist_a_b and desb !=exist_a_b:
                        normal_action_list.append(desa)                
            #############其它车的相对位置关系
            for aeroplane_label in other_list:
                for other_label in other_list:
                    desa = aeroplane_label+"_"+other_label
                    desb = other_label +"_"+aeroplane_label
                    if desa!=exist_a_b and desb != exist_a_b:
                        normal_action_list.append(desa) 
            ####飞机与飞机的运动位置关系
            print('normal_action_list', normal_action_list)
            if len(normal_action_list):
                print('normal_action_list', normal_action_list)
                for des in normal_action_list:
                    a_id_tmp, b_id_tmp = [int(vvv) for vvv in des.split('_')]
                    start_dict, action_dict, end_dict = track_data_select(label_dict, track_dict, a_id_tmp, b_id_tmp)
                    total_data = data_enhancement(6+100, start_dict, action_dict, end_dict, create_nums=20) ######modify40--->20
                    total_data_dict[6].append(total_data)
    for key in list(total_data_dict.keys()):
        sum_nums = 0
        for tmp_obj in total_data_dict[key]:
            sum_nums += tmp_obj.shape[0]
        print(key, sum_nums)
    with open('total.pkl', 'wb') as f:
        pickle.dump(total_data_dict, f)

    # exec()
# text_path = './video/exp/tracks/test.txt'
# for idx in range(labe_info['start_frame'], labe_info['end_frame']+1):
#     for id in [labe_info['a_id'], labe_info['b_id']]:
#         cls,conf, bbox_left, bbox_top, bbox_w, bbox_h = track_dict[idx][id]
