import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from model.network import *
import pickle

np.set_printoptions(suppress=True)


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k.replace('module.',''): v for k, v in da.items() if k.replace('module.','') in db and not any(x in k for x in exclude) and v.shape == db[k.replace('module.','')].shape}


def init_model(weight_path):

    class_num = 7
    time_len = 41
    # time_len = 101
    
    dp_rate = 0.9
    # dp_rate = 0.7
    coordinates_nums = 5
    model = DG_STA(class_num, dp_rate, time_len, coordinates_nums)


    model.load_state_dict(torch.load(weight_path, map_location='cpu')['model'])
    # checkpoint = torch.load(weight_path, map_location='cpu')
    # exclude = []  # exclude keys
    # csd = checkpoint  # checkpoint state_dict as FP32
    # csd = intersect_dicts(csd, model['model'], exclude=exclude)  # intersect

    # model.load_state_dict(csd)
    return model

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax





class Action_Detect():
    def __init__(self, weight_path, width, height) -> None:
        pass

        self.width = width
        self.height = height
        self.first_frame = None
        self.device = torch.device('cuda:0')
        
        self.model = init_model(weight_path)
        self.model.to(self.device)
        self.model.eval()

        self.check_ids = {}
        self.check_ids_np = {}
        self.check_ids_types = {}

    def predict(self, data):
        tmp_data = torch.from_numpy(data.astype(np.float32))
        with torch.no_grad():
            res = self.model(tmp_data.to(self.device))
            return res


    def add_frame_boxs(self, frame_idx, frame_boxs):
        
        draw_boxs = []
        if self.first_frame is None:
            self.first_frame = int(frame_idx)

        ids = list(frame_boxs.keys())
        for id in ids:
            id_info = frame_boxs[id]
            cls, bbox_left, bbox_top, bbox_w, bbox_h = id_info

            center_x = bbox_left + bbox_w / 2.0
            center_y = bbox_top + bbox_h / 2.0

            if not id in list(self.check_ids.keys()):
                self.check_ids[id] = {}
                self.check_ids_np[id] = {}
            self.check_ids_types[id] = cls
            self.check_ids[id][frame_idx] = id_info
            box_list = [center_x / self.width, center_y / self.height, bbox_w / self.width, bbox_h / self.height, cls]

            self.check_ids_np[id][frame_idx] = box_list
        ######检测框已经对应好
        aeroplane_list = []
        other_list = []
        for id in list(self.check_ids_np.keys()):
            if self.check_ids_types[id] == 0:
                aeroplane_list.append(id)
            else:
                other_list.append(id)
        # print('aeroplane_list', aeroplane_list, 'other_list', other_list)

        frame_lens = 41
        # frame_lens = 101
        #######如果超过41、101帧率---就需要构建关系进行检测
        frame_idx_int = int(frame_idx)
        if frame_idx_int - self.first_frame >= frame_lens:
            start_frame = frame_idx_int - frame_lens
            all_action_boxs = []

            #############飞机自己本身的检测
            for aeroplane_id in aeroplane_list:
                total_list = []
                for frame in range(start_frame, frame_idx_int):
                    try:
                        box_data = self.check_ids_np[aeroplane_id][str(frame)]
                    except Exception as e:
                        box_data = [0, 0, 0, 0, -1] # self.check_ids_types[aeroplane_id]
                    total_list.append([box_data, box_data])
                
                all_action_boxs.append(np.array(total_list))


            ##############飞机与其它场面车辆关系的检测
            for aeroplane_id in aeroplane_list:

                for other_id in other_list:
                    total_list = []
                    for frame in range(start_frame, frame_idx_int):
                        try:
                            box_data = self.check_ids_np[aeroplane_id][str(frame)]
                        except Exception as e:
                            box_data = [0, 0, 0, 0, -1] # self.check_ids_types[aeroplane_id]

                        try:
                            b_box_data = self.check_ids_np[other_id][str(frame)]
                        except Exception as e:
                            b_box_data = [0, 0, 0, 0, -1] # self.check_ids_types[other_id]

                        total_list.append([box_data, b_box_data])

                all_action_boxs.append(np.array(total_list))

            all_action_boxs_np = np.array(all_action_boxs)
            output = self.predict(all_action_boxs_np[:,:,:,0:])

            predict = softmax(output.cpu().numpy())
            predict_class = np.argmax(predict, axis=1)
            # print(10 * '>', 'frame_idx', frame_idx, predict.shape)
            # print('frame_idx', frame_idx, predict_class)
            return {
                'predict_class': predict_class,
                'predict': predict,
                'boxs': all_action_boxs_np[:,-1,:,:]
            }
        
        return {}

# 0 餐车配餐结束
# 1 餐车配餐开始
# 2 飞机入位
# 3 客梯车对接
# 4 飞机离位
# 5 客梯车分离
# 6 其它


if __name__ == "__main__":
    # weight_path = './weight/150_dp-0.2_lr-0.0001/epoch_239_acc_0.8989224138.pth'
    # weight_path = './distributed_log/train_distributed_main_old/checkpoint-best.pth'
    # weight_path = './distributed_log/train_distributed_main_2/checkpoint-best.pth'
    weight_path = './distributed_log/train_distributed_main_new2/checkpoint-best.pth' ######权重文件
    # width, height = 1920, 536
    # width, height = 960, 268
    width, height = 1920, 1080 ######
    action_detect = Action_Detect(weight_path, width, height)



    txt_path = '939机位右侧-配餐结束.txt' ######
    # {0: 'aeroplane', 1: 'elevator car', 2: 'plane refueller', 3: 'tractor', 4: 'catering truck'}
    # 0   aeroplane        飞机
    # 1   elevator         客梯车
    # 2   plane refueller  加油车
    # 3   tractor          牵引车
    # 4   catering truck   配餐车

    with open(txt_path, 'r') as f:
        data = f.readlines()
        datas = [[float(y) for y in x.strip('\n').split(',')] for x in data]

    action_list = {}
    all_frame_info = {}

    for data_line in datas:
        frame_idx, id, cls, conf, bbox_left, bbox_top, bbox_w, bbox_h, _, _, _, _ = data_line
        frame_idx = str(int(frame_idx))
        id = str(int(id))
        cls = int(cls)

        if int(cls) in [2, 3]:
            continue
        if not frame_idx in list(all_frame_info.keys()):
            all_frame_info[frame_idx] = {}

        all_frame_info[frame_idx][id] = [cls, bbox_left, bbox_top, bbox_w, bbox_h]

    all_info = {}
    for frame_idx in list(all_frame_info.keys()):
        frame_boxs = all_frame_info[frame_idx]
        action_info = action_detect.add_frame_boxs(frame_idx, frame_boxs)

        if 'predict_class' in action_info.keys():
            # print(all_frame_info[frame_idx])
            print(frame_idx, action_info['predict_class'])

            # if np.sum(action_info['predict_class']) / len(action_info['predict_class']) != 6:
            #     print('检测到其它类别')
            #     exec()
            # exec()
        all_info[frame_idx] = action_info

    with open('peicanjieshu.pkl', 'wb') as f: ######
        pickle.dump(all_info, f)       




 
