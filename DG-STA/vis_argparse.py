import random
import numpy as np
import cv2
import pickle
import numpy
from PIL import Image, ImageDraw, ImageFont
from frames_to_timenode import frame_to_timecode # 导入将视频帧转换为00:00:00:00的格式

import argparse # 导入模块

def parse_opt():
    parser = argparse.ArgumentParser(description='Command line parameters used for visualization module') # 创建解析器parser
    # 添加自定义命令行参数---这里pkl_path以及下面的两个都表示可选参数没有先后顺序 注意：当去掉--就表示必写参数且有先后顺序
    parser.add_argument("-p", "--pkl_path", default='peicanjieshu.pkl', help='the path of pkl file')
    parser.add_argument("-v", "--vis_vid_path", default='pcjs.mp4', help='the path of visualize video path')
    parser.add_argument("-s", "--save_vid_path", default='pcjs_track_keynode_vis_res.mp4', help='the path of save video path')
    opt = parser.parse_args() # 解析参数
    return opt # 返回解析后的所有自定义参数

def cv2ImgAddText(img, text, left, top, textColor=(255, 120, 10), textSize=20):

    if isinstance(img, numpy.ndarray):  # 判断是否为OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV的格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


args = parse_opt() # 调用函数parse_opt返回解析后的自定义命令行参数
print(args) # 打印所有自定义的命令行参数
pkl_path_final = args.pkl_path ######111
with open(pkl_path_final, 'rb') as f:  # 将行为检测的结果文件xxx.pkl文件打开---这个要修改
    all_info = pickle.load(f)  # 加载数据

label_des = {
    0: '餐车配餐开始',
    1: '餐车配餐结束',
    2: '飞机入位',
    3: '客梯车对接',
    4: '飞机离位',
    5: '客梯车分离',
    6: '其它'
}

# 0 餐车配餐开始
# 1 餐车配餐结束
# 2 飞机入位
# 3 客梯车对接
# 4 飞机离位
# 5 客梯车分离
# 6 其它

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)] # 随机生成颜色

plot_a_bool = True  ### 是否绘制A框---自定义设置根据自己的需求
plot_b_tool = True  ### 是否绘制B框---自定义设置根据自己的需求
video_path = args.vis_vid_path  # 要可视化视频的路径---这个要修改 ######222
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'MP4V') # 语法过时---替换为下面的正确形式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
save_path = args.save_vid_path ######333
out = cv2.VideoWriter(save_path, fourcc, int(fps), (int(width), int(height)))  # 输出视频的名称及路径---自定义设置
frame_idx = 0
while cap.isOpened():

    ret, frame = cap.read()
    if ret:
        frame_idx += 1
        try:
            action_info = all_info[str(frame_idx)]
        except Exception as e:

            action_info = {}

        if 'predict_class' in action_info.keys():
            # print(frame_idx)
            predict_class = action_info['predict_class']
            boxs = action_info['boxs']
            predict = action_info['predict']
            # print(boxs.shape)
            # print(frame_idx, predict_class)

            for predict_props, class_int, box in zip(predict, predict_class, boxs):
                plot_ids = [0, 1]  ### 默认a, b框都绘制

                if not plot_a_bool:  ### a的框不绘制
                    plot_ids = [1]
                if not plot_b_tool:  ### b的框不绘制
                    plot_ids = [0]

                if class_int in [0, 1, 4, 5]:
                    plot_ids = [1]

                for target_id in plot_ids:
                    center_x, center_y, bbox_w, bbox_h, cls = box[target_id]
                    center_x = center_x * width
                    bbox_w = bbox_w * width
                    center_y = center_y * height
                    bbox_h = bbox_h * height
                    x = int(center_x - bbox_w / 2)
                    y = int(center_y - bbox_h / 2)
                    w = int(bbox_w)
                    h = int(bbox_h)
                    # 绘制a、b的框
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), tuple(colors[6]), 2) # 框的颜色在这里设置

                    # 将帧数写入到视频中---转换为时间戳00:00:00:00的格式
                    # frame = cv2ImgAddText(frame, 'frame: ' + str(frame_idx), 60, 60, tuple(colors[2]), 50)
                    frame = cv2ImgAddText(frame, 'The current timestamp is: ' + str(frame_to_timecode(fps, frame_idx)), 60, 120, tuple(colors[2]), 50)
                    # 将行为预测的概率写入到视频帧上
                    # action_prob = round(float(predict_props[class_int]) * 100, 2)
                    # print(action_prob)
                    # action_prob_final = "%.2f%%" % action_prob
                    # frame = cv2ImgAddText(frame, str(label_des[class_int]) + ':' + str(action_prob_final), x, y - 10, (136, 255, 12), 30)
                    # frame = cv2ImgAddText(frame, str(label_des[class_int]) + ':' + str(round(predict_props[class_int], 2)), x, y - 10, (36, 255, 12), 30)
                    frame = cv2ImgAddText(frame, str(label_des[class_int]), x, y - 10, tuple(colors[1]), 30) # 文本的颜色在这里设置


            # cv2.imwrite('aaa.png', frame)
            # exec()
                    # frame = cv2.putText(frame, str(label_des[class_int]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            # cv2.imwrite('bbb.png', frame)
            # exec()
        # write the flipped frame

        # cv2.imwrite('./tmp_img/' + str(frame_idx) + ".jpg", frame)
        out.write(frame)

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
