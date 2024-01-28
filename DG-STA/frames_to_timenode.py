def frame_to_timecode(framerate, frames):
    """
    通过视频帧转换成时间
    :param framerate: 视频帧率
    :param frames: 当前视频帧数
    :return: 时间(00:00:00:00) 第一段00表示小时、第二段00表示分钟、第三段00表示秒、第四段00表示帧数
    """
    return '{0:02d}:{1:02d}:{2:02d}:{3:02d}'.format(int(frames / (3600 * framerate)),
                                                    int(frames / (60 * framerate) % 60),
                                                    int(frames / framerate % 60),
                                                    int(frames % framerate))




# if __name__ == '__main__':
#     result = frame_to_timecode(10, 345) # 常德机场的视频帧率fps = 10
#     print(result) # 00:00:34:05


