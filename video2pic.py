########## video to pic ############
import cv2
import os
import numpy as np

def Dirs2Pic(video_dir, save_dir, save_start_id, annotation_save_dir=''):
    listdirs = os.listdir(video_dir)
    for subdir in listdirs:
        subpath = os.path.join(video_dir, subdir)
        if os.path.isdir(subpath):
            save_start_id = Dirs2Pic(subpath, save_dir, save_start_id, annotation_save_dir)
        elif os.path.isfile(subpath):
            save_start_id = Video2Pic(subpath, save_dir, save_start_id, annotation_save_dir)
    return save_start_id

def Video2Pic(video_file, save_dir, save_start_id, annotation_save_dir=''):
    '''
    :param video_file: video file path
    :param save_dir: picture save path
    :param save_start_id: name of the first saved picture
    :param annotation_save_dir: saved dir of annotations file of lane
    :return: next video save start id
    '''
    print('Process video: ', video_file)
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_fps = int(fps / 5)
    save_num = 100000 # 有些视频损坏，直截取前面部分，没有损坏的设置个大值
    crop = [0, 0, 0, 0] # if the video need crop, [top, bottom, left, right]
    print('fps: ', fps)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count = save_start_id
    n = -1
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue
        n += 1
        if n % save_fps != 0:
            continue
        frame = frame[int(crop[0]):int(height - crop[1]), int(crop[2]):int(width - crop[3]), :]
        cv2.imwrite(os.path.join(save_dir, str(count) + '.jpg'), frame)
        if annotation_save_dir != '':
            if not os.path.exists(annotation_save_dir):
                os.makedirs(annotation_save_dir)
            for i in range(30):
                y = int(height - (crop[0] + crop[1]) - i * 20)
                horize = np.array([[0, y], [int(width - crop[2] - crop[3]), y]])
                cv2.polylines(frame, [horize], False, [0, 255, 255], 1)
            cv2.imwrite(os.path.join(annotation_save_dir, str(count) + '.jpg'), frame)
        count += 1
        if count >= save_num:
            break
    cap.release()
    print('Process video ', video_file, ' completely!')
    return count

def GeneratePic():
    video_dir = 'D:\\DataSet\\mylane\\test_video\\360CARDVR\\REC'
    save_dir = 'D:\\DataSet\\mylane\\lane5'
    annotation_save_dir = 'D:\\DataSet\\mylane\\annotation_lane5'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(annotation_save_dir):
        os.makedirs(annotation_save_dir)
    save_start_id = 0
    count = Dirs2Pic(video_dir, save_dir, save_start_id, annotation_save_dir)
    print('Total number: ', count)

if __name__ == '__main__':
    GeneratePic()