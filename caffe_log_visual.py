############ caffe日志可视化 ############

import matplotlib.pyplot as plt
import numpy as np
import argparse

def pause_yolov3_log(log_file, scale_num = 1):
    iterationes = []
    losses = []
    detection_eval = []
    for i in range(scale_num):
        locals()['noobj' + str(i)] = []
        locals()['obj' + str(i)] = []
        locals()['iou' + str(i)] = []
        locals()['cat' + str(i)] = []
        locals()['recall' + str(i)] = []
        locals()['recall75' + str(i)] = []
        locals()['count' + str(i)] = []
        locals()['count' + str(i)] = []
        locals()['yolov3_losses' + str(i)] = []

    num = 0
    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Iteration" in line and "Testing net" not in line and "lr" not in line:
                arr = line.split(']')[1].strip().split(',')
                if len(arr) < 3:
                    continue
                iter = arr[0].strip().split(' ')
                if iter[0].strip() == "Iteration":
                    iteration = int(iter[1].strip())
                    iterationes.append(iteration)
                loss_arr = arr[2].strip().split("=")
                if loss_arr[0].strip() == "loss":
                    loss = float(loss_arr[1].strip())
                    losses.append(loss)

            if "Train net output #" in line:
                arr = line.split('Train net output #')[1].strip().split(":")
                det_loss = float(arr[1].split("(")[0].strip().split('=')[1].strip())
                for i in range(scale_num):
                    if arr[0] == str(i):
                        locals()['yolov3_losses' + str(i)].append(det_loss)

            if "noobj" in line:
                arr = line.split("]")[1].strip().split(" ")
                for i in range(scale_num):
                    if num % scale_num == i:
                        locals()['noobj' + str(i)].append(float(arr[1]))
                        locals()['obj' + str(i)].append(float(arr[3]))
                        locals()['iou' + str(i)].append(float(arr[5]))
                        locals()['cat' + str(i)].append(float(arr[7]))
                        locals()['recall' + str(i)].append(float(arr[9]))
                        locals()['recall75' + str(i)].append(float(arr[11]))
                        locals()['count' + str(i)].append(float(arr[13]))
                num += 1

            if "Test net output #" in line:
                detection_eval.append(float(line.split("detection_eval = ")[1].strip()))

    logs = []
    for i in range(scale_num):
        log = {}
        log['iterationes'] = iterationes
        log['losses'] = losses
        log['detection_eval'] = detection_eval
        log['yolov3_losses'] = locals()['yolov3_losses' + str(i)]
        log['noobj'] = locals()['noobj' + str(i)]
        log['obj'] = locals()['obj' + str(i)]
        log['iou'] = locals()['iou' + str(i)]
        log['cat'] = locals()['cat' + str(i)]
        log['recall'] = locals()['recall' + str(i)]
        log['recall75'] = locals()['recall75' + str(i)]
        log['count'] = locals()['count' + str(i)]
        logs.append(log)

    return logs

def visual_yolov3_log(logs):
    scale_num = len(logs)
    for i in range(scale_num):
        iterationes = logs[i]['iterationes']
        losses = logs[i]['losses']
        yolov3_losses = logs[i]['yolov3_losses']
        obj = logs[i]['obj']
        iou = logs[i]['iou']
        recall = logs[i]['recall']
        recall75 = logs[i]['recall75']

        # plt.ion()
        fig, axes = plt.subplots(nrows=3, ncols=2)
        fig.suptitle('Scale ' + str(i))
        axes[0,0].set(title="Total loss")
        axes[0,0].plot(iterationes, losses, label="Total loss", color="red", linewidth=1)

        axes[0,1].set(title="Loss")
        axes[0,1].plot(iterationes, yolov3_losses, label="Loss", color="red", linewidth=1)

        axes[1,0].set(title="Obj")
        length = len(obj)
        x = np.linspace(0, length - 1, length)
        axes[1,0].plot(x, obj, label="Obj", color="green", linewidth=1)

        axes[1,1].set(title="Iou")
        length = len(iou)
        x = np.linspace(0, length - 1, length)
        axes[1,1].plot(x, iou, label="Iou", color="blue", linewidth=1)

        axes[2,0].set(title="Recall")
        length = len(recall)
        x = np.linspace(0, length - 1, length)
        axes[2,0].plot(x, recall, label="Recall", color="green", linewidth=1)

        axes[2,1].set(title="Recall75")
        length = len(recall75)
        x = np.linspace(0, length - 1, length)
        axes[2,1].plot(x, recall75, label="Recall75", color="green", linewidth=1)
        plt.show()
        # plt.pause(10)
        # plt.close()
    return

def pause_ssd_log(log_file, scale_num=1):
    iterationes = []
    losses = []
    detection_eval = []
    for i in range(scale_num):
        locals()['mbox_losses' + str(i)] = []

    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Iteration" in line and "Testing net" not in line and 'lr' not in line:
                arr = line.split(']')[1].strip().split(',')
                if len(arr) < 2:
                    continue
                iter = arr[0].strip().split(' ')
                if iter[0].strip() == "Iteration":
                    iteration = int(iter[1].strip())
                    iterationes.append(iteration)
                loss_arr = arr[1].strip().split("=")
                if loss_arr[0].strip() == "loss":
                    loss = float(loss_arr[1].strip())
                    losses.append(loss)

            if "Train net output #" in line:
                arr = line.split('Train net output #')[1].strip().split(":")
                mbox_loss = float(arr[1].split("(")[0].strip().split('=')[1].strip())
                for i in range(scale_num):
                    if arr[0] == str(i):
                        locals()['mbox_losses' + str(i)].append(mbox_loss)

            if "Test net output #" in line:
                detection_eval.append(float(line.split("detection_eval = ")[1].strip()))

    logs = []
    for i in range(scale_num):
        log = {}
        log['iterationes'] = iterationes
        log['losses'] = losses
        log['detection_eval'] = detection_eval
        log['mbox_losses'] = locals()['mbox_losses' + str(i)]
        logs.append(log)
    return logs

def visual_ssd_log(logs):
    scale_num = len(logs)
    for i in range(scale_num):
        iterationes = logs[i]['iterationes']
        losses = logs[i]['losses']
        detection_eval = logs[i]['detection_eval']
        mbox_losses = logs[i]['mbox_losses']

        # plt.ion()
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.suptitle('Scale ' + str(i))
        axes[0,0].set(title="Total loss")
        axes[0,0].plot(iterationes, losses, label="Total loss", color="red", linewidth=1)

        axes[0,1].set(title="Mbox_losses")
        axes[0,1].plot(iterationes, mbox_losses, label="mbox_losses", color="red", linewidth=1)

        axes[1,0].set(title="Detection_eval")
        length = len(detection_eval)
        x = np.linspace(0, length - 1, length)
        axes[1,0].plot(x, detection_eval, label="detection_eval", color="green", linewidth=1)

        plt.show()
        # plt.pause(30)
        # plt.close()
    return

def visual_log():
    parser = argparse.ArgumentParser(description='Log visualization arguments.')
    parser.add_argument('--net_type', type=str, default='ssd', help='Network type. Currently only support caffe_ssd and caffe_mobilenet_yolov3.')
    parser.add_argument('--log_file', type=str, default='', help='Log file path.')
    parser.add_argument('--scale_num', type=int, default=1, help='Loss scale number.')
    args = parser.parse_args()
    net_type = args.net_type
    log_file = args.log_file
    scale_num = args.scale_num
    if log_file == '':
        print('''Usage: python caffe_log_visual.py --log_file your_log_file_path 
            [--net_type your_network_type{ssd or yolov3, default ssd}] 
            [--scale_num your_loss_scale_number{1 or 2, default 1}].''')
        return
    if net_type == 'yolov3':
        logs = pause_yolov3_log(log_file, scale_num)
        visual_yolov3_log(logs)
    elif net_type == 'ssd':
        logs = pause_ssd_log(log_file, scale_num)
        visual_ssd_log(logs)
    return

if __name__ == '__main__':
    visual_log()
