import xml.dom.minidom as xdm
import os
import cv2
import shutil
import argparse

# 检查文件是否存在，不存在，创建之
def CheckDirs(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

# 将标签文件解析成列表，列表选项是字典，字典包含键值name、objs、width、height
# 其中objs也是字典，包含键值name、xmin、ymin、xmax、ymax、difficult
# 可以扩展自己的文件解析方法，返回同样的字典列表即可
def ParseLabel(label_file, img_dir):
    with open(label_file) as f:
        lines = f.readlines()
        dicts = []
        for line in lines:
            arr = line.strip().split('\t')
            name = arr[0]
            name = name.strip().split('/')[-1]
            img_file = os.path.join(img_dir, name)
            img = cv2.imread(img_file)
            if img is None:
                print(img_file, ' is not a valid image file!')
                continue
            dict = {}
            dict['name'] = name
            dict['width'] = img.shape[1]
            dict['height'] = img.shape[0]
            if len(img.shape) == 3:
                dict['channels'] = img.shape[2]
            else:
                dict['channels'] = 1
            if len(arr) < 2:
                objs = []
                dict['objs'] = objs
                dicts.append(dict)
                continue
            labels = arr[1]
            labels_arr = labels.strip().split(',')
            objs = []
            for label_str in labels_arr:
                obj = {}
                obj['name'] = 'crosswalk'
                label_arr = label_str.lstrip('[').rstrip(']').split('~')
                if len(label_arr) < 2:
                    continue
                top_left = label_arr[0]
                bottom_right = label_arr[1]
                first_arr = top_left.strip().lstrip('(').rstrip(')').split(';')
                first_arr = [float(x) for x in first_arr]
                second_arr = bottom_right.strip().lstrip('(').rstrip(')').split(';')
                second_arr = [float(x) for x in second_arr]
                tl_x = min(first_arr[0], second_arr[0])
                tl_y = min(first_arr[1], second_arr[1])
                br_x = max(first_arr[0], second_arr[0])
                br_y = max(first_arr[1], second_arr[1])
                obj['xmin'] = float(tl_x)
                obj['ymin'] = float(tl_y) - 400
                obj['xmax'] = float(br_x)
                obj['ymax'] = float(br_y) - 400
                obj['difficult'] = 0
                objs.append(obj)
            dict['objs'] = objs
            dicts.append(dict)

    return dicts

# 生成VOC格式XML文件，接收解析函数传过来的字典列表和XML目录名
def GenerateXml(dicts, xml_folder):
    for dict in dicts:
        name = dict['name']
        short_name = name.strip().split('.')[0]
        anno_path = os.path.join(xml_folder, short_name + '.xml')
        w = dict['width']
        h = dict['height']
        c = dict['channels']
        rects = dict['objs']
        xml = xdm.Document()
        annotation = xml.createElement('annotation')
        xml.appendChild(annotation)
        folder = xml.createElement('folder')
        folder_value = xml.createTextNode('crosswalk_crop')
        folder.appendChild(folder_value)
        filename = xml.createElement('filename')
        filename_value = xml.createTextNode(name)
        filename.appendChild(filename_value)
        size = xml.createElement('size')
        width = xml.createElement('width')
        height = xml.createElement('height')
        depth = xml.createElement('depth')
        width_value = xml.createTextNode(str(w))
        height_value = xml.createTextNode(str(h))
        depth_value = xml.createTextNode(str(c))
        width.appendChild(width_value)
        height.appendChild(height_value)
        depth.appendChild(depth_value)
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        annotation.appendChild(folder)
        annotation.appendChild(filename)
        annotation.appendChild(size)
        for rect in rects:
            left = round(rect['xmin'])
            top = round(rect['ymin'])
            right = round(rect['xmax'])
            bottom = round(rect['ymax'])

            object = xml.createElement('object')
            name = xml.createElement('name')
            bndbox = xml.createElement('bndbox')
            xmin = xml.createElement('xmin')
            ymin = xml.createElement('ymin')
            xmax = xml.createElement('xmax')
            ymax = xml.createElement('ymax')
            name_value = xml.createTextNode(rect['name'])
            xmin_value = xml.createTextNode(str(left))
            ymin_value = xml.createTextNode(str(top))
            xmax_value = xml.createTextNode(str(right))
            ymax_value = xml.createTextNode(str(bottom))
            name.appendChild(name_value)
            xmin.appendChild(xmin_value)
            ymin.appendChild(ymin_value)
            xmax.appendChild(xmax_value)
            ymax.appendChild(ymax_value)
            bndbox.appendChild(xmin)
            bndbox.appendChild(ymin)
            bndbox.appendChild(xmax)
            bndbox.appendChild(ymax)
            object.appendChild(name)
            object.appendChild(bndbox)
            annotation.appendChild(object)
        with open(anno_path, 'wb') as xmlfile:
            # xml.writexml(xmlfile, indent='\t', addindent='\t', newl='\n', encoding='utf-8')
            xmlfile.write(xml.toprettyxml(indent='\t', encoding='utf-8'))
    return

# 生成训练验证集和测试集的名字文件，接收解析函数传过来的字典列表和TXT文件路径
def GenerateTrainvalTest(dicts, txt_file):
    with open(txt_file, 'w') as f:
        for dict in dicts:
            name = dict['name']
            short_name = name.strip().split('.')[0]
            f.write(short_name)
    return

# 生成VOC图片与VOC XML文件的关联txt文件
def GenerateImageToXml(dicts, txt_file, jpegimg_dir='JPEGImages', xml_dir='Annotations'):
    with open(txt_file, 'w') as f:
        for dict in dicts:
            name = dict['name']
            short_name = name.strip().split('.')[0]
            jpg_path = os.path.join(jpegimg_dir, name)
            xml_path = os.path.join(xml_dir, short_name + '.xml')
            txt = jpg_path + ' ' + xml_path + '\n'
            f.write(txt)
    return

# 生成test_name_size，SSD专用
def GenerateTestNameSize(dicts, txt_file):
    with open(txt_file, 'w') as f:
        for dict in dicts:
            name = dict['name']
            short_name = name.strip().split('.')[0]
            width = dict['width']
            height = dict['height']
            txt = short_name + ' ' + str(height) + ' ' + str(width) + '\n'
            f.write(txt)
    return

# 移动图片到VOC目录JPEGImages，checkout校验文件是否是图片或者图片是否损坏
def CopyImages(origal_folder, new_folder, checkout=False):
    subs = os.listdir(origal_folder)
    for sub in subs:
        path = os.path.join(origal_folder, sub)
        if os.path.isdir(path):
            CopyImages(path, new_folder)
        if os.path.isfile(path):
            if checkout:
                img = cv2.imread(path, -1)
                if img is None:
                    print(path, ' is not a valid image file!')
                    continue
            new_path = os.path.join(new_folder, sub)
            shutil.copy(path, new_path)
    return

# 从普通文件目录转为VOC格式目录，并生成相应文件，需要文件目录和标签文件作为输入
# 可扩展自己的文件解析方式
def main():
    root_dir = '/dataset_sdf_8/ice/road/voc_crosswalk/' # VOC格式数据根目录
    jpegimg_dir = os.path.join(root_dir, 'JPEGImages')
    xml_dir = os.path.join(root_dir, 'Annotations')
    main_dir = os.path.join(root_dir, 'ImageSets/Main')
    CheckDirs(jpegimg_dir)
    CheckDirs(xml_dir)
    CheckDirs(main_dir)

    img_dir = '/dataset_sdf_8/ice/road/crosswalk_crop/'
    CopyImages(img_dir, jpegimg_dir)

    datasets = ['trainval', 'test']
    test_dataset_name = 'test'
    label_dir = './'
    for dataset in datasets:
        name_file = os.path.join(main_dir, dataset + '.txt')
        label_file = os.path.join(label_dir, dataset + '.label')
        txt_file = os.path.join(root_dir, dataset + '.txt')

        print('Parse label file: ', label_file)
        dicts = ParseLabel(label_file, jpegimg_dir)
        print('Parse label file successfully!')
        print('Generate xml files: ')
        GenerateXml(dicts, xml_dir)
        print('Generate xml files successfully!')
        print('Generate trainval and test name files: ')
        GenerateTrainvalTest(dicts, name_file)
        print('Generate trainval and test name files successfully!')
        print('Generate image files corresponding to xml files: ')
        GenerateImageToXml(dicts, txt_file, jpegimg_dir='JPEGImages', xml_dir='Annotations')
        print('Generate image files corresponding to xml files successfully!')

        if dataset == test_dataset_name:
            test_name_size_file = os.path.join(root_dir, 'test_name_size.txt')
            print('Generate test_name_size file: ')
            GenerateTestNameSize(dicts, test_name_size_file)
            print('Generate test_name_size file successfully!')
    return

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='To voc and generate ssd lmdb.')
    parse.add_argument('--to_lmdb', action='store_true', default=False, help='To lmdb?')
    args = parse.parse_args()
    main()
    if args.to_lmdb:
        print('Please transfer your own parameters!')
        os.system('python /usr/bin/python ./scripts/create_annoset.py '
                  '--anno-type=detection --label-map-file=./data/road/labelmap_crosswalk.prototxt --min-dim=0 '
                  '--max-dim=0 --resize-width=320 --resize-height=320 --check-label --encode-type=jpg '
                  '--encoded /dataset_sdf_8/ice/road/voc_crosswalk/ '
                  '/dataset_sdf_8/ice/road/voc_crosswalk/trainval.txt '
                  '/dataset_sdf_8/ice/road/voc_crosswalk/lmdb/trainval_lmdb examples/crosswalk '
                  '--redo') # 执行生成LMDB的python命令