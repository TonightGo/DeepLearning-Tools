import caffe
import cv2
import lmdb
from caffe.proto import caffe_pb2
import xml.dom.minidom as xdm
import json
import os
import numpy as np
import shutil
import random

# Process Opencv Mat data
def MatToDatum(img_file, resize_width=0, resize_height=0, min_dim=0, max_dim=0, is_color=True):
    '''

    :param img_file:
    :param resize_width:
    :param resize_height:
    :param min_dim:
    :param max_dim:
    :param is_color:
    :return:
    '''
    cv_read_flag = 1 if is_color == True else 0
    img = cv2.imread(img_file, cv_read_flag)
    if is_color:
        height, width, _ = img.shape
    else:
        height, width = img.shape
    if img is None:
        print("Could not open or find file ", img_file)
    if min_dim > 0 or max_dim > 0:
        num_rows = img.shape[0]
        num_cols = img.shape[1]
        min_num = min(num_rows, num_cols)
        max_num = max(num_rows, num_cols)
        scale_factor = 1
        if min_dim > 0 and min_dim > min_num:
            scale_factor = min_dim / min_num
        if max_dim > 0 and scale_factor * max_num > max_dim:
            scale_factor = max_dim / max_num
        if scale_factor == 1:
            img_resize = img
        else:
            img_resize = cv2.resize(img, fx=scale_factor, fy=scale_factor)
    elif resize_width > 0 and resize_height > 0:
        img_resize = cv2.resize(img, (resize_width, resize_height))
    else:
        img_resize = img
    # if is_color == True:
    #     img_swap = img_resize[:, :, ::-1].transpose((2, 0, 1))  # BGR2RGB and HWC2CHW
    # else:
    #     img_swap = img_resize
    img_swap = img_resize
    return img_swap, width, height

# Parse voc label
def VocToAnnotationDatum(img_file_path, line, resize_width=0, resize_height=0, min_dim=0, max_dim=0, is_color=True):
    '''

    :param line:
    :param resize_width:
    :param resize_height:
    :param min_dim:
    :param max_dim:
    :param is_color:
    :return:
    '''
    arr = line.strip().split(' ')
    img_file = arr[0]
    name = img_file.strip().split('/')[-1]
    xml_file = arr[1]
    img_data, width, height = MatToDatum(img_file, resize_width, resize_height, min_dim, max_dim, is_color)
    folder, filename, xml_width, xml_height, channels, objs = ParseVocXml(xml_file)
    if width != xml_width or height != xml_height:
        print(img_file, ' inconsistent image height.')
    return img_data, width, height, objs, name

# Parse txt label
def TxtToAnnotationDatum(img_file_path, line, resize_width=0, resize_height=0, min_dim=0, max_dim=0, is_color=True):
    '''

    :param line:
    :param resize_width:
    :param resize_height:
    :param min_dim:
    :param max_dim:
    :param is_color:
    :return:
    '''
    arr = line.strip().split('\t')
    name = arr[0]
    name = name.strip().split('/')[-1]
    img_file = img_file_path + name
    img_data, width, height = MatToDatum(img_file, resize_width, resize_height, min_dim, max_dim, is_color)
    if len(arr) < 2:
        objs = []
        return img_data, width, height, objs, name
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
    return img_data, width, height, objs, name

# Map name to number label use labelmap file.
def MapNameToLabel(label_map_file):
    with open(label_map_file) as f:
        lines = f.readlines()
        label_map = {}
        for line in lines:
            arr = line.strip().split(' ')
            label_map[arr[0]] = int(arr[1])
    return label_map

# Generate lmdb datum.
def GenerateAnnotatedDatum(img_data, label_map, objs, width, height, img_label = -1):
    '''

    :param img_data: CHW
    :param label_map: name map to label
    :param objs: object dictionaries include xmin, ymin, xmax, ymax, name, difficult. See voc xml.
    :param width: original image width, not resized
    :param height: original image height, not resized
    :param img_label: discarded
    :return:
    '''
    if img_data.dtype != 'uint8':
        print('Image data type must be unsigned byte!')
    anno_datum = caffe_pb2.AnnotatedDatum()
    anno_datum.type = 0
    enc_data = cv2.imencode('.jpg', img_data)
    anno_datum.datum.encoded = True
    anno_datum.datum.data = np.array(enc_data[1]).tostring()
    anno_datum.datum.channels = img_data.shape[2]
    anno_datum.datum.height = img_data.shape[0]
    anno_datum.datum.width = img_data.shape[1]
    anno_datum.datum.label = img_label

    # datum = caffe.io.array_to_datum(img_data, img_label)
    # anno_datum.datum.channels = datum.channels
    # anno_datum.datum.height = datum.height
    # anno_datum.datum.width = datum.width
    # anno_datum.datum.label = datum.label
    # anno_datum.datum.data = datum.data
    # anno_datum.datum.float_data.extend(datum.float_data)
    # anno_datum.datum.encoded = True

    for obj in objs:
        group_label = label_map[obj['name']]
        found_group = False
        for anno_group in anno_datum.annotation_group:
            if group_label == anno_group.group_label:
                if len(anno_group.annotation) == 0:
                    instance_id = 0
                else:
                    instance_id = anno_group.annotation[len(anno_group.annotation) - 1].instance_id + 1
                anno = anno_group.annotation.add()
                found_group = True
        if not found_group:
            anno_group = anno_datum.annotation_group.add()
            anno_group.group_label = group_label
            anno = anno_group.annotation.add()
            instance_id = 0
        anno.instance_id = instance_id
        instance_id += 1
        bbox = anno.bbox
        if obj['xmin'] > width or obj['ymin'] > height or obj['xmax'] > width \
                or obj['ymax'] > height or obj['xmin'] < 0 or obj['ymin'] < 0 or obj['xmax'] < 0 or obj['ymax'] < 0:
            print('Bounding box exceeds image boundary.')
        if obj['xmin'] > obj['xmax'] or obj['ymin'] > obj['ymax']:
            print('Bounding box irregular.')
        bbox.xmin = obj['xmin'] / width
        bbox.ymin = obj['ymin'] / height
        bbox.xmax = obj['xmax'] / width
        bbox.ymax = obj['ymax'] / height
        bbox.difficult = obj['difficult']
    return anno_datum

# Write lmdb, using callback function.
def WriteLmdb(lmdb_name, label_file, label_map_file, img_file_path, datum_generator=VocToAnnotationDatum,
              resize_width=0, resize_height=0, min_dim=0, max_dim=0, is_color=True, batch_size=1, redo=True,
              format='txt', parse_xml=None):
    '''

    :param lmdb_name:
    :param label_file:
    :param label_map_file:
    :param datum_generator:
    :param resize_width:
    :param resize_height:
    :param min_dim:
    :param max_dim:
    :param is_color:
    :param batch_size:
    :param redo:
    :param format:
    :param parse_xml:
    :return:
    '''
    if redo:
        if os.path.exists(lmdb_name):
            shutil.rmtree(lmdb_name)
    label_map = MapNameToLabel(label_map_file)
    lmdb_env = lmdb.open(lmdb_name, map_size=1e12)
    lmdb_txn = lmdb_env.begin(write=True)
    count = 0
    if not redo:
        for _ in lmdb_txn.cursor():
            count += 1
    if format == 'txt':
        with open(label_file) as f:
            lines = f.readlines()
            for line in lines:
                img_data, width, height, objs, name = datum_generator(img_file_path, line, resize_width, resize_height,
                                                                      min_dim, max_dim, is_color)
                key_str = '{:0>8d}'.format(count) + '_' + name
                anno_datum = GenerateAnnotatedDatum(img_data, label_map, objs, width, height, -1)
                lmdb_txn.put(key_str.encode(), anno_datum.SerializeToString())
                count += 1
                if count % batch_size == 0:
                    lmdb_txn.commit()
                    lmdb_txn = lmdb_env.begin(write=True)
                    print('Process ', str(count), ' files!')
            if count % batch_size != 0:
                lmdb_txn.commit()
                print('Process ', str(count), ' files!')
    elif format == 'xml':
        if parse_xml == None:
            print('If format is set to be xml, the callback function to parse your xml file must be provide. '
                  'The function parse xml to a list, every element in the list is the label of each image.')
        labels = parse_xml(label_file)
        for label in labels:
            img_data, width, height, objs, name = datum_generator(img_file_path, label, resize_width, resize_height,
                                                                  min_dim, max_dim, is_color)
            key_str = '{:0>8d}'.format(count) + '_' + name
            anno_datum = GenerateAnnotatedDatum(img_data, label_map, objs, width, height, -1)
            lmdb_txn.put(key_str.encode(), anno_datum.SerializeToString())
            count += 1
            if count % batch_size == 0:
                lmdb_txn.commit()
                lmdb_txn = lmdb_env.begin(write=True)
                print('Process ', str(count), ' files!')
        if count % batch_size != 0:
            lmdb_txn.commit()
            print('Process ', str(count), ' files!')
    print('Convert to lmdb completely!')
    return

# Parse voc label
def ParseVocXml(xml_file):
    '''

    :param xml_file: xml label file
    :return:
    '''
    domtree = xdm.parse(xml_file)
    annotation = domtree.documentElement
    folder = annotation.getElementsByTagName('folder')[0].childNodes[0].data
    filename = annotation.getElementsByTagName('filename')[0].childNodes[0].data
    size = annotation.getElementsByTagName('size')[0]
    width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size.getElementsByTagName('height')[0].childNodes[0].data)
    channels = int(size.getElementsByTagName('depth')[0].childNodes[0].data)
    objects = annotation.getElementsByTagName('object')
    objs = []
    for object in objects:
        obj = {}
        name = object.getElementsByTagName('name')[0].childNodes[0].data
        difficult = object.getElementsByTagName('difficult')
        bndbox = object.getElementsByTagName('bndbox')[0].childNodes[0].data
        xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].data
        ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].data
        xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].data
        ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].data
        obj['name'] = name
        obj['xmin'] = float(xmin)
        obj['ymin'] = float(ymin)
        obj['xmax'] = float(xmax)
        obj['ymax'] = float(ymax)
        if difficult is None:
            dif = 0
        else:
            dif = difficult[0].childNodes[0].data
        obj['difficult'] = int(dif)
        objs.append(obj)
    return folder, filename, width, height, channels, objs

# Parse my label format
def ParseMyXml(xml_file):
    domtree = xdm.parse(xml_file)
    collection = domtree.documentElement
    labels = collection.getElementsByTagName('LABEL')
    return labels

# Parse my label format
def MyXmlToAnnotationDatum(img_file_path, label, resize_width=0, resize_height=0, min_dim=0, max_dim=0, is_color=True):
    file_name = label.getAttribute('file')
    img_file = os.path.join(img_file_path, file_name)
    img_data, width, height = MatToDatum(img_file, resize_width, resize_height, min_dim, max_dim, is_color)
    roi = label.getElementsByTagName('ROI')[0]
    xml_width = int(roi.getAttribute('right'))
    xml_height = int(roi.getAttribute('bottom'))
    if width != xml_width or height != xml_height:
        print(img_file, ' inconsistent image height.')
    rects = label.getElementsByTagName('RECT')
    objs = []
    if rects is not None:
        for rect in rects:
            obj = {}
            obj['name'] = 'crosswalk'
            left = float(rect.getAttribute('left'))
            top = float(rect.getAttribute('top'))
            right = float(rect.getAttribute('right'))
            bottom = float(rect.getAttribute('bottom'))
            obj['xmin'] = left
            obj['ymin'] = top - 400
            obj['xmax'] = right
            obj['ymax'] = bottom - 400
            obj['difficult'] = 0
            objs.append(obj)
    return img_data, width, height, objs, file_name

# Parse lmdb
def ReadLmdb(lmdb_path):
    env = lmdb.open(lmdb_path)
    txn = env.begin()
    anno_datum = caffe_pb2.AnnotatedDatum()
    for key, val in txn.cursor():
        anno_datum.ParseFromString(val)
        data = anno_datum.datum.data
        print(key, anno_datum)
        np_data = np.frombuffer(data, dtype=np.uint8)
        cv_img = cv2.imdecode(np_data, -1)
        cv2.imshow('test', cv_img)
        cv2.waitKey(0)
    env.close()
    return

# shuffle the lmdb
def ShuffleLmdb(lmdb_path, batch_size):
    '''

    :param lmdb_path:
    :return:
    '''
    env = lmdb.open(lmdb_path, map_size=1e12)
    txn = env.begin(write=True)
    records = []
    for record in txn.cursor():
        records.append(record)
    env.close()
    shutil.rmtree(lmdb_path)
    random.shuffle(records)
    env = lmdb.open(lmdb_path, map_size=1e12)
    txn = env.begin(write=True)
    num = 0
    for record in records:
        # reconstruct index
        keystr = '{:0>8d}'.format(num) + record[0][8:]
        txn.put(keystr, record[1])
        num += 1
        if num % batch_size == 0:
            txn.commit()
            txn = env.begin(write=True)
    if num % batch_size != 0:
        txn.commit()
    env.close()
    print("Shuffle lmdb successfully!")
    return

# split trainval and test dataset according to my xml label file
def SplitMyXmlTrainvalTest(xml_file, trainval_xml_file, test_xml_file, split_ratio=0.1):
    domtree = xdm.parse(xml_file)
    collection = domtree.documentElement
    labels = collection.getElementsByTagName('LABEL')
    random.shuffle(labels)
    test_num = int(len(labels) * split_ratio)
    test_doms = labels[:test_num]
    train_doms = labels[test_num:]
    dom = xdm.Document()
    rootnode = dom.createElement('LABELS')
    dom.appendChild(rootnode)
    for test_dom in test_doms:
        rootnode.appendChild(test_dom)
    with open(test_xml_file, 'wb') as test_xml:
        test_xml.write(dom.toprettyxml(encoding='utf-8'))
    dom = xdm.Document()
    rootnode = dom.createElement('LABELS')
    dom.appendChild(rootnode)
    for train_dom in train_doms:
        rootnode.appendChild(train_dom)
    with open(trainval_xml_file, 'wb') as train_xml:
        train_xml.write(dom.toprettyxml(encoding='utf-8'))
    return

# split trainval and test dataset according to my txt label_file
def SplitTxtTrainvalTest(txt_file, trainval_txt_file, test_txt_file, split_ratio=0.1):
    trainval_fid = open(trainval_txt_file, 'w')
    test_fid = open(test_txt_file, 'w')
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        test_num = int(len(lines) * split_ratio)
        test_lines = lines[:test_num]
        trainval_lines = lines[test_num:]
        trainval_fid.writelines(trainval_lines)
        test_fid.writelines(test_lines)
    trainval_fid.close()
    test_fid.close()
    return

def main():
    db_name = ['trainval', 'test']
    lmdb_path = '/dataset_sdf_8/ice/road/lmdb/' # lmdb path
    label_map_file = lmdb_path + 'label_map.txt'

    ####### TXT #######
    img_file_path = '/dataset_sdf_8/ice/road/crosswalk_crop/'
    label_file = '/dataset_sdf_8/ice/road/lmdb/banmaxian.label'
    trainval_label = lmdb_path + db_name[0] + '_1.label'
    test_label = lmdb_path + db_name[1] + '_1.label'
    # split trainval and test dataset according to label file
    split_ratio = 0.1  # test and trainval dataset ratio
    SplitTxtTrainvalTest(label_file, trainval_label, test_label, split_ratio)
    format = 'txt'
    parse_xml = None
    ####### TXT #######

    ####### XML #######
    # img_file_path = '/dataset_sdf_8/ice/road/mydataset_crop/' # image file path
    # label_file = '/dataset_sdf_8/ice/road/lmdb/mydataset_label/images_list_new.label4'
    # trainval_label = lmdb_path + db_name[0] + '_2.label'
    # test_label = lmdb_path + db_name[1] + '_2.label'
    # split trainval and test dataset according to label file
    # split_ratio = 0.1  # test and trainval dataset ratio
    # SplitMyXmlTrainvalTest(label_file, trainval_label, test_label, split_ratio)
    # format = 'xml'
    # parse_xml = ParseMyXml
    ####### XML #######


    # generate lmdb
    batch_size = 128
    is_color = True
    resize_width = 320
    resize_height = 320
    min_dim = 0
    max_dim = 0
    redo = False # False: add new data to old lmdb. True: regenerate lmdb.'
    shuffle_db = True # After adding new data to old lmdb, you can shuffle the new lmdb using this parameter.

    for name in db_name:
        lmdb_name = lmdb_path + name + '_lmdb'
        sub_label_file = lmdb_path + name + '_1.label' # TXT
        # sub_label_file = lmdb_path + name + '_2.label' # XML
        if format == 'xml':
            WriteLmdb(lmdb_name, sub_label_file, label_map_file, img_file_path, MyXmlToAnnotationDatum,
                      resize_width, resize_height, min_dim, max_dim, is_color, batch_size, redo, format, parse_xml)
        elif format == 'txt':
            WriteLmdb(lmdb_name, sub_label_file, label_map_file, img_file_path, TxtToAnnotationDatum,
                      resize_width, resize_height, min_dim, max_dim, is_color, batch_size, redo, format, parse_xml)
        if shuffle_db == True:
            ShuffleLmdb(lmdb_name, batch_size)
    return

if __name__ == '__main__':
    main()