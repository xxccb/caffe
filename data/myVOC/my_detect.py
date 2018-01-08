#!/usr/bin/env python
# -*- codong:utf-8 -*-

import os
import sys
path_ = os.environ['HOME'] + '/caffe-ssd'
sys.path.insert(0, path_ + '/python')

import numpy as np
import caffe
import cv2
from google.protobuf import text_format
from caffe.proto import caffe_pb2


class my_detect(object):
    def __init__(self, args):
        if isinstance(args, list):
            self._labelmap_file = args[0]
            self._model_def = args[1]
            self._model_weights = args[2]

            self._use_gpu = args[3]
        else:
            self._labelmap_file = args.labelmap_file
            self._model_def = args.model_def
            self._model_weights = args.model_weights

            self._use_gpu = args.use_gpu

        self._img = []
        self._image_resize = 300
        self._image_shape = 0
        self._rect_color = (0, 0, 255)
        self._text_color = (0, 255, 0)
        self._threshold = 0.6
        self._mean = np.array([0, 0, 0], dtype=np.uint8)
        self._statistic = dict()

        self.init_net()


    def init_net(self):
        # set gpu mode
        if self._use_gpu:
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        
        # load PASCAL VOC labels
        file_ = open(self._labelmap_file, 'r')
        self._labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file_.read()), self._labelmap)

        # init caffe net
        self._net = caffe.Net(self._model_def,   # defines the structure of the model
                self._model_weights,             # contains the trained weights
                caffe.TEST)                      # use test mode (e.g., don't perform dropout)


    def get_labelname(self):
        self._num_labels = len(self._labelmap.item)
        labelnames = []
        if type(self._top_label_indices) is not list:
            self._top_label_indices = [self._top_label_indices]
        for label in self._top_label_indices:
            found = False
            for i in xrange(0, self._num_labels):
                if label == self._labelmap.item[i].label:
                    found = True
                    labelnames.append(self._labelmap.item[i].display_name)
                    break
            assert found == True
        return labelnames

   
    def statistic(self):
        for label in self._top_labels:
            if self._statistic.has_key(label):
                self._statistic[label] += 1
            else:
                self._statistic[label] = 1


    def detect_it(self, img):
        # pre-handle image
        self._img = []
        self._img = cv2.resize(img, (self._image_resize, self._image_resize))
        self._img -= self._mean
        self._img = self._img.transpose((2, 0, 1))

        # input
        self._net.blobs['data'].data[...] = self._img

        detections = self._net.forward()['detection_out']
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than self._threshold.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self._threshold]
        
        self._top_conf = det_conf[top_indices]
        self._top_label_indices = det_label[top_indices].tolist()
        self._top_labels = self.get_labelname()
        self._top_xmin = det_xmin[top_indices]
        self._top_ymin = det_ymin[top_indices]
        self._top_xmax = det_xmax[top_indices]
        self._top_ymax = det_ymax[top_indices] 
        self.statistic()


    def draw_it(self, img, show_rate):
        show_img = cv2.resize(img, (int(img.shape[1]*show_rate), int(img.shape[0]*show_rate)))
        for i in xrange(self._top_conf.shape[0]):
            xmin = int(round(self._top_xmin[i] * img.shape[1]))
            ymin = int(round(self._top_ymin[i] * img.shape[0]))
            xmax = int(round(self._top_xmax[i] * img.shape[1]))
            ymax = int(round(self._top_ymax[i] * img.shape[0]))

            score = self._top_conf[i]
            label = int(self._top_label_indices[i])
            if label == self._num_labels-1:
                # background
                continue
            label_name = self._top_labels[i]
            display_txt = '%s: %.2f '%(label_name, score)
            coords = ((int(xmin*show_rate), int(ymin*show_rate)), \
                   (int(xmax*show_rate), int(ymax*show_rate)))

            cv2.rectangle(show_img, coords[0], coords[1],  self._rect_color, thickness=2)
            cv2.putText(show_img, display_txt, (coords[0][0], coords[0][1] - 5), \
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self._text_color, thickness=2)

        return show_img


    def detect_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print 'can\'t open video file: {}'.format(video_file)
            return

        print 'detecting within file {}'.format(video_file)
        cv2.namedWindow('detect')
        self._statistic.clear()
        while True:
            ret, img = cap.read()
            if not ret:
                print 'finished!'
                break
            
            self.detect_it(img)
            img = self.draw_it(img, 1)
            cv2.imshow('detect', img)
            cv2.waitKey(10) 

        print 'statistics for detect:'
        for key in self._statistic:
            print '{}: {}'.format(key, self._statistic[key])


if __name__ == '__main__':
    args = list()
    args.append('{}/data/SigarVOC/labelmap_voc.prototxt'.format(path_))
    args.append('{}/models/VGGNet/SIG_VOC/SSD_300x300/deploy.prototxt'.format(path_))
    args.append('{}/models/VGGNet/SIG_VOC/SSD_300x300/VGG_SIG_VOC_SSD_300x300_iter_35000.caffemodel'.format(path_))
    args.append(True)

    file_ = '/home/12.1_clip.avi'
    sd = ssd_detect(args)
    sd.detect_video(file_)
