#!/usr/env/python
# -*- coding: utf-8 -*-

"""
CAFFE SSD : STEP 1
---------------------------------
delete image without xml file
create list file: *.jpg *.xml

"""

import os
import copy
import random
import subprocess

class CreateFileList(object):
    def __init__(self, imgPath, imgMark):
        self._imgPath = imgPath
        self.listFilter(imgMark)

    def listFilter(self, imgMark):
        filelist = os.listdir(self._imgPath)
        self._imglist = copy.deepcopy(filelist)
        for img in filelist:
            if img[-3:] != imgMark:
                self._imglist.remove(img)
        
        self._xmllist = copy.deepcopy(filelist)
        for xml in filelist:
            if xml[-3:] != 'xml':
                self._xmllist.remove(xml)

    def deleteImgFile(self):
        count = 0
        print 'deleting images...'
        filelist = copy.deepcopy(self._imglist)
        for imgfile in filelist:
            xmlfile = imgfile[:-3] + 'xml'
            if not (xmlfile in self._xmllist):
                os.remove(self._imgPath + imgfile)
                self._imglist.remove(imgfile)
                count += 1
                print '    %d images deleted\r'%count,
        print '\ndone!'

    def renameImgFile(self):
        count = 1
        maxNum = len(str(len(self._imglist)))
        imglist = []
        xmllist = []

        print 'renaming images...'
        for imgfile in self._imglist:
            xmlfile = imgfile[:-3] + 'xml'
            name = 'sigar'+'0'*(maxNum-len(str(count)))+str(count)
            os.rename(self._imgPath + imgfile, self._imgPath + name +'.jpg')
            os.rename(self._imgPath + xmlfile, self._imgPath + name +'.xml')
            imglist.append(name + '.jpg')
            xmllist.append(name + '.xml')
            print '    %d/%d images & xml renamed\r'%(count, len(self._imglist)),
            count += 1
        self._imglist = imglist
        self._xmllist = xmllist
        print '\n    done!'

    def createList(self, filelist_dir, rate=0.8):
        index_list = range(len(self._imglist))
        random.shuffle(index_list)

        trainval_end = int(rate*len(self._imglist))
        with open(filelist_dir + 'img_trainval.txt', 'w') as imglistfile:
            print 'writing trainval image file list to %s ...'%filelist_dir
            for index in range(trainval_end):
                imglistfile.write(self._imglist[index] + ' ')
                imglistfile.write(self._xmllist[index] + '\n')
            print '    done!'
        self._trainval_list_file = filelist_dir + 'img_trainval.txt'

        with open(filelist_dir + 'img_test.txt', 'w') as imglistfile:
            print 'writing test image file list to %s ...'%filelist_dir
            for index in range(trainval_end, len(self._imglist)):
                imglistfile.write(self._imglist[index] + ' ')
                imglistfile.write(self._xmllist[index] + '\n')
            print '    done!'
        self._test_list_file = filelist_dir + 'img_test.txt'

    def createVOCSize(self, caffe_ssd_dir, file):
        print 'creating voc size file...'
        cmd = "{}build/tools/get_image_size " \
              "{} {} {}" \
              .format(caffe_ssd_dir, self._imgPath,
                      self._test_list_file, file)
        subprocess.Popen(cmd.split())
        print '    done!'

def main():
    # image and xml are under the same path
    imagePath = '{}/DATA/'.format(os.environ['HOME'])
    caffe_ssd = '{}/caffe-ssd'.format(os.environ['HOME'] )
    listfile_dir = '{}/caffe-ssd/data/myVOC/'.format(os.environ['HOME'] )
    voc_size_file = '{}/caffe-ssd/data/myVOC/test_name_size.txt'.format(os.environ['HOME'] )

    filelist = CreateFileList(imagePath, 'jpg')
    filelist.deleteImgFile()
    filelist.renameImgFile()
    filelist.createList(listfile_dir)
    filelist.createVOCSize(caffe_ssd, voc_size_file)

if __name__ == '__main__':
    main()
