#!/usr/env/python
# -*- coding: utf-8 -*-

"""
CAFFE SSD : STEP 2
---------------------------------
file list -> lmdb

"""

import subprocess
from createFileList import CreateFileList as CF

class CreateLMDB(object):
    def __init__(self):
        self._db = 'lmdb'
        self._anno_type = 'detection'
        self._redo = 1

        self._min_dim = 0
        self._max_dim = 0
        self._width = 0
        self._height = 0

        self._extra_cmd = '--encode-type=jpg --encoded'
        if self._redo:
            self._extra_cmd += ' --redo'

    def createList(self, imgPath, imgMark, caffe_ssd, filelist_dir, voc_size_file):
        print 'creating file list...'
        self._filelist = CF(imgPath, imgMark)
        self._filelist.deleteImgFile()
        self._filelist.renameImgFile()
        self._filelist.createList(filelist_dir)
        self._filelist.createVOCSize(caffe_ssd, voc_size_file)
        print '    done!'

    def createDB(self, caffe_ssd_dir, mapfile, data_root, dblink, train_db_path, test_db_path):
        cmd = "python {}/create_annoset.py " \
              "--anno-type={} " \
              "--label-map-file={} " \
              "--min-dim={} " \
              "--max-dim={} " \
              "--resize-width={} " \
              "--resize-height={} " \
              "--check-label {} {} " \
              .format(caffe_ssd_dir+'scripts', self._anno_type,
                      mapfile, self._min_dim,
                      self._max_dim, self._width, self._height,
                      self._extra_cmd, data_root)

        # train cmd
        train_cmd = cmd + "{} {} {}" \
                    .format(self._filelist._trainval_list_file,
                      train_db_path,
                      dblink)
        subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)

        # test cmd
        test_cmd = cmd + "{} {} {}" \
                    .format(self._filelist._test_list_file,
                      test_db_path,
                      dblink)
        subprocess.Popen(test_cmd.split(), stdout=subprocess.PIPE)


def main():
    crDB = CreateLMDB()
    # image and xml are under the same path
    imagePath = '{}/DATA/image/'.format(os.environ['HOME'])
    caffe_ssd = '{}/caffe-ssd/'.format(os.environ['HOME'])
    listfile_dir = '{}/caffe-ssd/data/myVOC/'.format(os.environ['HOME'])
    voc_size_file = '{}/caffe-ssd/data/SigarVOC/test_name_size.txt'.format(os.environ['HOME'])
    crDB.createList(imagePath, 'jpg', caffe_ssd, listfile_dir, voc_size_file)

    # create lmdb
    mapfile = caffe_ssd + 'data/SigarVOC/labelmap_voc.prototxt'
    train_db = '{}/DATA/voc_trainval_lmdb'.format(os.environ['HOME'])
    test_db = '{}/DATA/voc_test_lmdb'.format(os.environ['HOME'])
    dblink = 'examples/sigar_voc'
    crDB.createDB(caffe_ssd, mapfile, imagePath[:-1], dblink, train_db, test_db)

if __name__ == '__main__':
    main()
