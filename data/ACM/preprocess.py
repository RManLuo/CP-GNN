#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/1 11:09
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : preprocess.py
# @Software: PyCharm
# @Describe:

import scipy.io

data_file_path = 'ACM.mat'
data = scipy.io.loadmat(data_file_path)
print(list(data.keys()))