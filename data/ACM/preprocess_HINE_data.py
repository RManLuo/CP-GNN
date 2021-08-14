#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/1/14 14:00
# @Author  : Raymond luo
# @Mail    : luolinhao1998@gmail.com
# @File    : preprocess_HINE_data.py
# @Software: PyCharm

import scipy.io

data_file_path = 'ACM.mat'
data = scipy.io.loadmat(data_file_path)
print(list(data.keys()))