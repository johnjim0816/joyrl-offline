#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 23:35:33
LastEditor: JiangJi
LastEditTime: 2023-04-16 23:41:46
Discription: 
'''
from enum import Enum

class BufferType(Enum):
    REPLAY = 1
    REPLAY_QUE = 2
    PER = 3
    PER_QUE = 4