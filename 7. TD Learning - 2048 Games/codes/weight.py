#!/usr/bin/env python3

"""
Basic framework for developing 2048 programs in Python

Author: Hung Guei (moporgic)
        Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
        http://www.aigames.nctu.edu.tw
Modifier: Kuo-Hao Ho (lukewayne123)
"""

from array import array


class weight:
    
    def __init__(self, len = 0):
        self.value = [0] * len
        return
    
    def __getitem__(self, index):
        return self.value[index]
    
    def __setitem__(self, index, value):
        self.value[index] = value
        return
    
    def __len__(self):
        return len(self.value)
    
    def save(self, output):
        """ serialize this weight to a file object """
        array('Q', [len(self.value)]).tofile(output)
        array('f', self.value).tofile(output)
        return True
    
    def load(self, input):
        """ deserialize from a file object """
        size = array('Q')
        size.fromfile(input, 1)
        size = size[0]
        value = array('f')
        value.fromfile(input, size)
        self.value = list(value)
        return True
    