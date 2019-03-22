#! /usr/bin/env python
# _*_ coding:utf-8 _*_

import os, sys

class CommonTool(object):

    
    def __init__(self):
        pass
    
    def current_path(self, modoule_name):
        file_name=sys.modules[modoule_name].__file__
        dir_name=os.path.dirname(file_name)
        return os.path.abspath(dir_name)

