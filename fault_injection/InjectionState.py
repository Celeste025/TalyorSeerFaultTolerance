import os
import math
import numpy as np
import torch 
import enum
class InjectionState:
    def __init__(self):
        self._current_step = 0
        self.info = ""
        self.inject_bit = -1  # 默认值，-1表示random注入，否则表示只注入哪一位
        self.profiling = False  # 是否开启profiling模式
        self.global_args = dict()
        self.global_args['protect'] = 'No'
        self.global_args['max_int'] = 0
        self.global_args['max_fp'] = 0

    def current_step(self):
        return self._current_step
    
    def set_step(self, step):
        self._current_step = step

    def set_info(self, info):
        self.info = info
    
    def add_info(self, info):
        self.info += info
    
    def get_info(self, info):
        return self.info
    
    def clear_info(self):
        self.info = ""
    
    def set_inject_bit(self, bit):  #bit:int
        self.inject_bit = bit
        print("Set inject bit to ", bit)


# 模块加载时创建单例
_injection_state = InjectionState()
