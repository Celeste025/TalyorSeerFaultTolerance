import os
import math
import numpy as np
import torch 
import enum

class Recorder:
    def __init__(self):
        self.name = "record_step"
        self.cache = dict()
        self.warning_flag = True
        self.on = True
        self.mem = list()
        self.save_folder = "results"

    def update_name(self, name):
        self.name = name

    def update_folder(self, folder):
        self.save_folder = folder
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def record(self, data):  #data为dict, key为变量名, value应该为torch.tensor
        for key, value in data.items():
            if key not in self.cache:
                self.cache[key] = []
            if isinstance(value, torch.Tensor):
                self.cache[key].append(value.detach().cpu().numpy())
            else:
                if self.warning_flag:
                    print(f"Warning: {key}: {value} is not a torch.Tensor, there may be something wrong.")
                    self.warning_flag = False
                self.cache[key].append(value)

    def pack(self):   #将cache打包压进mem 然后清空cache
        self.mem.append(self.cache)
        self.cache = dict()
    
    def save(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        for (id, cache) in enumerate(self.mem):
            file_path = os.path.join(self.save_folder, f"{self.name}_{id}.npz")
            np.savez(file_path, **cache)
        self.mem = []  # Clear memory after saving
        print(f"Recorded data saved to {file_path}")

_recorder = Recorder()


