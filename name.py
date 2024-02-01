import os
from os import listdir
from os.path import isfile, join

# 用法示例
folder_path = "C:/Users/Komorebi/Documents/SchoolWork/项目/2.1/双百23/workspace/LBP"
files = os.listdir(folder_path)
rec_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
print(rec_files[2])
