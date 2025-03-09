import os

def find_different_files(folder1, folder2):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    different_files = []

    for file in files1.symmetric_difference(files2):
        if os.path.isfile(os.path.join(folder1, file)):
            different_files.append(file)

    return different_files

# 用法示例
folder1 = 'F:\\datasets\\IR_WRF\\test\\sharp'
folder2 = 'F:\\datasets\\IR_WRF\\test\\blur'

different_files = find_different_files(folder1, folder2)

if different_files:
    print("文件名不同的文件：")
    for file in different_files:
        print(file)
else:
    print("两个文件夹中的文件名相同。")