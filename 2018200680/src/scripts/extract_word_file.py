import numpy as np
import os
# print(np.random.randint(0,10,10))
print(os.getcwd())
dir_path = os.getcwd()
# save_path2 = dir_path + '\extract'
# file_path = dir_path + '\\texfile'
save_path2 = './extract'
file_path = './texfile'
save_path1 = dir_path + '\\texfile'
i=0
for root, dirs, files in os.walk(dir_path + '\Arxiv6K'):  # 获取当前文件夹的信息
    print(root)
    print(dirs)
    print(files)
    # i+=1
    # if i>10:
    #     break
    for file in files:                       # 扫描所有文件
        if os.path.splitext(file)[1] == ".tex":# 提取出所有后缀名为md的文件
            i+=1
            print(root)
            os.chdir(root)
            # print(os.getcwd())

            # print("转换开始：" + "pandoc " + file + " -o " + os.path.splitext(file)[0] + ".docx")
            # print("pandoc " + "-s -o " + save_path1 + '\\' + str(i) + ".docx" + ' ' + root+'\\'+file)
            # print("转换开始：" + "pandoc " + file + " -o " + save_path + '\\' + str(i) + ".docx")
            # 使用os.system调用pandoc进行格式转化
            # os.system("pandoc " + file + " -o " + save_path + '\\' + str(i) + ".docx")
            # os.system("pandoc " + " -s -o " + save_path + '\\' + str(i) + ".docx" + ' ' + root+'\\'+file)
            print('copy' + root +'\\' + file + ' to ' + save_path1)
            os.system("copy " + file + ' ' + save_path1 + '\\' + str(i) + ".tex")
            # print("转换完成...",save_path1,'-',file,' ',i)
            # break
    # break
i=0
print('2')
for root, dirs, files in os.walk(file_path):  # 获取当前文件夹的信息
    print(root)
    print(dirs)
    print(files)
    # break
    # i+=1
    if i>10:
        break
    for file in files:                       # 扫描所有文件
        if os.path.splitext(file)[1] == ".tex":# 提取出所有后缀名为md的文件
            i+=1
            # print(root)
            # os.chdir(root)
            # print(os.getcwd())

            print("转换开始：" + "pandoc " + file + " -o " + os.path.splitext(file)[0] + ".docx")
            print("pandoc " + "-s -o " + save_path2 + '\\' + str(i) + ".docx" + ' ' + root+'\\'+file)
            # print("转换开始：" + "pandoc " + file + " -o " + save_path + '\\' + str(i) + ".docx")
            # 使用os.system调用pandoc进行格式转化
            # os.system("pandoc " + file + " -o " + save_path + '\\' + str(i) + ".docx")
            os.system("pandoc " + " -s -o " + save_path2 + '/' + str(i) + ".docx" + ' ' + file_path + '/' + file)
            # os.system("copy " + file + ' ' + save_path)
            print("转换完成...",save_path2,'-',file,' ',i)
            # break
    # break
print('3')