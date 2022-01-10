def CEF(base):
    flag = False
    # 遍历所有文件夹
    for root, ds, fs in os.walk(base):
        if len(ds) == 0 and len(fs) == 0 :
            os.rmdir(root)
            flag = True
        for f in fs:
            if '__' in f:
                os.remove(root + '/' + f)
            flag = True
    return flag

def Clean_EF(base):
    round = 0
    while True:
        round = round + 1
        print("This is " + str(round) + " round.")
        flag = CEF(base)
        if not flag:
            break

def walk_file(base):
    # 遍历所有文件夹并寻找.tex格式的文件
    for root, ds, fs in os.walk(base):
        for f in fs:
            if '.' not in f:
                continue
            if '.tex' in f:
                input_file = root + '/' + f
                output_file = root + '/' + f + '.docx'
                os.system("pandoc " + input_file + " -o " + output_file)
