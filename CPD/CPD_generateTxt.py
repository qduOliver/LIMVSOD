import os
import glob

##############################################################################################################

# davis_test
# Segtrack-v2
# Visal
# Easy-35
# VOS_test_png

# dataset_list = ['Visal','davis_test','Segtrack-v2']
dataset_list = ['Visal']
gg = 'bigxiu'
pt = ''
for a in range(0, len(dataset_list)):
    dataset_end = dataset_list[a]
    rootpath = r'F:\source\Visal17\36Smeasure_result\KFS_novalue/%s/%s/' % (pt,dataset_end)
    # rootpath = r'D:\source\VSOD_dataset/%s/' % (dataset_end)
    dataPath = rootpath
    seqList = os.listdir(dataPath)

    for i in range(0, len(seqList)):
        l = seqList[i]
        if not os.path.exists('./trained_txt/%s/%s/%s/' % (gg,pt,dataset_end)):
            os.makedirs('./trained_txt/%s/%s/%s/' % (gg,pt,dataset_end))

        outfile = './trained_txt/%s/%s/%s/%s.txt' % (gg,pt,dataset_end, l)
        # outfile = './trained_txt/KF5/%s/%s.txt' % (dataset_end, dataset_end)
        with open(outfile, "a") as file:
            s = dataPath + l
            #
            imgFiles = os.listdir(s)
            if len(imgFiles) != 0:
                #            print(l)
                length = int(len(imgFiles) / 4) * 4
                ##            print(length)
                for j in range(0, len(imgFiles)):
                    f = imgFiles[j]
                    print(f[:-4])
                    #
                    out_img = 'D:/source/VSOD_dataset/%s/%s/Imgs/%s' % (dataset_end, l, f[:-4] + '.jpg')
                    # 下面这行是rgb图的结尾是png 的时候
                    # out_img = r'D:\source\VSOD_dataset/%s/%s/%s' % (dataset_end, l, f[:-4] + '.jpg')
                    #
                    out_gt = r'F:\source\Visal17\36Smeasure_result\KFS_novalue/%s/%s/%s/%s' % (pt,dataset_end, l, f[:-4] + '.png')
                    file.write(out_img + ',' + out_gt + "\n")
            else:
                print('nog ood')
