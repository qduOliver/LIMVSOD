import os
import cv2

# FBMS-59/FBMS-Test
# Segtrack-v2
# Visal
# davis_test
# Easy-35
# VOS_test_png   VOS_test_png_gt

dataset_list = ['VOS_test_png']
for ss in range(0, len(dataset_list)):
    dataset = dataset_list[ss]
# img_total_path = r'D:\dataset\video/%s/VOS_test_png_gt/'%(dataset)
    img_total_path = r'D:\source\VSOD_dataset/%s/' % (dataset)
    # txt_total_path = r'D:\code\pycharm_workpeace\Yet-Another-EfficientDet-Pytorch-master\txt\new_1_total/%s/'%(dataset)
    txt_total_path = r'.\nms/txt/%s/' % (dataset)
    folers = os.listdir(txt_total_path)
    aa = len(folers)
    bb = len(txt_total_path)
    for i in range(0, len(folers)):
        folder = folers[i]
        imgs = os.listdir(img_total_path + folder + '/Imgs/')

        for k in range(0, len(imgs)):
            img_name = imgs[k]
            img_path = img_total_path + folder + '/Imgs/' + img_name
            img = cv2.imread(img_path)
            h = img.shape[0]
            w = img.shape[1]
            img_area = h * w
            print(img_name)
            # txt_path=txt_total_path+folder+'/'+img_name+'.txt'
            txt_path = txt_total_path + folder + '/' + img_name[:-4] + '.png' + '.txt'
            if os.path.exists(txt_path):
                f = open(txt_path)
                fmany = f.readlines()
                for j in range(0, len(fmany)):
                    line = fmany[j]
                    w = line.split(',')
                    num = w[0]
                    x1 = int(w[1])
                    y1 = int(w[2])
                    x2 = int(w[3])
                    y2 = int(w[4])
                    obj = w[6]
                    score = float(w[7])
                    n = img_total_path + folder + '/Imgs/' + img_name
                    width = x2 - x1
                    height = y2 - y1
                    # if (width < 17) or (height < 24):
                    #     continue
                    # if (height/width) > 3.5:
                    #     continue
                    #
                    # patch_area = width*height
                    # bizhong = patch_area/img_area
                    # if bizhong > 0.5:
                    #     continue
                    img = cv2.imread(n)
                    if y1 == y2 or x1 == x2:
                        continue
                    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                        continue
                    cropped = img[y1:y2, x1:x2]
                    if len(cropped) == 0:
                        continue

                    save_path = r'.\txt2patch4/%s/%s' % (dataset, folder)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    cv2.imwrite("%s/%s_%s_%.03f_z_%s_%s_%s_%s.jpg" % (save_path, img_name, obj, score, x1, y1, x2, y2),
                                cropped)
