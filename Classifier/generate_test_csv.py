import os
import csv

dataset_list = ['Visal']
for aa in range(0, len(dataset_list)):
    dataset = dataset_list[aa]
    # PATH=r'G:\spyder_workpeace_song\total\Yet-Another-EfficientDet-Pytorch-master\txt2patch4/%s/'%(dataset)
    PATH = r'F:\source\Visal17\12julei0_1_test/%s/' % (dataset)

    videos = os.listdir(PATH)
    for i in range(0, len(videos)):
        video = videos[i]
        cla_path = PATH + video
        clas = os.listdir(cla_path)
        num = 0
        csv_path = r'F:\source\Visal17/13test_csv_julei8_first/%s/' % (dataset)
        # csv_path = './csv_patch_test_adp/%s/test_all/' % (dataset)

        if not os.path.exists(csv_path):
            os.makedirs(csv_path)

        with open(csv_path + '/%s.csv' % (video), 'a', newline="", encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            #        string = ''',path,classes'''
            #        string=string.strip('''''')
            # string=eval(string)
            path = 'path'
            classes = 'classes'
            writer.writerow([path, classes])

            for cl in range(0, len(clas)):
                cla = clas[cl]
                img_path = cla_path + '/' + cla
                imgs = os.listdir(img_path)

                for j in range(0, len(imgs)):
                    img = imgs[j]
                    img_name = img
                    print(img_name)
                    #        img_label_reg=img[-9:-4]
                    #        nan=img[-7:-4]
                    img_label_cla = cla[0]

                    #        img_label_reg=0.000
                    #        img_label_cla=img[-5:-4]
                    print(img_label_cla)

                    flo_path = img_path + '/%s' % (img_name)

                    writer.writerow([num, flo_path, img_label_cla])
                    num = num + 1
