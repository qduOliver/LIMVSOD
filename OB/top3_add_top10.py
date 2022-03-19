import os

# davis_test
# Segtrack-v2
# Visal
# Easy-35
# VOS_test_png

dataset_list = ['VOS_test_png']
#dataset_list = ['Visal','VOS_test_png']

for sj in range(0,len(dataset_list)):

    dataset = dataset_list[sj]
    flo_txt_path_total = r'./txt/top3/%s/' % (dataset)
    rgb_txt_path_total = r'./txt/top10/%s/' % (dataset)

    videos = os.listdir(rgb_txt_path_total)

    for i in range(0, len(videos)):
        video = videos[i]
        rgb_imgs_path = rgb_txt_path_total + video
        print('rgb_imgs_path', rgb_imgs_path)
        rgb_imgs = os.listdir(rgb_imgs_path)

        for j in range(0, len(rgb_imgs)):

            img_name = rgb_imgs[j]
            print('img_name', img_name)
            rgb_txt_path = rgb_imgs_path + '/' + img_name
            flo_txt_path = flo_txt_path_total + video + '/' + img_name[:-8] + '.jpg.txt'
            # flo_txt_path = flo_txt_path_total + video + '/' + img_name

            if not os.path.exists(flo_txt_path):
                continue

            rgb_txt_file = open(rgb_txt_path)
            flo_txt_file = open(flo_txt_path)

            rgb_txt_lines = rgb_txt_file.readlines()
            flo_txt_lines = flo_txt_file.readlines()

            save_path = r'./txt/total/%s/%s' % (dataset, video)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            f = open(save_path + '/' + img_name, 'a')

            for j in range(0, len(rgb_txt_lines)):
                rgb_txt_line = rgb_txt_lines[j]
                f.write(rgb_txt_line)

            for j in range(0, len(flo_txt_lines)):
                flo_txt_line = flo_txt_lines[j]
                f.write(flo_txt_line)

            f.close()

