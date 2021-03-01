import os

# davis_test
# Segtrack-v2
# Visal
# Easy-35
# VOS_test_png

dataset = 'Easy-35'
flo_txt_path_total = r'D:\code\pycharm_workpeace\Yet-Another-EfficientDet-Pytorch-master\txt\new_1_flo/%s/M7_0.01_0.2_top3/'%(dataset)
rgb_txt_path_total = r'D:\code\pycharm_workpeace\Yet-Another-EfficientDet-Pytorch-master\txt\new_1/%s/M7_0.01_0.2_top10/'%(dataset)

videos = os.listdir(rgb_txt_path_total)

for i in range(0,len(videos)):
    video = videos[i]
    rgb_imgs_path = rgb_txt_path_total + video
    print('rgb_imgs_path',rgb_imgs_path)
    rgb_imgs = os.listdir(rgb_imgs_path)
    
    for j in range(0,len(rgb_imgs)):
        
        img_name = rgb_imgs[j]
        print('img_name',img_name)
        rgb_txt_path = rgb_imgs_path + '/' + img_name
        flo_txt_path = flo_txt_path_total + video + '/' + img_name[:4] + '.jpg.txt'
        # flo_txt_path = flo_txt_path_total + video + '/' + img_name
        
        if not os.path.exists(flo_txt_path):
            continue
        
        rgb_txt_file = open(rgb_txt_path)
        flo_txt_file = open(flo_txt_path)
        
        rgb_txt_lines = rgb_txt_file.readlines()
        flo_txt_lines = flo_txt_file.readlines()
        
        save_path = r'D:\code\pycharm_workpeace\Yet-Another-EfficientDet-Pytorch-master\txt\new_1_total/%s/%s'%(dataset,video)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        f = open(save_path + '/' + img_name,'a')
        
        for j in range(0,len(rgb_txt_lines)):
            rgb_txt_line = rgb_txt_lines[j]
            f.write(rgb_txt_line)
            
        for j in range(0,len(flo_txt_lines)):
            flo_txt_line = flo_txt_lines[j]
            f.write(flo_txt_line) 
        
    # f.close()
    