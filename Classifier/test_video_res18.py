from PIL import Image
import torch
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import  DataLoader
import torch.nn as nn
from torchvision.transforms import transforms

import os
import pandas as pd
import math

import cv2
import time
# In[12]:

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root,transform=None):
        super(MyDataset,self).__init__()
        file_info = pd.read_csv(root, index_col=0)
        
        file_path = file_info['path']
        file_class = file_info['classes']
        imgs = []
        imgs_name = []
        imglb = []
        cla_names = []
        shapes = []
        for i in range(0,len(file_path)):
            path = file_path[i]
            path = path.replace('\\','/')
#            path='../'+path
            if not os.path.isfile(path):
                print(path + '  does not exist!')
                return None
            img = Image.open(path).convert('RGB')
            shape = img.size
            img_name=path.split('/')[-1]
            cla_name = path.split('/')[-2]
            imgs_name.append(img_name)
            cla_names.append(cla_name)
            imgs.append(img)
            imglb.append(int(file_class[i]))
            shapes.append(shape)
        self.image = imgs
        self.img_names=imgs_name
        self.imglb = imglb
        self.root = root
        self.size = len(file_info)
        self.transform = transform
        self.imglb
        self.cla_names = cla_names
        self.shapes = shapes
        
    def __getitem__(self,index):
        img = self.image[index]
        label = self.imglb[index]
        img_name=self.img_names[index]
        cla_name = self.cla_names[index]
        shape = self.shapes[index]
        sample = {'image': img,'classes':label,'img_name':img_name,'cla_name':cla_name,'shape':shape}
        if self.transform:
            sample['image'] = self.transform(img)
        return sample
        
    def __len__(self):
        return self.size


# In[13]:
test_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor()
                                     ])

# In[14]:
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 
                     kernel_size = 3,stride = stride, 
                     padding = 1, bias = False)
    
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0, ceil_mode = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(512 * block.expansion * 4, num_classes)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
              nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size = 1, stride = stride, bias = False),
              nn. BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        print('conv1',x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print('maxpool',x.shape)
        
        x = self.layer1(x)
        print('layer1',x.shape)
        x = self.layer2(x)
        print('layer2',x.shape)
        x = self.layer3(x)
        print('layer3',x.shape)
        x = self.layer4(x)
        print('layer4',x.shape)
        
        x = self.avgpool(x)
        print('avgpool',x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        print('fc1',x.shape)
        
        return x

# In[17]:
def test_model(model,weight_path,dataset,test_data,video):
    Loss_list = {'test': []}
    Accuracy_list_classes = {'test': []}
        # Each epoch has a training and validation phase
    for phase in ['test']:
        
        model.eval()
        model.state_dict()
        
        running_loss = 0.0
        corrects_classes = 0
        count_good_num=0
        pre_good_num=0
        corrects_classes_1=0.0
        
#        print('video',video)

        for idx,data in enumerate(data_loaders[phase]):
            #print(phase+' processing: {}th batch.'.format(idx))
            inputs = Variable(data['image'].cuda())
#            print(data['image'])
#             print(inputs.shape)
            img_name=data['img_name']
            labels_classes = Variable(data['classes'].cuda())
            t1 = time.time()
            x_classes = model(inputs)
            t2 = time.time()
            print('euntime',t2-t1)
            save_video = video[:-4]
#            print('save_video',save_video)
            cla = data['cla_name'][0]
            save_path=r'D:/%s/%s/best_model/%s_result/%s'%(weight_path,dataset,test_data,video[:-4])

            if not os.path.exists(save_path):
                os.makedirs(save_path)
#
            #x_classes = x_classes.view(-1, 2)
            _, preds_classes = torch.max(x_classes, 1)
            corrects_classes += torch.sum(preds_classes == labels_classes)
            
            if labels_classes == 1:
                    count_good_num = count_good_num+1
            
            if preds_classes == 1:
                pre_good_num = pre_good_num+1
                if(preds_classes == labels_classes):
                    corrects_classes_1 = corrects_classes_1+1
                    
#            print('preds_classes',preds_classes.data.cpu().numpy())
#            print('labels_classes',labels_classes.data.cpu().numpy())
            preds_classes=preds_classes.data.cpu().numpy()[0]
            labels_classes=labels_classes.data.cpu().numpy()[0]
            # aa=save_path+'/'+img_name[0][:-4]+'_'+str(preds_classes)+'_'+str(labels_classes)+'.jpg'
            aa=save_path+'/'+img_name[0][:-4]+'_t3_'+str(preds_classes)+'.jpg'
#            print(aa)
#            print('aa',aa)
            print('img_name',img_name)
            img=data['image'].data.cpu().numpy().squeeze()*255
            img=img.transpose((1,2,0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            shape = data['shape']
            img = cv2.resize(img, (shape[0],shape[1]))
            # print('save_shape',img.shape)
            cv2.imwrite(aa,img)
            
        epoch_loss = running_loss / len(data_loaders[phase].dataset)
        Loss_list[phase].append(epoch_loss)

        epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
        
        Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
        
        print('{} Loss: {:.4f}  Acc_classes: {:.2%}'.format(phase, epoch_loss,epoch_acc_classes))

        # pre=corrects_classes_1+1/pre_good_num+1
        # print('pre',pre)
        
        # recall=corrects_classes_1/count_good_num
        # print('recall',recall)
    return model, Loss_list,Accuracy_list_classes

def resnet18(pretrained = False):
    model = ResNet(BasicBlock,[2,2,2,2])
    return model
# In[15]:
    
dataset = 'Visal'
test_data = 'test_all'
total_path ='./csv_patch_test_adp/%s/%s/'%(dataset,test_data)

videos = os.listdir(total_path) 
for i in range(0,len(videos)):
    
    video =videos[i]
    test_anno = total_path + video
    CLASSES = ['bad', 'good']
    
    test_dataset = MyDataset(root = test_anno,transform = test_transforms)
    
    train_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset)
    data_loaders = {'test': test_loader}

    # In[16]:
    model = resnet18()
    
    weight_path = 'adp_th_train_2_max_meanx2_nofix_noaug_adp'
    pretrained_dict = torch.load('./res18/%s/%s/%s/best_model.pt'%(weight_path,dataset,video[:-4]))
    # pretrained_dict = torch.load('./res18/%s/%s/15.pt' % (weight_path, 'davis_test'))
    model.load_state_dict(pretrained_dict)
    print('加载预训练模型')

    # In[18]:
    network = model.cuda()
    model, Loss_list, Accuracy_list_classes = test_model(network,weight_path,dataset,test_data,video)

