from PIL import Image
import torch
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import  DataLoader
import torch.nn as nn
from torchvision import transforms
import os
import pandas as pd  
import math
import copy
from datetime import datetime

# In[12]:
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,root,transform=None):
        super(MyDataset,self).__init__()
        file_info = pd.read_csv(root, index_col=0)
        file_path = file_info['path']
        file_class = file_info['classes']
        imgs = []
        imglb = []
        for i in range(0,len(file_path)):
            # print(i)
            # print(len(file_path))
            path = file_path[i]
#            path = path.replace('\\','/')
#            path='../'+path
            if not os.path.isfile(path):
                print(path + '  does not exist!')
                return None
            img = Image.open(path).convert('RGB')
            imgs.append(img)
            imglb.append(int(file_class[i]))
        self.image = imgs
        self.imglb = imglb
        self.root = root
        self.size = len(file_info)
        print('size',self.size)
        self.transform = transform
        
    def __getitem__(self,index):
        img = self.image[index]
        label = self.imglb[index]
        sample = {'image': img,'classes':label}
        if self.transform:
            sample['image'] = self.transform(img)
        return sample
        
    def __len__(self):
        return self.size
# In[13]:
train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       ])
val_transforms = transforms.Compose([transforms.Resize((500, 500)),
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
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x

        
def train_model(model, criterion, optimizer, scheduler, video, num_epochs,csv_name,dataset):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_classes = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # print('epoch {}/{}'.format(epoch,num_epochs - 1))
        # print('-*' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_classes = 0
            
            for idx,data in enumerate(data_loaders[phase]):
                #print(phase+' processing: {}th batch.'.format(idx))
                inputs = Variable(data['image'].cuda())
                labels_classes = Variable(data['classes'].cuda())
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    x_classes = model(inputs)
                    #返回一行中的最大值，并返回索引
                    _, preds_classes = torch.max(x_classes, 1)

                    loss = criterion(x_classes, labels_classes)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                      format(datetime.now(), epoch, num_epochs, idx, len(data_loaders[phase]), loss.data))
#        
                corrects_classes += torch.sum(preds_classes == labels_classes)
            
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_classes

            Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
            print('{} Loss: {:.4f}  Acc_classes: {:.2%}'.format(phase, epoch_loss,epoch_acc_classes))
            if not os.path.exists('res18/%s_nofix_noaug_adp/%s/'%(csv_name,dataset)+video):
                os.makedirs('res18/%s_nofix_noaug_adp/%s/'%(csv_name,dataset)+video)
                
            if epoch%1==0:
                model.load_state_dict(copy.deepcopy(model.state_dict()))
                torch.save(model.state_dict(), 'res18/{}_nofix_noaug_adp/{}/{}/{}.pt'.format(csv_name,dataset,video,epoch))
#            
            if phase == 'val' and epoch_acc >= best_acc:

                best_acc = epoch_acc_classes
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val classes Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'res18/{}_nofix_noaug_adp/{}/{}/{}.pt'.format(csv_name,dataset,video,epoch))
    print('Best val classes Acc: {:.2%}'.format(best_acc))
    return model, Loss_list,Accuracy_list_classes

def resnet18(pretrained = False):
    model = ResNet(BasicBlock,[2,2,2,2])
    return model

# In[15]:
#   Segtrack-v2
dataset_list = ['Visal','Easy-35','VOS_test_png']
for aa in range(0,len(dataset_list)):
    dataset = dataset_list[aa]
    csv_name = 'adp_th_train_3_max_meanx2'
    train_csv_path = './csv_patch_train_adp/%s/%s/'%(dataset,csv_name)
    cvss = os.listdir(train_csv_path)
    for h in range(0,len(cvss)):
        video = cvss[h]
        print('video',video)
        TRAIN_ANNO = train_csv_path+'/%s'%(video)
        VAL_ANNO = './csv_patch_test_adp/%s/val/%s'%(dataset,video)
    
        CLASSES = ['bad', 'good']
        train_dataset = MyDataset(root = TRAIN_ANNO,transform = train_transforms)
        test_dataset = MyDataset(root = VAL_ANNO,transform = val_transforms)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset)
        print('train_loader',len(train_loader))
        data_loaders = {'train': train_loader, 'val': test_loader}

        model = resnet18()
        model_dict = model.state_dict()
        pretrained_dict = torch.load('./premodel/resnet18-5c106cde.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('加载预训练模型')
        # In[17]:
        
        # for name, param in model.named_parameters():
        #     if (name != 'fc1.weight') and (name != 'fc1.bias'):
        #         # print(param.requires_grad)
        #         param.requires_grad = False
        #         print(param.requires_grad)
                
        for name, param in model.named_parameters():
            print(name,param.requires_grad)
    
        # In[18]:
        network = model.cuda()
        optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Decay LR by a factor of 0.1 every 1 epochs
        save_video = video[:-4]
        model, Loss_list, Accuracy_list_classes = train_model(network, criterion, optimizer, exp_lr_scheduler, save_video,num_epochs=20, csv_name=csv_name, dataset=dataset)






