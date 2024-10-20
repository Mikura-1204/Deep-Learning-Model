import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim

from VGG_Net import VGG_16   #引入网络

##预训练下载存放下载权重文件
Model_Urls={}                               

##三元运算判断调用的训练设备 有可用的GPU则输出 cuda：0
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(Device)

##训练集集合处理
Data_Transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),    #随机裁剪图像到指定大小(224x224)，并进行缩放和填充。
                                 transforms.RandomHorizontalFlip(), #以一定概率随机水平翻转图像。
                                 transforms.RandomAffine(degrees=(30,70),translate=(0.1, 0.3),fill=(255, 255, 255)),      #随机旋转 平移 填充
                                 transforms.ToTensor(),                #将图像数据转换为张量（Tensor）格式。
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  #对图像的每个通道进行标准化，即减去均值、除以标准差，以便模型更好地进行训练
    "val": transforms.Compose([transforms.Resize(256),                # 将图像的短边调整为256像素，保持原始图像的宽高比不变。
                               transforms.CenterCrop(224),            #从调整后的图像中心裁剪出一个大小为224x224的正方形图像。
                               transforms.ToTensor(),  #将图像转换为Tensor数据类型，即将图像转换为在[0, 1]范围内的浮点数张量。同时，图像的通道顺序也会从常见的RGB（红绿蓝）转换为PyTorch默认的BGR（蓝绿红）顺序
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

##训练数据路径及参数设定
Image_Path = ''

Train_dataset = datasets.ImageFolder(root=Image_Path+"/train",
                                     transform=Data_Transform["train"]) 
Train_num = len(Train_dataset)                      #训练集数量
print("Splited Train Data Set Num: " + Train_num)   #输出训练集数量

##花卉数据集分类及json字典建立
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = Train_dataset.class_to_idx    #train_dataset.class_to_idx是一个字典，它将花卉类别映射到索引。例如，键值对'daisy': 0表示"daisy"类别对应索引为0。
cla_dict = dict((val, key) for key, val in flower_list.items())  #创建了一个新的字典cla_dict，其中键是索引，值是对应的花卉类别。这是为了方便后续根据索引查找花卉类别。
 #write dict into json file
json_str = json.dumps(cla_dict, indent=4)     #将cla_dict转换为格式化后的JSON字符串。indent=4参数用于指定缩进为4个空格，使得生成的JSON文件更易读。
with open('class_indices.json', 'w') as json_file:   #代码打开一个名为"class_indices.json"的文件，并将JSON字符串写入该文件中。这样就将字典保存为一个JSON文件了。
    json_file.write(json_str)

##验证集划分
batch_size = 16
train_loader = torch.utils.data.DataLoader(Train_dataset,
                                           batch_size=batch_size, shuffle=True,     #打乱随机
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=Image_Path + "/val",
                                        transform=Data_Transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)
print("Splited Validate Data Set Num: " + val_num)   #输出验证集数量

##加载网络并开始训练
net = VGG_16()
net.to(Device)

loss_function = nn.CrossEntropyLoss()   #交叉熵损失函数-用于多分类
optimizer = optim.Adam(net.parameters(), lr=0.0001)
best_acc = 0.0
save_path = './model/VGG_16.pth'    #权重文件路径

for epoch in range(3):              #训练轮数
    # train
    net.train()
    running_loss = 0.0
    train_acc = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data  #将输入数据和标签移动到指定的计算设备上
        optimizer.zero_grad()
        logits = net(images.to(Device))
        loss = loss_function(logits, labels.to(Device))   #通过前向传播获得模型的输出logits，并计算损失值loss。然后进行反向传播，计算梯度并更新模型的参数optimizer.step()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step+1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    test_acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(Device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]  #随后根据模型输出得到预测类别 predict_y
            test_acc += (predict_y == val_labels.to(Device)).sum().item()
        val_accurate = test_acc / val_num  #最终计算出在当前批次上模型的准确率，并将其累加到总准确率 acc
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')