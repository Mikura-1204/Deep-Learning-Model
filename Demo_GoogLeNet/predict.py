import os
import json
 
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
 
from model import GoogLeNet

def main():
    # 如果有NVIDA显卡,转到GPU训练，否则用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    # 将多个transforms的操作整合在一起
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
    # 加载图片
    # ciallo: change this for your test dir
    img_path = "./dataset/test/animal.jpg"
    # 确定图片存在，否则反馈错误
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # imshow()：对图像进行处理并显示其格式，show()则是将imshow()处理后的函数显示出来
    # ciallo: Server not show !
    # plt.imshow(img)
    # [C, H, W]，转换图像格式
    img = data_transform(img)
    # [N, C, H, W]，增加一个维度N 代表batch
    img = torch.unsqueeze(img, dim=0)
 
    # 获取结果类型
    json_path = './class_indices.json'
    # 确定路径存在，否则反馈错误
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    # 读取内容
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    
    # ciallo: 以下是模型推理具体过程
    # 模型实例化，将模型转到device，结果类型有5种
    # 实例化模型时不需要辅助分类器
    model = GoogLeNet(num_classes=5, aux_logits=False).to(device)
    # 载入模型权重
    weights_path = "./param/googleNet.pth"
    # 确定模型存在，否则反馈错误
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # 在加载训练好的模型参数时，由于其中是包含有辅助分类器的，需要设置strict=False舍弃不需要的参数
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)
    # 进入验证阶段
    model.eval()
    with torch.no_grad():
        # 预测类别
        # squeeze()：维度压缩，返回一个tensor（张量），其中input中大小为1的所有维都已删除
        output = torch.squeeze(model(img.to(device))).cpu()
        # softmax：归一化指数函数，将预测结果输入进行非负性和归一化处理，最后将某一维度值处理为0-1之内的分类概率
        predict = torch.softmax(output, dim=0)
        # argmax(input)：返回指定维度最大值的序号
        # .numpy()：把tensor转换成numpy的格式
        predict_cla = torch.argmax(predict).numpy()
 
    # 输出的预测值与真实值
    # ciallo: output is all the probability of classes
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    
    # 图片标题
    # Server not show !
    # plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    # ciallo: this is final result
    print("Ciallo(∠・ω< )⌒☆      Class: ", class_indict[str(predict_cla)], "      Probability: ", predict[predict_cla].numpy())
    # plt.show()
 
 
if __name__ == '__main__':
    main()