---
title: Dog classification
copyright: true
permalink: 1
top: 0
date: 2019-06-26 13:58:32
tags: python
categories: Deep learning
password:
mathjax: true
---

## 卷积神经网络（Convolutional Neural Network, CNN）

## 项目：实现一个狗品种识别算法App

在这个notebook文件中，有些模板代码已经提供给你，但你还需要实现更多的功能来完成这个项目。除非有明确要求，你无须修改任何已给出的代码。以**'(练习)'**开始的标题表示接下来的代码部分中有你需要实现的功能。这些部分都配有详细的指导，需要实现的部分也会在注释中以'TODO'标出。请仔细阅读所有的提示。

除了实现代码外，你还**需要**回答一些与项目及代码相关的问题。每个需要回答的问题都会以 **'问题 X'** 标记。请仔细阅读每个问题，并且在问题后的 **'回答'** 部分写出完整的答案。我们将根据 你对问题的回答 和 撰写代码实现的功能 来对你提交的项目进行评分。

>**提示：**Code 和 Markdown 区域可通过 **Shift + Enter** 快捷键运行。此外，Markdown可以通过双击进入编辑模式。

项目中显示为_选做_的部分可以帮助你的项目脱颖而出，而不是仅仅达到通过的最低要求。如果你决定追求更高的挑战，请在此 notebook 中完成_选做_部分的代码。

---

### 让我们开始吧
在这个notebook中，你将迈出第一步，来开发可以作为移动端或 Web应用程序一部分的算法。在这个项目的最后，你的程序将能够把用户提供的任何一个图像作为输入。如果可以从图像中检测到一只狗，它会输出对狗品种的预测。如果图像中是一个人脸，它会预测一个与其最相似的狗的种类。下面这张图展示了完成项目后可能的输出结果。（……实际上我们希望每个学生的输出结果不相同！）

![Sample Dog Output](images/sample_dog_output.png)

在现实世界中，你需要拼凑一系列的模型来完成不同的任务；举个例子，用来预测狗种类的算法会与预测人类的算法不同。在做项目的过程中，你可能会遇到不少失败的预测，因为并不存在完美的算法和模型。你最终提交的不完美的解决方案也一定会给你带来一个有趣的学习经验！

### 项目内容

我们将这个notebook分为不同的步骤，你可以使用下面的链接来浏览此notebook。

* [Step 0](#step0): 导入数据集
* [Step 1](#step1): 检测人脸
* [Step 2](#step2): 检测狗狗
* [Step 3](#step3): 从头创建一个CNN来分类狗品种
* [Step 4](#step4): 使用一个CNN来区分狗的品种(使用迁移学习)
* [Step 5](#step5): 建立一个CNN来分类狗的品种（使用迁移学习）
* [Step 6](#step6): 完成你的算法
* [Step 7](#step7): 测试你的算法

在该项目中包含了如下的问题：

* [问题 1](#question1)
* [问题 2](#question2)
* [问题 3](#question3)
* [问题 4](#question4)
* [问题 5](#question5)
* [问题 6](#question6)
* [问题 7](#question7)
* [问题 8](#question8)
* [问题 9](#question9)
* [问题 10](#question10)
* [问题 11](#question11)


---
<a id='step0'></a>
## 步骤 0: 导入数据集

### 导入狗数据集
在下方的代码单元（cell）中，我们导入了一个狗图像的数据集。我们使用 scikit-learn 库中的 `load_files` 函数来获取一些变量：
- `train_files`, `valid_files`, `test_files` - 包含图像的文件路径的numpy数组
- `train_targets`, `valid_targets`, `test_targets` - 包含独热编码分类标签的numpy数组
- `dog_names` - 由字符串构成的与标签相对应的狗的种类


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# 定义函数来加载train，test和validation数据集
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# 加载train，test和validation数据集
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# 加载狗品种列表
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# 打印数据统计描述
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    Using TensorFlow backend.


    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


### 导入人脸数据集

在下方的代码单元中，我们导入人脸图像数据集，文件所在路径存储在名为 `human_files` 的 numpy 数组。


```python
import random
random.seed(8675309)

# 加载打乱后的人脸数据集的文件名
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# 打印数据集的数据量
print('There are %d total human images.' % len(human_files))
```

    There are 13233 total human images.


---
<a id='step1'></a>
## 步骤1：检测人脸
 
我们将使用 OpenCV 中的 [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) 来检测图像中的人脸。OpenCV 提供了很多预训练的人脸检测模型，它们以XML文件保存在 [github](https://github.com/opencv/opencv/tree/master/data/haarcascades)。我们已经下载了其中一个检测模型，并且把它存储在 `haarcascades` 的目录中。

在如下代码单元中，我们将演示如何使用这个检测模型在样本图像中找到人脸。


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# 提取预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# 加载彩色（通道顺序为BGR）图像
img = cv2.imread(human_files[3])

# 将BGR图像进行灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 在图像中找出脸
faces = face_cascade.detectMultiScale(gray)

# 打印图像中检测到的脸的个数
print('Number of faces detected:', len(faces))

# 获取每一个所检测到的脸的识别框
for (x,y,w,h) in faces:
    # 在人脸图像中绘制出识别框
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# 将BGR图像转变为RGB图像以打印
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 展示含有识别框的图像
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](output_5_1.png)


在使用任何一个检测模型之前，将图像转换为灰度图是常用过程。`detectMultiScale` 函数使用储存在 `face_cascade` 中的的数据，对输入的灰度图像进行分类。

在上方的代码中，`faces` 以 numpy 数组的形式，保存了识别到的面部信息。它其中每一行表示一个被检测到的脸，该数据包括如下四个信息：前两个元素  `x`、`y` 代表识别框左上角的 x 和 y 坐标（参照上图，注意 y 坐标的方向和我们默认的方向不同）；后两个元素代表识别框在 x 和 y 轴两个方向延伸的长度 `w` 和 `d`。 

### 写一个人脸识别器

我们可以将这个程序封装为一个函数。该函数的输入为人脸图像的**路径**，当图像中包含人脸时，该函数返回 `True`，反之返回 `False`。该函数定义如下所示。


```python
# 如果img_path路径表示的图像检测到了脸，返回"True" 
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### **【练习】** 评估人脸检测模型


---

<a id='question1'></a>
### __问题 1:__ 

在下方的代码块中，使用 `face_detector` 函数，计算：

- `human_files` 的前100张图像中，能够检测到**人脸**的图像占比多少？
- `dog_files` 的前100张图像中，能够检测到**人脸**的图像占比多少？

理想情况下，人图像中检测到人脸的概率应当为100%，而狗图像中检测到人脸的概率应该为0%。你会发现我们的算法并非完美，但结果仍然是可以接受的。我们从每个数据集中提取前100个图像的文件路径，并将它们存储在`human_files_short`和`dog_files_short`中。


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
## 请不要修改上方代码


## TODO: 基于human_files_short和dog_files_short
## 中的图像测试face_detector的表现
face_result = np.vectorize(face_detector)
human_face_result = face_result(human_files_short)
dog_face_result = face_result(dog_files_short)

print("human_files的前一百张图像中，能够检测到的人脸图像比例：%.4f%%" %(float(sum(human_face_result)/len(human_files_short))*100))
print("dog_files的前一百张图像中，能够检测到的人脸图像比例：%.4f%%" %(float(sum(dog_face_result)/len(dog_files_short))*100))
```

    human_files的前一百张图像中，能够检测到的人脸图像比例：99.0000%
    dog_files的前一百张图像中，能够检测到的人脸图像比例：11.0000%


---

<a id='question2'></a>

### __问题 2:__ 

就算法而言，该算法成功与否的关键在于，用户能否提供含有清晰面部特征的人脸图像。
那么你认为，这样的要求在实际使用中对用户合理吗？如果你觉得不合理，你能否想到一个方法，即使图像中并没有清晰的面部特征，也能够检测到人脸？

__回答:__

不合理，有些时候一个的脸部没有完全出现在图片中，将会没有清晰的面部特征

使用卷积神经网络，通过训练好的网络提取特征，然后用作分类

---

<a id='Selection1'></a>
### 选做：

我们建议在你的算法中使用opencv的人脸检测模型去检测人类图像，不过你可以自由地探索其他的方法，尤其是尝试使用深度学习来解决它:)。请用下方的代码单元来设计和测试你的面部监测算法。如果你决定完成这个_选做_任务，你需要报告算法在每一个数据集上的表现。


```python
## (选做) TODO: 报告另一个面部检测算法在LFW数据集上的表现
### 你可以随意使用所需的代码单元数
```

---
<a id='step2'></a>

## 步骤 2: 检测狗狗

在这个部分中，我们使用预训练的 [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) 模型去检测图像中的狗。下方的第一行代码就是下载了 ResNet-50 模型的网络结构参数，以及基于 [ImageNet](http://www.image-net.org/) 数据集的预训练权重。

ImageNet 这目前一个非常流行的数据集，常被用来测试图像分类等计算机视觉任务相关的算法。它包含超过一千万个 URL，每一个都链接到 [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 中所对应的一个物体的图像。任给输入一个图像，该 ResNet-50 模型会返回一个对图像中物体的预测结果。


```python
from keras.applications.resnet50 import ResNet50

# 定义ResNet50模型
ResNet50_model = ResNet50(weights='imagenet')
```

### 数据预处理

- 在使用 TensorFlow 作为后端的时候，在 Keras 中，CNN 的输入是一个4维数组（也被称作4维张量），它的各维度尺寸为 `(nb_samples, rows, columns, channels)`。其中 `nb_samples` 表示图像（或者样本）的总数，`rows`, `columns`, 和 `channels` 分别表示图像的行数、列数和通道数。


- 下方的 `path_to_tensor` 函数实现如下将彩色图像的字符串型的文件路径作为输入，返回一个4维张量，作为 Keras CNN 输入。因为我们的输入图像是彩色图像，因此它们具有三个通道（ `channels` 为 `3`）。
    1. 该函数首先读取一张图像，然后将其缩放为 224×224 的图像。
    2. 随后，该图像被调整为具有4个维度的张量。
    3. 对于任一输入图像，最后返回的张量的维度是：`(1, 224, 224, 3)`。


- `paths_to_tensor` 函数将图像路径的字符串组成的 numpy 数组作为输入，并返回一个4维张量，各维度尺寸为 `(nb_samples, 224, 224, 3)`。 在这里，`nb_samples`是提供的图像路径的数据中的样本数量或图像数量。你也可以将 `nb_samples` 理解为数据集中3维张量的个数（每个3维张量表示一个不同的图像。


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # 用PIL加载RGB图像为PIL.Image.Image类型
    img = image.load_img(img_path, target_size=(224, 224))
    # 将PIL.Image.Image类型转化为格式为(224, 224, 3)的3维张量
    x = image.img_to_array(img)
    # 将3维张量转化为格式为(1, 224, 224, 3)的4维张量并返回
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### 基于 ResNet-50 架构进行预测

对于通过上述步骤得到的四维张量，在把它们输入到 ResNet-50 网络、或 Keras 中其他类似的预训练模型之前，还需要进行一些额外的处理：
1. 首先，这些图像的通道顺序为 RGB，我们需要重排他们的通道顺序为 BGR。
2. 其次，预训练模型的输入都进行了额外的归一化过程。因此我们在这里也要对这些张量进行归一化，即对所有图像所有像素都减去像素均值 `[103.939, 116.779, 123.68]`（以 RGB 模式表示，根据所有的 ImageNet 图像算出）。

导入的 `preprocess_input` 函数实现了这些功能。如果你对此很感兴趣，可以在 [这里](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py) 查看 `preprocess_input`的代码。


在实现了图像处理的部分之后，我们就可以使用模型来进行预测。这一步通过 `predict` 方法来实现，它返回一个向量，向量的第 i 个元素表示该图像属于第 i 个 ImageNet 类别的概率。这通过如下的 `ResNet50_predict_labels` 函数实现。

通过对预测出的向量取用 argmax 函数（找到有最大概率值的下标序号），我们可以得到一个整数，即模型预测到的物体的类别。进而根据这个 [清单](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)，我们能够知道这具体是哪个品种的狗狗。



```python
from keras.applications.resnet50 import preprocess_input, decode_predictions
def ResNet50_predict_labels(img_path):
    # 返回img_path路径的图像的预测向量
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### 完成狗检测模型


在研究该 [清单](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 的时候，你会注意到，狗类别对应的序号为151-268。因此，在检查预训练模型判断图像是否包含狗的时候，我们只需要检查如上的 `ResNet50_predict_labels` 函数是否返回一个介于151和268之间（包含区间端点）的值。

我们通过这些想法来完成下方的 `dog_detector` 函数，如果从图像中检测到狗就返回 `True`，否则返回 `False`。


```python
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### 【作业】评估狗狗检测模型

---

<a id='question3'></a>
### __问题 3:__ 

在下方的代码块中，使用 `dog_detector` 函数，计算：

- `human_files_short`中图像检测到狗狗的百分比？
- `dog_files_short`中图像检测到狗狗的百分比？


```python
### TODO: 测试dog_detector函数在human_files_short和dog_files_short的表现
dog_result = np.vectorize(dog_detector)
human_picture_result = dog_result(human_files_short)
dog_picture_result = dog_result(dog_files_short)

print("human_files的前一百张图像中，能够检测到的狗狗比例：%.4f%%" %(float(sum(human_picture_result)/len(human_files_short))*100))
print("dog_files的前一百张图像中，能够检测到的狗狗比例：%.4f%%" %(float(sum(dog_picture_result)/len(dog_files_short))*100))
```

    human_files的前一百张图像中，能够检测到的狗狗比例：1.0000%
    dog_files的前一百张图像中，能够检测到的狗狗比例：100.0000%


---

<a id='step3'></a>

## 步骤 3: 从头开始创建一个CNN来分类狗品种


现在我们已经实现了一个函数，能够在图像中识别人类及狗狗。但我们需要更进一步的方法，来对狗的类别进行识别。在这一步中，你需要实现一个卷积神经网络来对狗的品种进行分类。你需要__从头实现__你的卷积神经网络（在这一阶段，你还不能使用迁移学习），并且你需要达到超过1%的测试集准确率。在本项目的步骤五种，你还有机会使用迁移学习来实现一个准确率大大提高的模型。

在添加卷积层的时候，注意不要加上太多的（可训练的）层。更多的参数意味着更长的训练时间，也就是说你更可能需要一个 GPU 来加速训练过程。万幸的是，Keras 提供了能够轻松预测每次迭代（epoch）花费时间所需的函数。你可以据此推断你算法所需的训练时间。

值得注意的是，对狗的图像进行分类是一项极具挑战性的任务。因为即便是一个正常人，也很难区分布列塔尼犬和威尔士史宾格犬。


布列塔尼犬（Brittany） | 威尔士史宾格犬（Welsh Springer Spaniel）
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

不难发现其他的狗品种会有很小的类间差别（比如金毛寻回犬和美国水猎犬）。


金毛寻回犬（Curly-Coated Retriever） | 美国水猎犬（American Water Spaniel）
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">

同样，拉布拉多犬（labradors）有黄色、棕色和黑色这三种。那么你设计的基于视觉的算法将不得不克服这种较高的类间差别，以达到能够将这些不同颜色的同类狗分到同一个品种中。

黄色拉布拉多犬（Yellow Labrador） | 棕色拉布拉多犬（Chocolate Labrador） | 黑色拉布拉多犬（Black Labrador）
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

我们也提到了随机分类将得到一个非常低的结果：不考虑品种略有失衡的影响，随机猜测到正确品种的概率是1/133，相对应的准确率是低于1%的。

请记住，在深度学习领域，实践远远高于理论。大量尝试不同的框架吧，相信你的直觉！当然，玩得开心！


### 数据预处理


通过对每张图像的像素值除以255，我们对图像实现了归一化处理。


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# Keras中的数据预处理过程
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|████████████████████████████████████████████████████| 6680/6680 [00:58<00:00, 114.97it/s]
    100%|██████████████████████████████████████████████████████| 835/835 [00:06<00:00, 121.39it/s]
    100%|██████████████████████████████████████████████████████| 836/836 [00:06<00:00, 124.04it/s]


### 【练习】模型架构


创建一个卷积神经网络来对狗品种进行分类。在你代码块的最后，执行 `model.summary()` 来输出你模型的总结信息。
    
我们已经帮你导入了一些所需的 Python 库，如有需要你可以自行导入。如果你在过程中遇到了困难，如下是给你的一点小提示——该模型能够在5个 epoch 内取得超过1%的测试准确率，并且能在CPU上很快地训练。

![Sample CNN](images/sample_cnn.png)

---

<a id='question4'></a>  

### __问题 4:__ 

在下方的代码块中尝试使用 Keras 搭建卷积网络的架构，并回答相关的问题。

1. 你可以尝试自己搭建一个卷积网络的模型，那么你需要回答你搭建卷积网络的具体步骤（用了哪些层）以及为什么这样搭建。
2. 你也可以根据上图提示的步骤搭建卷积网络，那么请说明为何如上的架构能够在该问题上取得很好的表现。

__回答:__
CNN通过卷积层提取特征，多次进行卷积操作有利于提取中更多特征用于图像分类。
通过池化层减少自由参数个数，防止过拟合。


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: 定义你的网络架构
model.add(Conv2D(16,(2,2),input_shape=(224,224,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(64,(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(GlobalAveragePooling2D())
model.add(Dense(units=133,activation='softmax'))

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 223, 223, 16)      208       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 111, 111, 16)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 110, 110, 32)      2080      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 55, 55, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 54, 54, 64)        8256      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 27, 27, 64)        0         
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 64)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 133)               8645      
    =================================================================
    Total params: 19,189.0
    Trainable params: 19,189.0
    Non-trainable params: 0.0
    _________________________________________________________________



```python
## 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## 【练习】训练模型


---

<a id='question5'></a>  

### __问题 5:__ 

在下方代码单元训练模型。使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。

可选题：你也可以对训练集进行 [数据增强](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)，来优化模型的表现。




```python
from keras.callbacks import ModelCheckpoint  

### TODO: 设置训练模型的epochs的数量

### 不要修改下方代码

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=10, batch_size=20, callbacks=[checkpointer], verbose=1)
```
    Epoch 10/10
    4300/6680 [==================>...........] - ETA: 32s - loss: 4.5941 - acc: 0.0000e+ - ETA: 32s - loss: 4.8016 - acc: 0.0250   - ETA: 33s - loss: 4.6996 - acc: 0.01 - ETA: 33s - loss: 4.6529 - acc: 0.02 - ETA: 33s - loss: 4.6433 - acc: 0.02 - ETA: 33s - loss: 4.6076 - acc: 0.03 - ETA: 33s - loss: 4.5948 - acc: 0.03 - ETA: 33s - loss: 4.6082 - acc: 0.03 - ETA: 33s - loss: 4.6435 - acc: 0.03 - ETA: 32s - loss: 4.6236 - acc: 0.05 - ETA: 32s - loss: 4.6158 - acc: 0.05 - ETA: 32s - loss: 4.6118 - acc: 0.04 - ETA: 32s - loss: 4.6143 - acc: 0.04 - ETA: 32s - loss: 4.6104 - acc: 0.05 - ETA: 32s - loss: 4.6083 - acc: 0.05 - ETA: 32s - loss: 4.6032 - acc: 0.05 - ETA: 32s - loss: 4.6133 - acc: 0.04 - ETA: 32s - loss: 4.6168 - acc: 0.04 - ETA: 31s - loss: 4.6158 - acc: 0.04 - ETA: 31s - loss: 4.6243 - acc: 0.04 - ETA: 31s - loss: 4.6153 - acc: 0.04 - ETA: 31s - loss: 4.6246 - acc: 0.04 - ETA: 31s - loss: 4.6294 - acc: 0.03 - ETA: 31s - loss: 4.6262 - acc: 0.03 - ETA: 31s - loss: 4.6243 - acc: 0.04 - ETA: 31s - loss: 4.6161 - acc: 0.04 - ETA: 31s - loss: 4.6216 - acc: 0.04 - ETA: 30s - loss: 4.6182 - acc: 0.03 - ETA: 30s - loss: 4.6204 - acc: 0.04 - ETA: 30s - loss: 4.6030 - acc: 0.04 - ETA: 30s - loss: 4.6033 - acc: 0.04 - ETA: 30s - loss: 4.6030 - acc: 0.04 - ETA: 30s - loss: 4.6053 - acc: 0.04 - ETA: 30s - loss: 4.6087 - acc: 0.04 - ETA: 30s - loss: 4.6058 - acc: 0.04 - ETA: 30s - loss: 4.6020 - acc: 0.04 - ETA: 29s - loss: 4.6069 - acc: 0.04 - ETA: 29s - loss: 4.6077 - acc: 0.04 - ETA: 29s - loss: 4.6077 - acc: 0.04 - ETA: 29s - loss: 4.6111 - acc: 0.04 - ETA: 29s - loss: 4.6117 - acc: 0.04 - ETA: 29s - loss: 4.6150 - acc: 0.04 - ETA: 29s - loss: 4.6100 - acc: 0.04 - ETA: 29s - loss: 4.6069 - acc: 0.04 - ETA: 29s - loss: 4.6028 - acc: 0.04 - ETA: 29s - loss: 4.6028 - acc: 0.04 - ETA: 28s - loss: 4.5995 - acc: 0.04 - ETA: 28s - loss: 4.5973 - acc: 0.04 - ETA: 28s - loss: 4.6025 - acc: 0.04 - ETA: 28s - loss: 4.6043 - acc: 0.04 - ETA: 28s - loss: 4.6039 - acc: 0.04 - ETA: 28s - loss: 4.6089 - acc: 0.03 - ETA: 28s - loss: 4.6080 - acc: 0.03 - ETA: 28s - loss: 4.6122 - acc: 0.03 - ETA: 28s - loss: 4.6118 - acc: 0.03 - ETA: 28s - loss: 4.6159 - acc: 0.03 - ETA: 27s - loss: 4.6145 - acc: 0.03 - ETA: 27s - loss: 4.6111 - acc: 0.03 - ETA: 27s - loss: 4.6153 - acc: 0.03 - ETA: 27s - loss: 4.6127 - acc: 0.04 - ETA: 27s - loss: 4.6111 - acc: 0.04 - ETA: 27s - loss: 4.6154 - acc: 0.04 - ETA: 27s - loss: 4.6140 - acc: 0.04 - ETA: 27s - loss: 4.6151 - acc: 0.04 - ETA: 27s - loss: 4.6142 - acc: 0.04 - ETA: 27s - loss: 4.6135 - acc: 0.04 - ETA: 26s - loss: 4.6137 - acc: 0.04 - ETA: 26s - loss: 4.6134 - acc: 0.04 - ETA: 26s - loss: 4.6183 - acc: 0.04 - ETA: 26s - loss: 4.6151 - acc: 0.04 - ETA: 26s - loss: 4.6136 - acc: 0.04 - ETA: 26s - loss: 4.6132 - acc: 0.04 - ETA: 26s - loss: 4.6172 - acc: 0.04 - ETA: 26s - loss: 4.6199 - acc: 0.04 - ETA: 26s - loss: 4.6203 - acc: 0.04 - ETA: 26s - loss: 4.6199 - acc: 0.04 - ETA: 25s - loss: 4.6206 - acc: 0.04 - ETA: 25s - loss: 4.6182 - acc: 0.04 - ETA: 25s - loss: 4.6187 - acc: 0.04 - ETA: 25s - loss: 4.6200 - acc: 0.04 - ETA: 25s - loss: 4.6179 - acc: 0.04 - ETA: 25s - loss: 4.6206 - acc: 0.04 - ETA: 25s - loss: 4.6220 - acc: 0.04 - ETA: 25s - loss: 4.6253 - acc: 0.04 - ETA: 25s - loss: 4.6235 - acc: 0.04 - ETA: 25s - loss: 4.6234 - acc: 0.04 - ETA: 24s - loss: 4.6215 - acc: 0.04 - ETA: 24s - loss: 4.6221 - acc: 0.04 - ETA: 24s - loss: 4.6228 - acc: 0.04 - ETA: 24s - loss: 4.6247 - acc: 0.04 - ETA: 24s - loss: 4.6221 - acc: 0.04 - ETA: 24s - loss: 4.6212 - acc: 0.04 - ETA: 24s - loss: 4.6216 - acc: 0.03 - ETA: 24s - loss: 4.6196 - acc: 0.03 - ETA: 24s - loss: 4.6202 - acc: 0.03 - ETA: 24s - loss: 4.6200 - acc: 0.03 - ETA: 23s - loss: 4.6203 - acc: 0.03 - ETA: 23s - loss: 4.6197 - acc: 0.03 - ETA: 23s - loss: 4.6199 - acc: 0.03 - ETA: 23s - loss: 4.6218 - acc: 0.03 - ETA: 23s - loss: 4.6226 - acc: 0.03 - ETA: 23s - loss: 4.6259 - acc: 0.03 - ETA: 23s - loss: 4.6235 - acc: 0.03 - ETA: 23s - loss: 4.6249 - acc: 0.03 - ETA: 23s - loss: 4.6228 - acc: 0.03 - ETA: 23s - loss: 4.6235 - acc: 0.03 - ETA: 22s - loss: 4.6215 - acc: 0.03 - ETA: 22s - loss: 4.6226 - acc: 0.03 - ETA: 22s - loss: 4.6228 - acc: 0.03 - ETA: 22s - loss: 4.6237 - acc: 0.03 - ETA: 22s - loss: 4.6243 - acc: 0.03 - ETA: 22s - loss: 4.6235 - acc: 0.03 - ETA: 22s - loss: 4.6205 - acc: 0.03 - ETA: 22s - loss: 4.6202 - acc: 0.03 - ETA: 22s - loss: 4.6215 - acc: 0.03 - ETA: 22s - loss: 4.6212 - acc: 0.03 - ETA: 21s - loss: 4.6216 - acc: 0.03 - ETA: 21s - loss: 4.6224 - acc: 0.03 - ETA: 21s - loss: 4.6233 - acc: 0.03 - ETA: 21s - loss: 4.6222 - acc: 0.03 - ETA: 21s - loss: 4.6212 - acc: 0.03 - ETA: 21s - loss: 4.6208 - acc: 0.03 - ETA: 21s - loss: 4.6208 - acc: 0.03 - ETA: 21s - loss: 4.6213 - acc: 0.03 - ETA: 21s - loss: 4.6216 - acc: 0.03 - ETA: 20s - loss: 4.6228 - acc: 0.03 - ETA: 20s - loss: 4.6235 - acc: 0.03 - ETA: 20s - loss: 4.6229 - acc: 0.03 - ETA: 20s - loss: 4.6225 - acc: 0.03 - ETA: 20s - loss: 4.6247 - acc: 0.03 - ETA: 20s - loss: 4.6229 - acc: 0.03 - ETA: 20s - loss: 4.6244 - acc: 0.03 - ETA: 20s - loss: 4.6236 - acc: 0.03 - ETA: 20s - loss: 4.6239 - acc: 0.03 - ETA: 20s - loss: 4.6235 - acc: 0.03 - ETA: 19s - loss: 4.6243 - acc: 0.03 - ETA: 19s - loss: 4.6253 - acc: 0.03 - ETA: 19s - loss: 4.6235 - acc: 0.03 - ETA: 19s - loss: 4.6217 - acc: 0.03 - ETA: 19s - loss: 4.6237 - acc: 0.03 - ETA: 19s - loss: 4.6230 - acc: 0.03 - ETA: 19s - loss: 4.6219 - acc: 0.03 - ETA: 19s - loss: 4.6241 - acc: 0.03 - ETA: 19s - loss: 4.6238 - acc: 0.03 - ETA: 19s - loss: 4.6238 - acc: 0.03 - ETA: 18s - loss: 4.6240 - acc: 0.03 - ETA: 18s - loss: 4.6229 - acc: 0.03 - ETA: 18s - loss: 4.6239 - acc: 0.03 - ETA: 18s - loss: 4.6247 - acc: 0.03 - ETA: 18s - loss: 4.6235 - acc: 0.03 - ETA: 18s - loss: 4.6229 - acc: 0.03 - ETA: 18s - loss: 4.6228 - acc: 0.03 - ETA: 18s - loss: 4.6202 - acc: 0.03 - ETA: 18s - loss: 4.6206 - acc: 0.03 - ETA: 18s - loss: 4.6200 - acc: 0.03 - ETA: 17s - loss: 4.6187 - acc: 0.03 - ETA: 17s - loss: 4.6174 - acc: 0.03 - ETA: 17s - loss: 4.6184 - acc: 0.03 - ETA: 17s - loss: 4.6186 - acc: 0.03 - ETA: 17s - loss: 4.6188 - acc: 0.03 - ETA: 17s - loss: 4.6187 - acc: 0.03 - ETA: 17s - loss: 4.6191 - acc: 0.03 - ETA: 17s - loss: 4.6190 - acc: 0.03 - ETA: 17s - loss: 4.6163 - acc: 0.03 - ETA: 17s - loss: 4.6172 - acc: 0.03 - ETA: 16s - loss: 4.6175 - acc: 0.03 - ETA: 16s - loss: 4.6180 - acc: 0.03 - ETA: 16s - loss: 4.6174 - acc: 0.03 - ETA: 16s - loss: 4.6173 - acc: 0.03 - ETA: 16s - loss: 4.6163 - acc: 0.03 - ETA: 16s - loss: 4.6156 - acc: 0.03 - ETA: 16s - loss: 4.6147 - acc: 0.03 - ETA: 16s - loss: 4.6129 - acc: 0.03 - ETA: 16s - loss: 4.6135 - acc: 0.03 - ETA: 16s - loss: 4.6136 - acc: 0.03 - ETA: 15s - loss: 4.6131 - acc: 0.03 - ETA: 15s - loss: 4.6137 - acc: 0.03 - ETA: 15s - loss: 4.6137 - acc: 0.03 - ETA: 15s - loss: 4.6134 - acc: 0.03 - ETA: 15s - loss: 4.6142 - acc: 0.03 - ETA: 15s - loss: 4.6131 - acc: 0.03 - ETA: 15s - loss: 4.6133 - acc: 0.03 - ETA: 15s - loss: 4.6120 - acc: 0.03 - ETA: 15s - loss: 4.6119 - acc: 0.03 - ETA: 15s - loss: 4.6120 - acc: 0.03 - ETA: 14s - loss: 4.6121 - acc: 0.03 - ETA: 14s - loss: 4.6125 - acc: 0.03 - ETA: 14s - loss: 4.6130 - acc: 0.03 - ETA: 14s - loss: 4.6125 - acc: 0.03 - ETA: 14s - loss: 4.6121 - acc: 0.03 - ETA: 14s - loss: 4.6123 - acc: 0.03 - ETA: 14s - loss: 4.6124 - acc: 0.03 - ETA: 14s - loss: 4.6137 - acc: 0.03 - ETA: 14s - loss: 4.6144 - acc: 0.03 - ETA: 14s - loss: 4.6163 - acc: 0.03 - ETA: 13s - loss: 4.6164 - acc: 0.03 - ETA: 13s - loss: 4.6161 - acc: 0.03 - ETA: 13s - loss: 4.6166 - acc: 0.03 - ETA: 13s - loss: 4.6158 - acc: 0.03 - ETA: 13s - loss: 4.6147 - acc: 0.03 - ETA: 13s - loss: 4.6148 - acc: 0.03 - ETA: 13s - loss: 4.6155 - acc: 0.03 - ETA: 13s - loss: 4.6157 - acc: 0.03 - ETA: 13s - loss: 4.6159 - acc: 0.03 - ETA: 13s - loss: 4.6146 - acc: 0.03 - ETA: 12s - loss: 4.6140 - acc: 0.03 - ETA: 12s - loss: 4.6147 - acc: 0.03 - ETA: 12s - loss: 4.6146 - acc: 0.03 - ETA: 12s - loss: 4.6154 - acc: 0.03 - ETA: 12s - loss: 4.6159 - acc: 0.03 - ETA: 12s - loss: 4.6177 - acc: 0.03 - ETA: 12s - loss: 4.6176 - acc: 0.03 - ETA: 12s - loss: 4.6184 - acc: 0.03 - ETA: 12s - loss: 4.6190 - acc: 0.03 - ETA: 11s - loss: 4.6197 - acc: 0.03676660/6680 [============================>.] - ETA: 11s - loss: 4.6189 - acc: 0.03 - ETA: 11s - loss: 4.6198 - acc: 0.03 - ETA: 11s - loss: 4.6201 - acc: 0.03 - ETA: 11s - loss: 4.6198 - acc: 0.03 - ETA: 11s - loss: 4.6198 - acc: 0.03 - ETA: 11s - loss: 4.6198 - acc: 0.03 - ETA: 11s - loss: 4.6187 - acc: 0.03 - ETA: 11s - loss: 4.6182 - acc: 0.03 - ETA: 11s - loss: 4.6180 - acc: 0.03 - ETA: 10s - loss: 4.6180 - acc: 0.03 - ETA: 10s - loss: 4.6189 - acc: 0.03 - ETA: 10s - loss: 4.6186 - acc: 0.03 - ETA: 10s - loss: 4.6182 - acc: 0.03 - ETA: 10s - loss: 4.6178 - acc: 0.03 - ETA: 10s - loss: 4.6183 - acc: 0.03 - ETA: 10s - loss: 4.6177 - acc: 0.03 - ETA: 10s - loss: 4.6172 - acc: 0.03 - ETA: 10s - loss: 4.6158 - acc: 0.03 - ETA: 10s - loss: 4.6161 - acc: 0.03 - ETA: 9s - loss: 4.6157 - acc: 0.0366 - ETA: 9s - loss: 4.6157 - acc: 0.036 - ETA: 9s - loss: 4.6159 - acc: 0.037 - ETA: 9s - loss: 4.6150 - acc: 0.037 - ETA: 9s - loss: 4.6144 - acc: 0.037 - ETA: 9s - loss: 4.6138 - acc: 0.037 - ETA: 9s - loss: 4.6142 - acc: 0.037 - ETA: 9s - loss: 4.6145 - acc: 0.037 - ETA: 9s - loss: 4.6141 - acc: 0.037 - ETA: 9s - loss: 4.6143 - acc: 0.037 - ETA: 8s - loss: 4.6147 - acc: 0.037 - ETA: 8s - loss: 4.6140 - acc: 0.037 - ETA: 8s - loss: 4.6135 - acc: 0.037 - ETA: 8s - loss: 4.6129 - acc: 0.037 - ETA: 8s - loss: 4.6132 - acc: 0.037 - ETA: 8s - loss: 4.6138 - acc: 0.037 - ETA: 8s - loss: 4.6135 - acc: 0.037 - ETA: 8s - loss: 4.6134 - acc: 0.037 - ETA: 8s - loss: 4.6140 - acc: 0.037 - ETA: 8s - loss: 4.6142 - acc: 0.037 - ETA: 7s - loss: 4.6144 - acc: 0.037 - ETA: 7s - loss: 4.6145 - acc: 0.036 - ETA: 7s - loss: 4.6142 - acc: 0.036 - ETA: 7s - loss: 4.6149 - acc: 0.037 - ETA: 7s - loss: 4.6142 - acc: 0.036 - ETA: 7s - loss: 4.6138 - acc: 0.036 - ETA: 7s - loss: 4.6148 - acc: 0.036 - ETA: 7s - loss: 4.6139 - acc: 0.037 - ETA: 7s - loss: 4.6137 - acc: 0.036 - ETA: 7s - loss: 4.6142 - acc: 0.037 - ETA: 6s - loss: 4.6130 - acc: 0.037 - ETA: 6s - loss: 4.6122 - acc: 0.037 - ETA: 6s - loss: 4.6122 - acc: 0.037 - ETA: 6s - loss: 4.6126 - acc: 0.037 - ETA: 6s - loss: 4.6122 - acc: 0.037 - ETA: 6s - loss: 4.6130 - acc: 0.037 - ETA: 6s - loss: 4.6122 - acc: 0.037 - ETA: 6s - loss: 4.6118 - acc: 0.037 - ETA: 6s - loss: 4.6113 - acc: 0.037 - ETA: 6s - loss: 4.6131 - acc: 0.037 - ETA: 5s - loss: 4.6134 - acc: 0.037 - ETA: 5s - loss: 4.6135 - acc: 0.037 - ETA: 5s - loss: 4.6137 - acc: 0.037 - ETA: 5s - loss: 4.6133 - acc: 0.037 - ETA: 5s - loss: 4.6123 - acc: 0.037 - ETA: 5s - loss: 4.6126 - acc: 0.037 - ETA: 5s - loss: 4.6129 - acc: 0.037 - ETA: 5s - loss: 4.6133 - acc: 0.037 - ETA: 5s - loss: 4.6132 - acc: 0.037 - ETA: 5s - loss: 4.6131 - acc: 0.037 - ETA: 4s - loss: 4.6135 - acc: 0.037 - ETA: 4s - loss: 4.6133 - acc: 0.037 - ETA: 4s - loss: 4.6133 - acc: 0.036 - ETA: 4s - loss: 4.6136 - acc: 0.037 - ETA: 4s - loss: 4.6132 - acc: 0.037 - ETA: 4s - loss: 4.6132 - acc: 0.037 - ETA: 4s - loss: 4.6125 - acc: 0.037 - ETA: 4s - loss: 4.6131 - acc: 0.037 - ETA: 4s - loss: 4.6131 - acc: 0.037 - ETA: 4s - loss: 4.6130 - acc: 0.037 - ETA: 3s - loss: 4.6137 - acc: 0.037 - ETA: 3s - loss: 4.6133 - acc: 0.037 - ETA: 3s - loss: 4.6143 - acc: 0.037 - ETA: 3s - loss: 4.6132 - acc: 0.037 - ETA: 3s - loss: 4.6136 - acc: 0.037 - ETA: 3s - loss: 4.6140 - acc: 0.037 - ETA: 3s - loss: 4.6144 - acc: 0.036 - ETA: 3s - loss: 4.6150 - acc: 0.036 - ETA: 3s - loss: 4.6149 - acc: 0.037 - ETA: 3s - loss: 4.6149 - acc: 0.037 - ETA: 2s - loss: 4.6151 - acc: 0.037 - ETA: 2s - loss: 4.6154 - acc: 0.036 - ETA: 2s - loss: 4.6155 - acc: 0.036 - ETA: 2s - loss: 4.6149 - acc: 0.036 - ETA: 2s - loss: 4.6144 - acc: 0.036 - ETA: 2s - loss: 4.6134 - acc: 0.036 - ETA: 2s - loss: 4.6136 - acc: 0.037 - ETA: 2s - loss: 4.6138 - acc: 0.037 - ETA: 2s - loss: 4.6129 - acc: 0.037 - ETA: 2s - loss: 4.6133 - acc: 0.037 - ETA: 1s - loss: 4.6144 - acc: 0.037 - ETA: 1s - loss: 4.6150 - acc: 0.037 - ETA: 1s - loss: 4.6150 - acc: 0.037 - ETA: 1s - loss: 4.6150 - acc: 0.037 - ETA: 1s - loss: 4.6148 - acc: 0.037 - ETA: 1s - loss: 4.6153 - acc: 0.038 - ETA: 1s - loss: 4.6152 - acc: 0.038 - ETA: 1s - loss: 4.6154 - acc: 0.038 - ETA: 1s - loss: 4.6149 - acc: 0.038 - ETA: 1s - loss: 4.6143 - acc: 0.038 - ETA: 0s - loss: 4.6137 - acc: 0.038 - ETA: 0s - loss: 4.6145 - acc: 0.038 - ETA: 0s - loss: 4.6153 - acc: 0.038 - ETA: 0s - loss: 4.6155 - acc: 0.038 - ETA: 0s - loss: 4.6158 - acc: 0.038 - ETA: 0s - loss: 4.6161 - acc: 0.037 - ETA: 0s - loss: 4.6157 - acc: 0.037 - ETA: 0s - loss: 4.6154 - acc: 0.037 - ETA: 0s - loss: 4.6160 - acc: 0.0377Epoch 00009: val_loss improved from 4.70605 to 4.68202, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 35s - loss: 4.6163 - acc: 0.0376 - val_loss: 4.6820 - val_acc: 0.0359





    <keras.callbacks.History at 0x1dc30b06908>




```python
## 加载具有最好验证loss的模型

model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### 测试模型

在狗图像的测试数据集上试用你的模型。确保测试准确率大于1%。


```python
# 获取测试数据集中每一个图像所预测的狗品种的index
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 2.8708%


---
<a id='step4'></a>
## 步骤 4: 使用一个CNN来区分狗的品种


使用 迁移学习（Transfer Learning）的方法，能帮助我们在不损失准确率的情况下大大减少训练时间。在以下步骤中，你可以尝试使用迁移学习来训练你自己的CNN。


### 得到从图像中提取的特征向量（Bottleneck Features）


```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### 模型架构

该模型使用预训练的 VGG-16 模型作为固定的图像特征提取器，其中 VGG-16 最后一层卷积层的输出被直接输入到我们的模型。我们只需要添加一个全局平均池化层以及一个全连接层，其中全连接层使用 softmax 激活函数，对每一个狗的种类都包含一个节点。


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_2 ( (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229.0
    Trainable params: 68,229.0
    Non-trainable params: 0.0
    _________________________________________________________________



```python
## 编译模型

VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```


```python
## 训练模型

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)


```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6560/6680 [============================>.] - ETA: 116s - loss: 14.5417 - acc: 0.0000e+ - ETA: 14s - loss: 15.1006 - acc: 0.0111     - ETA: 8s - loss: 14.7703 - acc: 0.014 - ETA: 6s - loss: 14.8021 - acc: 0.01 - ETA: 5s - loss: 14.5485 - acc: 0.02 - ETA: 4s - loss: 14.4764 - acc: 0.02 - ETA: 4s - loss: 14.3825 - acc: 0.02 - ETA: 3s - loss: 14.2818 - acc: 0.03 - ETA: 3s - loss: 14.1786 - acc: 0.03 - ETA: 3s - loss: 14.1721 - acc: 0.03 - ETA: 2s - loss: 14.0742 - acc: 0.03 - ETA: 2s - loss: 14.0336 - acc: 0.04 - ETA: 2s - loss: 13.9145 - acc: 0.04 - ETA: 2s - loss: 13.8753 - acc: 0.04 - ETA: 2s - loss: 13.8026 - acc: 0.04 - ETA: 2s - loss: 13.7170 - acc: 0.05 - ETA: 1s - loss: 13.6609 - acc: 0.05 - ETA: 1s - loss: 13.6038 - acc: 0.05 - ETA: 1s - loss: 13.5152 - acc: 0.06 - ETA: 1s - loss: 13.4061 - acc: 0.06 - ETA: 1s - loss: 13.3569 - acc: 0.06 - ETA: 1s - loss: 13.2835 - acc: 0.06 - ETA: 1s - loss: 13.1807 - acc: 0.07 - ETA: 1s - loss: 13.1386 - acc: 0.07 - ETA: 1s - loss: 13.1219 - acc: 0.07 - ETA: 1s - loss: 13.0601 - acc: 0.08 - ETA: 1s - loss: 13.0323 - acc: 0.08 - ETA: 0s - loss: 13.0134 - acc: 0.08 - ETA: 0s - loss: 12.9677 - acc: 0.08 - ETA: 0s - loss: 12.8962 - acc: 0.08 - ETA: 0s - loss: 12.8907 - acc: 0.08 - ETA: 0s - loss: 12.8406 - acc: 0.09 - ETA: 0s - loss: 12.7948 - acc: 0.09 - ETA: 0s - loss: 12.7807 - acc: 0.09 - ETA: 0s - loss: 12.7363 - acc: 0.09 - ETA: 0s - loss: 12.7031 - acc: 0.10 - ETA: 0s - loss: 12.6948 - acc: 0.10 - ETA: 0s - loss: 12.6566 - acc: 0.10 - ETA: 0s - loss: 12.6318 - acc: 0.10 - ETA: 0s - loss: 12.5984 - acc: 0.10 - ETA: 0s - loss: 12.6081 - acc: 0.10 - ETA: 0s - loss: 12.5765 - acc: 0.1082Epoch 00000: val_loss improved from inf to 11.21598, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 12.5515 - acc: 0.1100 - val_loss: 11.2160 - val_acc: 0.1916
    Epoch 2/20
    6540/6680 [============================>.] - ETA: 2s - loss: 10.7310 - acc: 0.25 - ETA: 2s - loss: 10.6853 - acc: 0.23 - ETA: 2s - loss: 10.5496 - acc: 0.24 - ETA: 2s - loss: 10.7041 - acc: 0.23 - ETA: 2s - loss: 10.5462 - acc: 0.23 - ETA: 2s - loss: 10.6823 - acc: 0.23 - ETA: 2s - loss: 10.5500 - acc: 0.24 - ETA: 1s - loss: 10.6576 - acc: 0.23 - ETA: 1s - loss: 10.6291 - acc: 0.23 - ETA: 1s - loss: 10.6166 - acc: 0.23 - ETA: 1s - loss: 10.6002 - acc: 0.24 - ETA: 1s - loss: 10.6196 - acc: 0.24 - ETA: 1s - loss: 10.5935 - acc: 0.24 - ETA: 1s - loss: 10.5174 - acc: 0.24 - ETA: 1s - loss: 10.5965 - acc: 0.24 - ETA: 1s - loss: 10.6440 - acc: 0.24 - ETA: 1s - loss: 10.6391 - acc: 0.24 - ETA: 1s - loss: 10.6920 - acc: 0.24 - ETA: 1s - loss: 10.6834 - acc: 0.24 - ETA: 1s - loss: 10.7207 - acc: 0.23 - ETA: 1s - loss: 10.7492 - acc: 0.23 - ETA: 1s - loss: 10.7824 - acc: 0.23 - ETA: 1s - loss: 10.7770 - acc: 0.23 - ETA: 1s - loss: 10.7792 - acc: 0.23 - ETA: 0s - loss: 10.7738 - acc: 0.23 - ETA: 0s - loss: 10.8192 - acc: 0.23 - ETA: 0s - loss: 10.7941 - acc: 0.24 - ETA: 0s - loss: 10.7601 - acc: 0.24 - ETA: 0s - loss: 10.7438 - acc: 0.24 - ETA: 0s - loss: 10.7570 - acc: 0.24 - ETA: 0s - loss: 10.7552 - acc: 0.24 - ETA: 0s - loss: 10.7436 - acc: 0.24 - ETA: 0s - loss: 10.7302 - acc: 0.24 - ETA: 0s - loss: 10.7407 - acc: 0.25 - ETA: 0s - loss: 10.7300 - acc: 0.25 - ETA: 0s - loss: 10.7171 - acc: 0.25 - ETA: 0s - loss: 10.7196 - acc: 0.25 - ETA: 0s - loss: 10.7204 - acc: 0.25 - ETA: 0s - loss: 10.7209 - acc: 0.25 - ETA: 0s - loss: 10.7071 - acc: 0.25 - ETA: 0s - loss: 10.7207 - acc: 0.25 - ETA: 0s - loss: 10.7138 - acc: 0.2540Epoch 00001: val_loss improved from 11.21598 to 10.71026, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 10.7049 - acc: 0.2549 - val_loss: 10.7103 - val_acc: 0.2551
    Epoch 3/20
    6640/6680 [============================>.] - ETA: 2s - loss: 8.3306 - acc: 0.400 - ETA: 2s - loss: 9.5591 - acc: 0.355 - ETA: 2s - loss: 9.9084 - acc: 0.344 - ETA: 2s - loss: 9.9087 - acc: 0.342 - ETA: 2s - loss: 9.7852 - acc: 0.340 - ETA: 1s - loss: 9.9821 - acc: 0.329 - ETA: 1s - loss: 10.1898 - acc: 0.32 - ETA: 1s - loss: 10.2557 - acc: 0.32 - ETA: 1s - loss: 10.3741 - acc: 0.31 - ETA: 1s - loss: 10.2817 - acc: 0.31 - ETA: 1s - loss: 10.1927 - acc: 0.32 - ETA: 1s - loss: 10.2258 - acc: 0.31 - ETA: 1s - loss: 10.2107 - acc: 0.31 - ETA: 1s - loss: 10.2017 - acc: 0.31 - ETA: 1s - loss: 10.2660 - acc: 0.31 - ETA: 1s - loss: 10.2499 - acc: 0.31 - ETA: 1s - loss: 10.2902 - acc: 0.31 - ETA: 1s - loss: 10.2158 - acc: 0.31 - ETA: 1s - loss: 10.2441 - acc: 0.31 - ETA: 1s - loss: 10.2311 - acc: 0.31 - ETA: 1s - loss: 10.1910 - acc: 0.32 - ETA: 1s - loss: 10.2102 - acc: 0.32 - ETA: 1s - loss: 10.2734 - acc: 0.31 - ETA: 1s - loss: 10.2956 - acc: 0.31 - ETA: 0s - loss: 10.3169 - acc: 0.31 - ETA: 0s - loss: 10.2765 - acc: 0.31 - ETA: 0s - loss: 10.2649 - acc: 0.31 - ETA: 0s - loss: 10.2587 - acc: 0.31 - ETA: 0s - loss: 10.2953 - acc: 0.31 - ETA: 0s - loss: 10.2908 - acc: 0.31 - ETA: 0s - loss: 10.3031 - acc: 0.31 - ETA: 0s - loss: 10.3427 - acc: 0.31 - ETA: 0s - loss: 10.3412 - acc: 0.31 - ETA: 0s - loss: 10.3345 - acc: 0.31 - ETA: 0s - loss: 10.3498 - acc: 0.31 - ETA: 0s - loss: 10.3587 - acc: 0.31 - ETA: 0s - loss: 10.3654 - acc: 0.31 - ETA: 0s - loss: 10.3756 - acc: 0.31 - ETA: 0s - loss: 10.3815 - acc: 0.31 - ETA: 0s - loss: 10.3772 - acc: 0.31 - ETA: 0s - loss: 10.3855 - acc: 0.31 - ETA: 0s - loss: 10.3961 - acc: 0.30 - ETA: 0s - loss: 10.3994 - acc: 0.3090Epoch 00002: val_loss improved from 10.71026 to 10.52035, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 10.3930 - acc: 0.3094 - val_loss: 10.5204 - val_acc: 0.2814
    Epoch 4/20
    6620/6680 [============================>.] - ETA: 2s - loss: 12.1513 - acc: 0.25 - ETA: 2s - loss: 10.5722 - acc: 0.30 - ETA: 2s - loss: 10.3198 - acc: 0.32 - ETA: 2s - loss: 10.2740 - acc: 0.32 - ETA: 2s - loss: 10.5196 - acc: 0.31 - ETA: 2s - loss: 10.6302 - acc: 0.30 - ETA: 1s - loss: 10.5214 - acc: 0.31 - ETA: 1s - loss: 10.3773 - acc: 0.32 - ETA: 1s - loss: 10.4181 - acc: 0.31 - ETA: 1s - loss: 10.4782 - acc: 0.31 - ETA: 1s - loss: 10.3883 - acc: 0.32 - ETA: 1s - loss: 10.4552 - acc: 0.31 - ETA: 1s - loss: 10.4043 - acc: 0.31 - ETA: 1s - loss: 10.3756 - acc: 0.32 - ETA: 1s - loss: 10.3837 - acc: 0.31 - ETA: 1s - loss: 10.2869 - acc: 0.32 - ETA: 1s - loss: 10.2071 - acc: 0.32 - ETA: 1s - loss: 10.1680 - acc: 0.32 - ETA: 1s - loss: 10.1948 - acc: 0.32 - ETA: 1s - loss: 10.1984 - acc: 0.32 - ETA: 1s - loss: 10.2322 - acc: 0.32 - ETA: 1s - loss: 10.2297 - acc: 0.32 - ETA: 1s - loss: 10.2108 - acc: 0.32 - ETA: 1s - loss: 10.2262 - acc: 0.32 - ETA: 0s - loss: 10.2504 - acc: 0.32 - ETA: 0s - loss: 10.2396 - acc: 0.32 - ETA: 0s - loss: 10.1915 - acc: 0.32 - ETA: 0s - loss: 10.1368 - acc: 0.32 - ETA: 0s - loss: 10.1178 - acc: 0.33 - ETA: 0s - loss: 10.1015 - acc: 0.33 - ETA: 0s - loss: 10.1230 - acc: 0.32 - ETA: 0s - loss: 10.1191 - acc: 0.32 - ETA: 0s - loss: 10.1214 - acc: 0.33 - ETA: 0s - loss: 10.1248 - acc: 0.32 - ETA: 0s - loss: 10.1419 - acc: 0.32 - ETA: 0s - loss: 10.1500 - acc: 0.32 - ETA: 0s - loss: 10.1506 - acc: 0.32 - ETA: 0s - loss: 10.1703 - acc: 0.32 - ETA: 0s - loss: 10.1612 - acc: 0.32 - ETA: 0s - loss: 10.1603 - acc: 0.32 - ETA: 0s - loss: 10.1654 - acc: 0.32 - ETA: 0s - loss: 10.1520 - acc: 0.32 - ETA: 0s - loss: 10.1396 - acc: 0.3299Epoch 00003: val_loss improved from 10.52035 to 10.28915, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 10.1364 - acc: 0.3302 - val_loss: 10.2892 - val_acc: 0.2958
    Epoch 5/20
    6520/6680 [============================>.] - ETA: 2s - loss: 9.0330 - acc: 0.400 - ETA: 2s - loss: 9.8361 - acc: 0.368 - ETA: 2s - loss: 10.1016 - acc: 0.35 - ETA: 2s - loss: 10.2052 - acc: 0.35 - ETA: 2s - loss: 10.0266 - acc: 0.35 - ETA: 2s - loss: 10.1128 - acc: 0.35 - ETA: 1s - loss: 10.0588 - acc: 0.35 - ETA: 1s - loss: 10.0062 - acc: 0.35 - ETA: 1s - loss: 10.0939 - acc: 0.34 - ETA: 1s - loss: 10.0726 - acc: 0.35 - ETA: 1s - loss: 10.0495 - acc: 0.35 - ETA: 1s - loss: 10.0237 - acc: 0.35 - ETA: 1s - loss: 10.0468 - acc: 0.35 - ETA: 1s - loss: 10.0030 - acc: 0.35 - ETA: 1s - loss: 10.0347 - acc: 0.35 - ETA: 1s - loss: 10.0181 - acc: 0.35 - ETA: 1s - loss: 10.0012 - acc: 0.35 - ETA: 1s - loss: 9.9640 - acc: 0.3596 - ETA: 1s - loss: 9.9974 - acc: 0.358 - ETA: 1s - loss: 9.9807 - acc: 0.358 - ETA: 1s - loss: 9.9265 - acc: 0.362 - ETA: 1s - loss: 9.9095 - acc: 0.361 - ETA: 1s - loss: 9.9650 - acc: 0.358 - ETA: 1s - loss: 9.9806 - acc: 0.357 - ETA: 0s - loss: 9.9740 - acc: 0.357 - ETA: 0s - loss: 9.9268 - acc: 0.360 - ETA: 0s - loss: 9.9340 - acc: 0.359 - ETA: 0s - loss: 9.9712 - acc: 0.357 - ETA: 0s - loss: 9.9885 - acc: 0.355 - ETA: 0s - loss: 9.9714 - acc: 0.356 - ETA: 0s - loss: 9.9559 - acc: 0.356 - ETA: 0s - loss: 9.9626 - acc: 0.355 - ETA: 0s - loss: 9.9425 - acc: 0.357 - ETA: 0s - loss: 9.9349 - acc: 0.357 - ETA: 0s - loss: 9.9427 - acc: 0.355 - ETA: 0s - loss: 9.9346 - acc: 0.356 - ETA: 0s - loss: 9.9228 - acc: 0.356 - ETA: 0s - loss: 9.9058 - acc: 0.357 - ETA: 0s - loss: 9.9059 - acc: 0.356 - ETA: 0s - loss: 9.9292 - acc: 0.354 - ETA: 0s - loss: 9.9275 - acc: 0.354 - ETA: 0s - loss: 9.9343 - acc: 0.3541Epoch 00004: val_loss improved from 10.28915 to 10.07849, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 9.9519 - acc: 0.3533 - val_loss: 10.0785 - val_acc: 0.3078
    Epoch 6/20
    6520/6680 [============================>.] - ETA: 2s - loss: 8.3866 - acc: 0.450 - ETA: 2s - loss: 10.0937 - acc: 0.34 - ETA: 2s - loss: 9.7600 - acc: 0.3750 - ETA: 2s - loss: 9.6629 - acc: 0.383 - ETA: 2s - loss: 9.8101 - acc: 0.372 - ETA: 2s - loss: 9.9438 - acc: 0.362 - ETA: 2s - loss: 9.8382 - acc: 0.367 - ETA: 1s - loss: 9.7850 - acc: 0.369 - ETA: 1s - loss: 9.6892 - acc: 0.373 - ETA: 1s - loss: 9.6896 - acc: 0.373 - ETA: 1s - loss: 9.6129 - acc: 0.375 - ETA: 1s - loss: 9.6462 - acc: 0.374 - ETA: 1s - loss: 9.6811 - acc: 0.370 - ETA: 1s - loss: 9.7033 - acc: 0.369 - ETA: 1s - loss: 9.6609 - acc: 0.373 - ETA: 1s - loss: 9.6695 - acc: 0.374 - ETA: 1s - loss: 9.6721 - acc: 0.372 - ETA: 1s - loss: 9.6931 - acc: 0.369 - ETA: 1s - loss: 9.6691 - acc: 0.371 - ETA: 1s - loss: 9.6663 - acc: 0.370 - ETA: 1s - loss: 9.6511 - acc: 0.370 - ETA: 1s - loss: 9.6628 - acc: 0.369 - ETA: 1s - loss: 9.7172 - acc: 0.364 - ETA: 1s - loss: 9.6802 - acc: 0.366 - ETA: 0s - loss: 9.6693 - acc: 0.366 - ETA: 0s - loss: 9.6647 - acc: 0.365 - ETA: 0s - loss: 9.6719 - acc: 0.365 - ETA: 0s - loss: 9.6586 - acc: 0.366 - ETA: 0s - loss: 9.6321 - acc: 0.367 - ETA: 0s - loss: 9.6236 - acc: 0.367 - ETA: 0s - loss: 9.6241 - acc: 0.367 - ETA: 0s - loss: 9.6307 - acc: 0.366 - ETA: 0s - loss: 9.6084 - acc: 0.367 - ETA: 0s - loss: 9.6025 - acc: 0.368 - ETA: 0s - loss: 9.6269 - acc: 0.366 - ETA: 0s - loss: 9.6438 - acc: 0.365 - ETA: 0s - loss: 9.6422 - acc: 0.365 - ETA: 0s - loss: 9.6208 - acc: 0.366 - ETA: 0s - loss: 9.6209 - acc: 0.366 - ETA: 0s - loss: 9.6063 - acc: 0.367 - ETA: 0s - loss: 9.5794 - acc: 0.368 - ETA: 0s - loss: 9.5856 - acc: 0.3676Epoch 00005: val_loss improved from 10.07849 to 9.73418, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 9.5836 - acc: 0.3680 - val_loss: 9.7342 - val_acc: 0.3269
    Epoch 7/20
    6620/6680 [============================>.] - ETA: 1s - loss: 8.2345 - acc: 0.400 - ETA: 2s - loss: 9.0365 - acc: 0.400 - ETA: 2s - loss: 9.1624 - acc: 0.391 - ETA: 2s - loss: 9.3301 - acc: 0.390 - ETA: 2s - loss: 9.3720 - acc: 0.389 - ETA: 1s - loss: 9.2283 - acc: 0.396 - ETA: 1s - loss: 9.3445 - acc: 0.389 - ETA: 1s - loss: 9.3488 - acc: 0.392 - ETA: 1s - loss: 9.3508 - acc: 0.393 - ETA: 1s - loss: 9.4630 - acc: 0.387 - ETA: 1s - loss: 9.5512 - acc: 0.381 - ETA: 1s - loss: 9.5051 - acc: 0.382 - ETA: 1s - loss: 9.5105 - acc: 0.380 - ETA: 1s - loss: 9.5097 - acc: 0.380 - ETA: 1s - loss: 9.5059 - acc: 0.380 - ETA: 1s - loss: 9.4551 - acc: 0.382 - ETA: 1s - loss: 9.3602 - acc: 0.388 - ETA: 1s - loss: 9.3789 - acc: 0.387 - ETA: 1s - loss: 9.3692 - acc: 0.388 - ETA: 1s - loss: 9.3420 - acc: 0.389 - ETA: 1s - loss: 9.3177 - acc: 0.390 - ETA: 1s - loss: 9.3196 - acc: 0.390 - ETA: 1s - loss: 9.3195 - acc: 0.390 - ETA: 1s - loss: 9.2833 - acc: 0.392 - ETA: 0s - loss: 9.3043 - acc: 0.391 - ETA: 0s - loss: 9.3063 - acc: 0.391 - ETA: 0s - loss: 9.3225 - acc: 0.389 - ETA: 0s - loss: 9.2842 - acc: 0.391 - ETA: 0s - loss: 9.2840 - acc: 0.391 - ETA: 0s - loss: 9.2901 - acc: 0.391 - ETA: 0s - loss: 9.2849 - acc: 0.392 - ETA: 0s - loss: 9.2796 - acc: 0.392 - ETA: 0s - loss: 9.2443 - acc: 0.394 - ETA: 0s - loss: 9.2491 - acc: 0.394 - ETA: 0s - loss: 9.2629 - acc: 0.393 - ETA: 0s - loss: 9.2713 - acc: 0.392 - ETA: 0s - loss: 9.2693 - acc: 0.392 - ETA: 0s - loss: 9.2771 - acc: 0.391 - ETA: 0s - loss: 9.2647 - acc: 0.392 - ETA: 0s - loss: 9.2684 - acc: 0.392 - ETA: 0s - loss: 9.2549 - acc: 0.393 - ETA: 0s - loss: 9.2368 - acc: 0.3944Epoch 00006: val_loss improved from 9.73418 to 9.50479, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 9.2400 - acc: 0.3945 - val_loss: 9.5048 - val_acc: 0.3329
    Epoch 8/20
    6600/6680 [============================>.] - ETA: 2s - loss: 7.2668 - acc: 0.550 - ETA: 2s - loss: 9.6827 - acc: 0.366 - ETA: 2s - loss: 9.2905 - acc: 0.394 - ETA: 2s - loss: 9.4611 - acc: 0.380 - ETA: 2s - loss: 9.3606 - acc: 0.383 - ETA: 1s - loss: 9.3669 - acc: 0.385 - ETA: 1s - loss: 9.4514 - acc: 0.380 - ETA: 1s - loss: 9.4125 - acc: 0.385 - ETA: 1s - loss: 9.3156 - acc: 0.392 - ETA: 1s - loss: 9.3449 - acc: 0.392 - ETA: 1s - loss: 9.1051 - acc: 0.406 - ETA: 1s - loss: 9.1387 - acc: 0.402 - ETA: 1s - loss: 9.1393 - acc: 0.404 - ETA: 1s - loss: 9.1300 - acc: 0.404 - ETA: 1s - loss: 9.1165 - acc: 0.405 - ETA: 1s - loss: 9.1167 - acc: 0.406 - ETA: 1s - loss: 9.1112 - acc: 0.407 - ETA: 1s - loss: 9.0676 - acc: 0.410 - ETA: 1s - loss: 9.0732 - acc: 0.411 - ETA: 1s - loss: 9.0485 - acc: 0.413 - ETA: 1s - loss: 9.0495 - acc: 0.413 - ETA: 1s - loss: 8.9979 - acc: 0.416 - ETA: 1s - loss: 8.9594 - acc: 0.418 - ETA: 0s - loss: 8.9665 - acc: 0.418 - ETA: 0s - loss: 9.0083 - acc: 0.415 - ETA: 0s - loss: 9.0404 - acc: 0.414 - ETA: 0s - loss: 9.0123 - acc: 0.416 - ETA: 0s - loss: 8.9999 - acc: 0.416 - ETA: 0s - loss: 9.0125 - acc: 0.416 - ETA: 0s - loss: 9.0281 - acc: 0.415 - ETA: 0s - loss: 9.0316 - acc: 0.415 - ETA: 0s - loss: 9.0339 - acc: 0.415 - ETA: 0s - loss: 9.0262 - acc: 0.415 - ETA: 0s - loss: 9.0089 - acc: 0.415 - ETA: 0s - loss: 9.0356 - acc: 0.414 - ETA: 0s - loss: 9.0632 - acc: 0.412 - ETA: 0s - loss: 9.0481 - acc: 0.414 - ETA: 0s - loss: 9.0570 - acc: 0.413 - ETA: 0s - loss: 9.0762 - acc: 0.412 - ETA: 0s - loss: 9.0722 - acc: 0.412 - ETA: 0s - loss: 9.0678 - acc: 0.413 - ETA: 0s - loss: 9.0668 - acc: 0.4138Epoch 00007: val_loss improved from 9.50479 to 9.44717, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 9.0582 - acc: 0.4142 - val_loss: 9.4472 - val_acc: 0.3485
    Epoch 9/20
    6580/6680 [============================>.] - ETA: 2s - loss: 12.0887 - acc: 0.25 - ETA: 2s - loss: 9.2708 - acc: 0.4167 - ETA: 2s - loss: 9.6955 - acc: 0.391 - ETA: 2s - loss: 9.1532 - acc: 0.424 - ETA: 1s - loss: 9.1979 - acc: 0.419 - ETA: 1s - loss: 9.2743 - acc: 0.414 - ETA: 1s - loss: 9.2209 - acc: 0.417 - ETA: 1s - loss: 9.2222 - acc: 0.415 - ETA: 1s - loss: 9.1374 - acc: 0.420 - ETA: 1s - loss: 9.1648 - acc: 0.418 - ETA: 1s - loss: 9.1626 - acc: 0.417 - ETA: 1s - loss: 9.1549 - acc: 0.418 - ETA: 1s - loss: 9.1334 - acc: 0.418 - ETA: 1s - loss: 9.0545 - acc: 0.424 - ETA: 1s - loss: 9.0456 - acc: 0.425 - ETA: 1s - loss: 8.9808 - acc: 0.430 - ETA: 1s - loss: 8.9437 - acc: 0.432 - ETA: 1s - loss: 8.9089 - acc: 0.434 - ETA: 1s - loss: 8.8552 - acc: 0.436 - ETA: 1s - loss: 8.8521 - acc: 0.437 - ETA: 1s - loss: 8.8315 - acc: 0.438 - ETA: 1s - loss: 8.8280 - acc: 0.438 - ETA: 1s - loss: 8.8382 - acc: 0.437 - ETA: 0s - loss: 8.8622 - acc: 0.435 - ETA: 0s - loss: 8.8581 - acc: 0.435 - ETA: 0s - loss: 8.8589 - acc: 0.435 - ETA: 0s - loss: 8.8524 - acc: 0.435 - ETA: 0s - loss: 8.8760 - acc: 0.434 - ETA: 0s - loss: 8.8869 - acc: 0.433 - ETA: 0s - loss: 8.9146 - acc: 0.431 - ETA: 0s - loss: 8.9260 - acc: 0.431 - ETA: 0s - loss: 8.9289 - acc: 0.430 - ETA: 0s - loss: 8.9365 - acc: 0.430 - ETA: 0s - loss: 8.9416 - acc: 0.430 - ETA: 0s - loss: 8.9583 - acc: 0.429 - ETA: 0s - loss: 8.9771 - acc: 0.428 - ETA: 0s - loss: 8.9670 - acc: 0.428 - ETA: 0s - loss: 8.9780 - acc: 0.427 - ETA: 0s - loss: 8.9853 - acc: 0.427 - ETA: 0s - loss: 9.0050 - acc: 0.426 - ETA: 0s - loss: 9.0059 - acc: 0.426 - ETA: 0s - loss: 9.0242 - acc: 0.4249Epoch 00008: val_loss improved from 9.44717 to 9.40818, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 9.0131 - acc: 0.4257 - val_loss: 9.4082 - val_acc: 0.3617
    Epoch 10/20
    6660/6680 [============================>.] - ETA: 2s - loss: 10.5413 - acc: 0.30 - ETA: 2s - loss: 8.7304 - acc: 0.4500 - ETA: 2s - loss: 8.8214 - acc: 0.441 - ETA: 2s - loss: 8.6870 - acc: 0.450 - ETA: 2s - loss: 8.7576 - acc: 0.447 - ETA: 1s - loss: 8.7869 - acc: 0.446 - ETA: 1s - loss: 8.8185 - acc: 0.444 - ETA: 1s - loss: 8.6890 - acc: 0.452 - ETA: 1s - loss: 8.7227 - acc: 0.449 - ETA: 1s - loss: 8.7540 - acc: 0.446 - ETA: 1s - loss: 8.8289 - acc: 0.441 - ETA: 1s - loss: 8.9009 - acc: 0.437 - ETA: 1s - loss: 9.0692 - acc: 0.427 - ETA: 1s - loss: 9.1264 - acc: 0.423 - ETA: 1s - loss: 9.1518 - acc: 0.420 - ETA: 1s - loss: 9.2022 - acc: 0.417 - ETA: 1s - loss: 9.0900 - acc: 0.424 - ETA: 1s - loss: 9.0053 - acc: 0.430 - ETA: 1s - loss: 8.9991 - acc: 0.431 - ETA: 1s - loss: 9.0728 - acc: 0.426 - ETA: 1s - loss: 9.0748 - acc: 0.426 - ETA: 1s - loss: 9.0467 - acc: 0.428 - ETA: 1s - loss: 9.0320 - acc: 0.428 - ETA: 1s - loss: 9.0264 - acc: 0.429 - ETA: 0s - loss: 9.0417 - acc: 0.428 - ETA: 0s - loss: 9.0737 - acc: 0.426 - ETA: 0s - loss: 9.0663 - acc: 0.426 - ETA: 0s - loss: 9.0796 - acc: 0.425 - ETA: 0s - loss: 9.0557 - acc: 0.426 - ETA: 0s - loss: 9.0709 - acc: 0.425 - ETA: 0s - loss: 9.0241 - acc: 0.428 - ETA: 0s - loss: 9.0213 - acc: 0.428 - ETA: 0s - loss: 8.9846 - acc: 0.431 - ETA: 0s - loss: 8.9529 - acc: 0.433 - ETA: 0s - loss: 8.9772 - acc: 0.431 - ETA: 0s - loss: 8.9695 - acc: 0.432 - ETA: 0s - loss: 8.9937 - acc: 0.430 - ETA: 0s - loss: 8.9741 - acc: 0.431 - ETA: 0s - loss: 8.9773 - acc: 0.431 - ETA: 0s - loss: 9.0108 - acc: 0.429 - ETA: 0s - loss: 8.9927 - acc: 0.430 - ETA: 0s - loss: 8.9857 - acc: 0.430 - ETA: 0s - loss: 8.9781 - acc: 0.4315Epoch 00009: val_loss improved from 9.40818 to 9.35232, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.9875 - acc: 0.4310 - val_loss: 9.3523 - val_acc: 0.3605
    Epoch 11/20
    6660/6680 [============================>.] - ETA: 2s - loss: 12.0912 - acc: 0.25 - ETA: 2s - loss: 8.6220 - acc: 0.4611 - ETA: 2s - loss: 8.5541 - acc: 0.464 - ETA: 2s - loss: 8.8635 - acc: 0.445 - ETA: 2s - loss: 8.8674 - acc: 0.445 - ETA: 2s - loss: 8.9075 - acc: 0.443 - ETA: 1s - loss: 9.1050 - acc: 0.431 - ETA: 1s - loss: 9.0865 - acc: 0.432 - ETA: 1s - loss: 8.9663 - acc: 0.438 - ETA: 1s - loss: 8.9812 - acc: 0.437 - ETA: 1s - loss: 9.0263 - acc: 0.434 - ETA: 1s - loss: 8.9850 - acc: 0.436 - ETA: 1s - loss: 9.0310 - acc: 0.433 - ETA: 1s - loss: 9.0112 - acc: 0.435 - ETA: 1s - loss: 9.0423 - acc: 0.432 - ETA: 1s - loss: 9.0450 - acc: 0.432 - ETA: 1s - loss: 8.9326 - acc: 0.438 - ETA: 1s - loss: 8.9075 - acc: 0.440 - ETA: 1s - loss: 8.8888 - acc: 0.441 - ETA: 1s - loss: 8.8909 - acc: 0.441 - ETA: 1s - loss: 8.9219 - acc: 0.438 - ETA: 1s - loss: 8.9034 - acc: 0.439 - ETA: 1s - loss: 8.9107 - acc: 0.438 - ETA: 1s - loss: 8.9109 - acc: 0.438 - ETA: 0s - loss: 8.9150 - acc: 0.438 - ETA: 0s - loss: 8.9164 - acc: 0.438 - ETA: 0s - loss: 8.9065 - acc: 0.438 - ETA: 0s - loss: 8.9302 - acc: 0.436 - ETA: 0s - loss: 8.9223 - acc: 0.436 - ETA: 0s - loss: 8.9557 - acc: 0.433 - ETA: 0s - loss: 8.9564 - acc: 0.433 - ETA: 0s - loss: 8.9903 - acc: 0.431 - ETA: 0s - loss: 8.9784 - acc: 0.432 - ETA: 0s - loss: 8.9469 - acc: 0.434 - ETA: 0s - loss: 8.9347 - acc: 0.434 - ETA: 0s - loss: 8.9220 - acc: 0.434 - ETA: 0s - loss: 8.9488 - acc: 0.433 - ETA: 0s - loss: 8.9715 - acc: 0.432 - ETA: 0s - loss: 8.9566 - acc: 0.433 - ETA: 0s - loss: 8.9445 - acc: 0.434 - ETA: 0s - loss: 8.9375 - acc: 0.434 - ETA: 0s - loss: 8.9377 - acc: 0.433 - ETA: 0s - loss: 8.9253 - acc: 0.4342Epoch 00010: val_loss improved from 9.35232 to 9.31277, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.9198 - acc: 0.4344 - val_loss: 9.3128 - val_acc: 0.3665
    Epoch 12/20
    6660/6680 [============================>.] - ETA: 2s - loss: 8.8699 - acc: 0.450 - ETA: 2s - loss: 8.8791 - acc: 0.443 - ETA: 2s - loss: 9.2277 - acc: 0.421 - ETA: 2s - loss: 9.3085 - acc: 0.414 - ETA: 2s - loss: 9.2248 - acc: 0.420 - ETA: 2s - loss: 9.2272 - acc: 0.420 - ETA: 1s - loss: 9.0505 - acc: 0.427 - ETA: 1s - loss: 8.9463 - acc: 0.433 - ETA: 1s - loss: 8.8876 - acc: 0.435 - ETA: 1s - loss: 8.9155 - acc: 0.433 - ETA: 1s - loss: 8.9608 - acc: 0.430 - ETA: 1s - loss: 8.9833 - acc: 0.427 - ETA: 1s - loss: 8.9760 - acc: 0.429 - ETA: 1s - loss: 8.9209 - acc: 0.432 - ETA: 1s - loss: 8.8675 - acc: 0.434 - ETA: 1s - loss: 8.8957 - acc: 0.432 - ETA: 1s - loss: 8.8490 - acc: 0.436 - ETA: 1s - loss: 8.8641 - acc: 0.435 - ETA: 1s - loss: 8.8570 - acc: 0.436 - ETA: 1s - loss: 8.8490 - acc: 0.436 - ETA: 1s - loss: 8.8577 - acc: 0.436 - ETA: 1s - loss: 8.8286 - acc: 0.437 - ETA: 1s - loss: 8.8207 - acc: 0.437 - ETA: 1s - loss: 8.8192 - acc: 0.437 - ETA: 0s - loss: 8.7596 - acc: 0.440 - ETA: 0s - loss: 8.7686 - acc: 0.440 - ETA: 0s - loss: 8.7561 - acc: 0.441 - ETA: 0s - loss: 8.7862 - acc: 0.439 - ETA: 0s - loss: 8.8182 - acc: 0.437 - ETA: 0s - loss: 8.8499 - acc: 0.435 - ETA: 0s - loss: 8.8599 - acc: 0.434 - ETA: 0s - loss: 8.8521 - acc: 0.434 - ETA: 0s - loss: 8.8505 - acc: 0.434 - ETA: 0s - loss: 8.8441 - acc: 0.434 - ETA: 0s - loss: 8.8508 - acc: 0.433 - ETA: 0s - loss: 8.8296 - acc: 0.434 - ETA: 0s - loss: 8.7963 - acc: 0.436 - ETA: 0s - loss: 8.7696 - acc: 0.438 - ETA: 0s - loss: 8.7774 - acc: 0.437 - ETA: 0s - loss: 8.7486 - acc: 0.439 - ETA: 0s - loss: 8.7476 - acc: 0.439 - ETA: 0s - loss: 8.7363 - acc: 0.440 - ETA: 0s - loss: 8.7425 - acc: 0.4402Epoch 00011: val_loss improved from 9.31277 to 9.19319, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.7380 - acc: 0.4406 - val_loss: 9.1932 - val_acc: 0.3677
    Epoch 13/20
    6660/6680 [============================>.] - ETA: 2s - loss: 9.6709 - acc: 0.400 - ETA: 2s - loss: 9.0701 - acc: 0.433 - ETA: 2s - loss: 8.7094 - acc: 0.450 - ETA: 2s - loss: 8.9159 - acc: 0.434 - ETA: 2s - loss: 8.7540 - acc: 0.443 - ETA: 2s - loss: 8.7802 - acc: 0.443 - ETA: 2s - loss: 8.7873 - acc: 0.440 - ETA: 1s - loss: 8.8040 - acc: 0.440 - ETA: 1s - loss: 8.8777 - acc: 0.436 - ETA: 1s - loss: 8.7509 - acc: 0.445 - ETA: 1s - loss: 8.7814 - acc: 0.441 - ETA: 1s - loss: 8.8008 - acc: 0.440 - ETA: 1s - loss: 8.8282 - acc: 0.438 - ETA: 1s - loss: 8.8563 - acc: 0.436 - ETA: 1s - loss: 8.7944 - acc: 0.438 - ETA: 1s - loss: 8.7574 - acc: 0.441 - ETA: 1s - loss: 8.7233 - acc: 0.443 - ETA: 1s - loss: 8.6528 - acc: 0.447 - ETA: 1s - loss: 8.7110 - acc: 0.444 - ETA: 1s - loss: 8.6707 - acc: 0.447 - ETA: 1s - loss: 8.6790 - acc: 0.446 - ETA: 1s - loss: 8.6875 - acc: 0.445 - ETA: 1s - loss: 8.7143 - acc: 0.443 - ETA: 1s - loss: 8.7591 - acc: 0.440 - ETA: 0s - loss: 8.7213 - acc: 0.442 - ETA: 0s - loss: 8.7066 - acc: 0.443 - ETA: 0s - loss: 8.7017 - acc: 0.443 - ETA: 0s - loss: 8.6843 - acc: 0.445 - ETA: 0s - loss: 8.6451 - acc: 0.447 - ETA: 0s - loss: 8.6776 - acc: 0.446 - ETA: 0s - loss: 8.6780 - acc: 0.446 - ETA: 0s - loss: 8.6794 - acc: 0.446 - ETA: 0s - loss: 8.6549 - acc: 0.447 - ETA: 0s - loss: 8.6251 - acc: 0.449 - ETA: 0s - loss: 8.6312 - acc: 0.449 - ETA: 0s - loss: 8.6133 - acc: 0.449 - ETA: 0s - loss: 8.6085 - acc: 0.449 - ETA: 0s - loss: 8.6229 - acc: 0.448 - ETA: 0s - loss: 8.5951 - acc: 0.450 - ETA: 0s - loss: 8.6089 - acc: 0.449 - ETA: 0s - loss: 8.5972 - acc: 0.450 - ETA: 0s - loss: 8.6044 - acc: 0.450 - ETA: 0s - loss: 8.6030 - acc: 0.4503Epoch 00012: val_loss improved from 9.19319 to 9.09968, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.6039 - acc: 0.4503 - val_loss: 9.0997 - val_acc: 0.3737
    Epoch 14/20
    6600/6680 [============================>.] - ETA: 2s - loss: 8.0833 - acc: 0.500 - ETA: 2s - loss: 7.9762 - acc: 0.475 - ETA: 2s - loss: 8.0988 - acc: 0.475 - ETA: 2s - loss: 8.2304 - acc: 0.468 - ETA: 2s - loss: 8.4241 - acc: 0.460 - ETA: 2s - loss: 8.3955 - acc: 0.463 - ETA: 1s - loss: 8.3710 - acc: 0.466 - ETA: 1s - loss: 8.4363 - acc: 0.461 - ETA: 1s - loss: 8.3602 - acc: 0.467 - ETA: 1s - loss: 8.4272 - acc: 0.462 - ETA: 1s - loss: 8.4889 - acc: 0.459 - ETA: 1s - loss: 8.4659 - acc: 0.460 - ETA: 1s - loss: 8.4489 - acc: 0.460 - ETA: 1s - loss: 8.4651 - acc: 0.459 - ETA: 1s - loss: 8.4726 - acc: 0.459 - ETA: 1s - loss: 8.4789 - acc: 0.460 - ETA: 1s - loss: 8.4737 - acc: 0.460 - ETA: 1s - loss: 8.4209 - acc: 0.465 - ETA: 1s - loss: 8.4076 - acc: 0.466 - ETA: 1s - loss: 8.4513 - acc: 0.463 - ETA: 1s - loss: 8.4860 - acc: 0.461 - ETA: 1s - loss: 8.4777 - acc: 0.461 - ETA: 1s - loss: 8.4652 - acc: 0.461 - ETA: 0s - loss: 8.4369 - acc: 0.463 - ETA: 0s - loss: 8.4403 - acc: 0.463 - ETA: 0s - loss: 8.4417 - acc: 0.463 - ETA: 0s - loss: 8.4901 - acc: 0.461 - ETA: 0s - loss: 8.4646 - acc: 0.462 - ETA: 0s - loss: 8.4402 - acc: 0.464 - ETA: 0s - loss: 8.4331 - acc: 0.465 - ETA: 0s - loss: 8.4448 - acc: 0.464 - ETA: 0s - loss: 8.4382 - acc: 0.465 - ETA: 0s - loss: 8.4127 - acc: 0.466 - ETA: 0s - loss: 8.4178 - acc: 0.466 - ETA: 0s - loss: 8.4245 - acc: 0.465 - ETA: 0s - loss: 8.4351 - acc: 0.465 - ETA: 0s - loss: 8.4590 - acc: 0.463 - ETA: 0s - loss: 8.4744 - acc: 0.462 - ETA: 0s - loss: 8.4875 - acc: 0.461 - ETA: 0s - loss: 8.5239 - acc: 0.459 - ETA: 0s - loss: 8.5071 - acc: 0.460 - ETA: 0s - loss: 8.4923 - acc: 0.4611Epoch 00013: val_loss improved from 9.09968 to 9.02432, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.4945 - acc: 0.4609 - val_loss: 9.0243 - val_acc: 0.3832
    Epoch 15/20
    6540/6680 [============================>.] - ETA: 2s - loss: 8.0597 - acc: 0.500 - ETA: 2s - loss: 7.7348 - acc: 0.511 - ETA: 2s - loss: 7.6988 - acc: 0.517 - ETA: 2s - loss: 7.6839 - acc: 0.518 - ETA: 2s - loss: 8.0778 - acc: 0.487 - ETA: 1s - loss: 8.2128 - acc: 0.481 - ETA: 1s - loss: 8.2431 - acc: 0.479 - ETA: 1s - loss: 8.2230 - acc: 0.480 - ETA: 1s - loss: 8.2400 - acc: 0.480 - ETA: 1s - loss: 8.2592 - acc: 0.477 - ETA: 1s - loss: 8.3886 - acc: 0.468 - ETA: 1s - loss: 8.3508 - acc: 0.469 - ETA: 1s - loss: 8.4142 - acc: 0.466 - ETA: 1s - loss: 8.4350 - acc: 0.463 - ETA: 1s - loss: 8.4459 - acc: 0.462 - ETA: 1s - loss: 8.4199 - acc: 0.464 - ETA: 1s - loss: 8.3878 - acc: 0.466 - ETA: 1s - loss: 8.4439 - acc: 0.462 - ETA: 1s - loss: 8.4381 - acc: 0.462 - ETA: 1s - loss: 8.3570 - acc: 0.467 - ETA: 1s - loss: 8.4112 - acc: 0.463 - ETA: 1s - loss: 8.4703 - acc: 0.460 - ETA: 1s - loss: 8.4437 - acc: 0.461 - ETA: 1s - loss: 8.4212 - acc: 0.462 - ETA: 0s - loss: 8.4230 - acc: 0.463 - ETA: 0s - loss: 8.3813 - acc: 0.465 - ETA: 0s - loss: 8.3550 - acc: 0.467 - ETA: 0s - loss: 8.3341 - acc: 0.468 - ETA: 0s - loss: 8.3402 - acc: 0.467 - ETA: 0s - loss: 8.3507 - acc: 0.466 - ETA: 0s - loss: 8.3100 - acc: 0.468 - ETA: 0s - loss: 8.2703 - acc: 0.470 - ETA: 0s - loss: 8.2871 - acc: 0.469 - ETA: 0s - loss: 8.2706 - acc: 0.470 - ETA: 0s - loss: 8.2775 - acc: 0.470 - ETA: 0s - loss: 8.2967 - acc: 0.469 - ETA: 0s - loss: 8.3360 - acc: 0.467 - ETA: 0s - loss: 8.3711 - acc: 0.465 - ETA: 0s - loss: 8.3926 - acc: 0.463 - ETA: 0s - loss: 8.4085 - acc: 0.463 - ETA: 0s - loss: 8.4057 - acc: 0.463 - ETA: 0s - loss: 8.4076 - acc: 0.4633Epoch 00014: val_loss improved from 9.02432 to 8.91660, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.3920 - acc: 0.4644 - val_loss: 8.9166 - val_acc: 0.3868
    Epoch 16/20
    6560/6680 [============================>.] - ETA: 2s - loss: 12.0887 - acc: 0.25 - ETA: 2s - loss: 8.2049 - acc: 0.4833 - ETA: 2s - loss: 8.6565 - acc: 0.450 - ETA: 2s - loss: 8.5016 - acc: 0.462 - ETA: 2s - loss: 8.5907 - acc: 0.451 - ETA: 1s - loss: 8.6131 - acc: 0.451 - ETA: 1s - loss: 8.5280 - acc: 0.456 - ETA: 1s - loss: 8.6642 - acc: 0.448 - ETA: 1s - loss: 8.6524 - acc: 0.450 - ETA: 1s - loss: 8.6537 - acc: 0.450 - ETA: 1s - loss: 8.6456 - acc: 0.450 - ETA: 1s - loss: 8.5778 - acc: 0.455 - ETA: 1s - loss: 8.5358 - acc: 0.458 - ETA: 1s - loss: 8.5225 - acc: 0.460 - ETA: 1s - loss: 8.5338 - acc: 0.459 - ETA: 1s - loss: 8.5639 - acc: 0.458 - ETA: 1s - loss: 8.5262 - acc: 0.461 - ETA: 1s - loss: 8.4544 - acc: 0.465 - ETA: 1s - loss: 8.4077 - acc: 0.468 - ETA: 1s - loss: 8.4162 - acc: 0.468 - ETA: 1s - loss: 8.4043 - acc: 0.469 - ETA: 1s - loss: 8.3909 - acc: 0.470 - ETA: 1s - loss: 8.4357 - acc: 0.468 - ETA: 0s - loss: 8.3979 - acc: 0.470 - ETA: 0s - loss: 8.3751 - acc: 0.471 - ETA: 0s - loss: 8.4056 - acc: 0.469 - ETA: 0s - loss: 8.4288 - acc: 0.468 - ETA: 0s - loss: 8.4194 - acc: 0.469 - ETA: 0s - loss: 8.4167 - acc: 0.468 - ETA: 0s - loss: 8.4501 - acc: 0.466 - ETA: 0s - loss: 8.4375 - acc: 0.467 - ETA: 0s - loss: 8.4504 - acc: 0.466 - ETA: 0s - loss: 8.4243 - acc: 0.467 - ETA: 0s - loss: 8.4146 - acc: 0.468 - ETA: 0s - loss: 8.4049 - acc: 0.468 - ETA: 0s - loss: 8.4071 - acc: 0.468 - ETA: 0s - loss: 8.4005 - acc: 0.468 - ETA: 0s - loss: 8.3920 - acc: 0.468 - ETA: 0s - loss: 8.3994 - acc: 0.468 - ETA: 0s - loss: 8.3471 - acc: 0.471 - ETA: 0s - loss: 8.3407 - acc: 0.472 - ETA: 0s - loss: 8.3453 - acc: 0.4716Epoch 00015: val_loss improved from 8.91660 to 8.90740, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.3284 - acc: 0.4728 - val_loss: 8.9074 - val_acc: 0.3820
    Epoch 17/20
    6560/6680 [============================>.] - ETA: 2s - loss: 8.8653 - acc: 0.450 - ETA: 2s - loss: 8.5181 - acc: 0.466 - ETA: 2s - loss: 8.5173 - acc: 0.464 - ETA: 2s - loss: 8.4232 - acc: 0.470 - ETA: 2s - loss: 8.3326 - acc: 0.478 - ETA: 2s - loss: 8.3004 - acc: 0.480 - ETA: 1s - loss: 8.2294 - acc: 0.484 - ETA: 1s - loss: 8.2982 - acc: 0.478 - ETA: 1s - loss: 8.3229 - acc: 0.477 - ETA: 1s - loss: 8.3090 - acc: 0.478 - ETA: 1s - loss: 8.2953 - acc: 0.479 - ETA: 1s - loss: 8.2948 - acc: 0.479 - ETA: 1s - loss: 8.2091 - acc: 0.485 - ETA: 1s - loss: 8.1594 - acc: 0.488 - ETA: 1s - loss: 8.0511 - acc: 0.495 - ETA: 1s - loss: 8.0778 - acc: 0.493 - ETA: 1s - loss: 8.1012 - acc: 0.491 - ETA: 1s - loss: 8.1599 - acc: 0.488 - ETA: 1s - loss: 8.2219 - acc: 0.484 - ETA: 1s - loss: 8.2294 - acc: 0.484 - ETA: 1s - loss: 8.2114 - acc: 0.485 - ETA: 1s - loss: 8.2188 - acc: 0.485 - ETA: 1s - loss: 8.2485 - acc: 0.483 - ETA: 1s - loss: 8.2826 - acc: 0.480 - ETA: 0s - loss: 8.2536 - acc: 0.481 - ETA: 0s - loss: 8.2953 - acc: 0.479 - ETA: 0s - loss: 8.2599 - acc: 0.481 - ETA: 0s - loss: 8.2778 - acc: 0.480 - ETA: 0s - loss: 8.3126 - acc: 0.478 - ETA: 0s - loss: 8.3088 - acc: 0.478 - ETA: 0s - loss: 8.3269 - acc: 0.476 - ETA: 0s - loss: 8.3446 - acc: 0.475 - ETA: 0s - loss: 8.3038 - acc: 0.478 - ETA: 0s - loss: 8.3216 - acc: 0.477 - ETA: 0s - loss: 8.3178 - acc: 0.477 - ETA: 0s - loss: 8.3117 - acc: 0.477 - ETA: 0s - loss: 8.3050 - acc: 0.477 - ETA: 0s - loss: 8.2712 - acc: 0.479 - ETA: 0s - loss: 8.2962 - acc: 0.478 - ETA: 0s - loss: 8.2729 - acc: 0.479 - ETA: 0s - loss: 8.2585 - acc: 0.480 - ETA: 0s - loss: 8.2598 - acc: 0.4803Epoch 00016: val_loss improved from 8.90740 to 8.75791, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.2691 - acc: 0.4795 - val_loss: 8.7579 - val_acc: 0.3892
    Epoch 18/20
    6580/6680 [============================>.] - ETA: 1s - loss: 6.5289 - acc: 0.550 - ETA: 2s - loss: 7.0979 - acc: 0.543 - ETA: 2s - loss: 7.9643 - acc: 0.490 - ETA: 2s - loss: 7.7101 - acc: 0.502 - ETA: 2s - loss: 7.8133 - acc: 0.495 - ETA: 2s - loss: 7.8368 - acc: 0.493 - ETA: 1s - loss: 7.9209 - acc: 0.491 - ETA: 1s - loss: 7.8476 - acc: 0.496 - ETA: 1s - loss: 7.8405 - acc: 0.497 - ETA: 1s - loss: 7.8416 - acc: 0.495 - ETA: 1s - loss: 7.8073 - acc: 0.498 - ETA: 1s - loss: 7.7689 - acc: 0.499 - ETA: 1s - loss: 7.8496 - acc: 0.494 - ETA: 1s - loss: 7.9522 - acc: 0.489 - ETA: 1s - loss: 7.9692 - acc: 0.489 - ETA: 1s - loss: 7.9630 - acc: 0.489 - ETA: 1s - loss: 7.9885 - acc: 0.489 - ETA: 1s - loss: 8.0040 - acc: 0.488 - ETA: 1s - loss: 8.0638 - acc: 0.486 - ETA: 1s - loss: 8.0820 - acc: 0.485 - ETA: 1s - loss: 8.0918 - acc: 0.485 - ETA: 1s - loss: 8.1237 - acc: 0.482 - ETA: 1s - loss: 8.1160 - acc: 0.482 - ETA: 1s - loss: 8.0988 - acc: 0.482 - ETA: 0s - loss: 8.1375 - acc: 0.479 - ETA: 0s - loss: 8.1412 - acc: 0.479 - ETA: 0s - loss: 8.1518 - acc: 0.478 - ETA: 0s - loss: 8.1139 - acc: 0.480 - ETA: 0s - loss: 8.1174 - acc: 0.480 - ETA: 0s - loss: 8.0939 - acc: 0.482 - ETA: 0s - loss: 8.1057 - acc: 0.480 - ETA: 0s - loss: 8.0945 - acc: 0.482 - ETA: 0s - loss: 8.0405 - acc: 0.485 - ETA: 0s - loss: 8.0468 - acc: 0.484 - ETA: 0s - loss: 8.0366 - acc: 0.485 - ETA: 0s - loss: 8.0382 - acc: 0.485 - ETA: 0s - loss: 8.0343 - acc: 0.485 - ETA: 0s - loss: 8.0016 - acc: 0.487 - ETA: 0s - loss: 7.9765 - acc: 0.489 - ETA: 0s - loss: 7.9774 - acc: 0.489 - ETA: 0s - loss: 7.9944 - acc: 0.488 - ETA: 0s - loss: 7.9763 - acc: 0.490 - ETA: 0s - loss: 7.9952 - acc: 0.4892Epoch 00017: val_loss improved from 8.75791 to 8.63126, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.0145 - acc: 0.4880 - val_loss: 8.6313 - val_acc: 0.3988
    Epoch 19/20
    6600/6680 [============================>.] - ETA: 2s - loss: 5.6414 - acc: 0.650 - ETA: 2s - loss: 7.9768 - acc: 0.500 - ETA: 2s - loss: 7.7379 - acc: 0.511 - ETA: 2s - loss: 8.2281 - acc: 0.484 - ETA: 2s - loss: 8.0992 - acc: 0.492 - ETA: 1s - loss: 8.0807 - acc: 0.492 - ETA: 1s - loss: 8.0520 - acc: 0.494 - ETA: 1s - loss: 7.9096 - acc: 0.503 - ETA: 1s - loss: 7.9436 - acc: 0.501 - ETA: 1s - loss: 8.1100 - acc: 0.491 - ETA: 1s - loss: 8.0756 - acc: 0.493 - ETA: 1s - loss: 8.1143 - acc: 0.491 - ETA: 1s - loss: 8.1526 - acc: 0.488 - ETA: 1s - loss: 8.1394 - acc: 0.489 - ETA: 1s - loss: 8.1104 - acc: 0.490 - ETA: 1s - loss: 8.0939 - acc: 0.491 - ETA: 1s - loss: 8.1161 - acc: 0.490 - ETA: 1s - loss: 8.1209 - acc: 0.489 - ETA: 1s - loss: 8.0940 - acc: 0.491 - ETA: 1s - loss: 8.0531 - acc: 0.493 - ETA: 1s - loss: 8.0259 - acc: 0.494 - ETA: 1s - loss: 7.9659 - acc: 0.497 - ETA: 1s - loss: 7.9382 - acc: 0.498 - ETA: 0s - loss: 7.9690 - acc: 0.497 - ETA: 0s - loss: 7.9790 - acc: 0.496 - ETA: 0s - loss: 7.9687 - acc: 0.497 - ETA: 0s - loss: 7.9498 - acc: 0.498 - ETA: 0s - loss: 7.9865 - acc: 0.495 - ETA: 0s - loss: 7.9949 - acc: 0.494 - ETA: 0s - loss: 8.0127 - acc: 0.493 - ETA: 0s - loss: 7.9879 - acc: 0.495 - ETA: 0s - loss: 7.9710 - acc: 0.496 - ETA: 0s - loss: 7.9825 - acc: 0.495 - ETA: 0s - loss: 7.9688 - acc: 0.496 - ETA: 0s - loss: 7.9630 - acc: 0.496 - ETA: 0s - loss: 7.9582 - acc: 0.496 - ETA: 0s - loss: 7.9416 - acc: 0.497 - ETA: 0s - loss: 7.9294 - acc: 0.498 - ETA: 0s - loss: 7.9277 - acc: 0.498 - ETA: 0s - loss: 7.9212 - acc: 0.498 - ETA: 0s - loss: 7.9377 - acc: 0.497 - ETA: 0s - loss: 7.9177 - acc: 0.4988Epoch 00018: val_loss improved from 8.63126 to 8.47696, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 7.9001 - acc: 0.5000 - val_loss: 8.4770 - val_acc: 0.4096
    Epoch 20/20
    6540/6680 [============================>.] - ETA: 2s - loss: 4.0316 - acc: 0.750 - ETA: 2s - loss: 6.6298 - acc: 0.588 - ETA: 2s - loss: 7.4347 - acc: 0.535 - ETA: 2s - loss: 7.1845 - acc: 0.552 - ETA: 2s - loss: 7.6124 - acc: 0.522 - ETA: 1s - loss: 7.6207 - acc: 0.523 - ETA: 1s - loss: 7.7182 - acc: 0.517 - ETA: 1s - loss: 7.7321 - acc: 0.515 - ETA: 1s - loss: 7.7138 - acc: 0.515 - ETA: 1s - loss: 7.6969 - acc: 0.517 - ETA: 1s - loss: 7.7433 - acc: 0.514 - ETA: 1s - loss: 7.8370 - acc: 0.509 - ETA: 1s - loss: 7.8823 - acc: 0.506 - ETA: 1s - loss: 7.8192 - acc: 0.510 - ETA: 1s - loss: 7.8459 - acc: 0.507 - ETA: 1s - loss: 7.7872 - acc: 0.511 - ETA: 1s - loss: 7.8335 - acc: 0.507 - ETA: 1s - loss: 7.8077 - acc: 0.508 - ETA: 1s - loss: 7.8050 - acc: 0.509 - ETA: 1s - loss: 7.8398 - acc: 0.507 - ETA: 1s - loss: 7.8141 - acc: 0.508 - ETA: 1s - loss: 7.8498 - acc: 0.506 - ETA: 1s - loss: 7.8770 - acc: 0.504 - ETA: 1s - loss: 7.8649 - acc: 0.504 - ETA: 0s - loss: 7.8359 - acc: 0.506 - ETA: 0s - loss: 7.8493 - acc: 0.506 - ETA: 0s - loss: 7.8497 - acc: 0.506 - ETA: 0s - loss: 7.8667 - acc: 0.505 - ETA: 0s - loss: 7.8676 - acc: 0.504 - ETA: 0s - loss: 7.8882 - acc: 0.503 - ETA: 0s - loss: 7.8508 - acc: 0.505 - ETA: 0s - loss: 7.8463 - acc: 0.505 - ETA: 0s - loss: 7.8595 - acc: 0.504 - ETA: 0s - loss: 7.8709 - acc: 0.504 - ETA: 0s - loss: 7.8772 - acc: 0.503 - ETA: 0s - loss: 7.8623 - acc: 0.504 - ETA: 0s - loss: 7.8511 - acc: 0.505 - ETA: 0s - loss: 7.8718 - acc: 0.503 - ETA: 0s - loss: 7.8610 - acc: 0.504 - ETA: 0s - loss: 7.8532 - acc: 0.505 - ETA: 0s - loss: 7.8270 - acc: 0.506 - ETA: 0s - loss: 7.8353 - acc: 0.5058Epoch 00019: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 7.8393 - acc: 0.5057 - val_loss: 8.5213 - val_acc: 0.4168





    <keras.callbacks.History at 0x1dc39a36e80>




```python
## 加载具有最好验证loss的模型

VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### 测试模型
现在，我们可以测试此CNN在狗图像测试数据集中识别品种的效果如何。我们在下方打印出测试准确率。


```python
# 获取测试数据集中每一个图像所预测的狗品种的index
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 40.7895%


### 使用模型预测狗的品种


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # 提取bottleneck特征
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # 获取预测向量
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # 返回此模型预测的狗的品种
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## 步骤 5: 建立一个CNN来分类狗的品种（使用迁移学习）

现在你将使用迁移学习来建立一个CNN，从而可以从图像中识别狗的品种。你的 CNN 在测试集上的准确率必须至少达到60%。

在步骤4中，我们使用了迁移学习来创建一个使用基于 VGG-16 提取的特征向量来搭建一个 CNN。在本部分内容中，你必须使用另一个预训练模型来搭建一个 CNN。为了让这个任务更易实现，我们已经预先对目前 keras 中可用的几种网络进行了预训练：

- [VGG-19](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/DLND+documents/DogXceptionData.npz) bottleneck features

这些文件被命名为为：

    Dog{network}Data.npz

其中 `{network}` 可以是 `VGG19`、`Resnet50`、`InceptionV3` 或 `Xception` 中的一个。选择上方网络架构中的一个，下载相对应的bottleneck特征，并将所下载的文件保存在目录 `bottleneck_features/` 中。


### 【练习】获取模型的特征向量

在下方代码块中，通过运行下方代码提取训练、测试与验证集相对应的bottleneck特征。

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: 从另一个预训练的CNN获取bottleneck特征
bottleneck_features_Xception = np.load('bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features_Xception['train']
valid_Xception = bottleneck_features_Xception['valid']
test_Xception = bottleneck_features_Xception['test']
```

### 【练习】模型架构

建立一个CNN来分类狗品种。在你的代码单元块的最后，通过运行如下代码输出网络的结构：
    
        <your model's name>.summary()
   
---

<a id='question6'></a>  

### __问题 6:__ 


在下方的代码块中尝试使用 Keras 搭建最终的网络架构，并回答你实现最终 CNN 架构的步骤与每一步的作用，并描述你在迁移学习过程中，使用该网络架构的原因。


__回答:__ 
GlobalAveragePooling2D层把多维的输入一维化，batch_size不影响
Dropout层防止过拟合，同时加快训练
Dense units=256增加分类信息
Dense units=133，activation='softmax'作为分类的输出

*选择Xpection的原因是Xception模块首先使用1 *1的卷积核将特征图的各个通道映射到一个新的空间，在这一过程中学习通道间的相关性；再通过常规的3 *3或5 *5的卷积核进行卷积，以同时学习空间上的相关性和通道间的相关性。能有更多有效的特征用于分类 

 早期的VGG16迁移学习，本身特征太少，使用全局平均池化层，大大损失了图像的特征
 从Flatten层到最后的输出tensor，参数减少的太快
 可能VGG-16分辨效果不如VGG-19


```python
train_Xception.shape[1:]
```




    (7, 7, 2048)




```python
### TODO: 定义你的框架
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.models import Sequential

Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape = train_Xception.shape[1:]))
Xception_model.add(Dense(units=1024, activation='relu'))
Xception_model.add(Dropout(0.2))
Xception_model.add(Dense(units=256,activation='relu'))
Xception_model.add(Dropout(0.2))
Xception_model.add(Dense(units=133,activation='softmax'))

Xception_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_3 ( (None, 2048)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1024)              2098176   
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 256)               262400    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 133)               34181     
    =================================================================
    Total params: 2,394,757.0
    Trainable params: 2,394,757.0
    Non-trainable params: 0.0
    _________________________________________________________________



```python
### TODO: 编译模型
Xception_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

### 【练习】训练模型

<a id='question7'></a>  

### __问题 7:__ 

在下方代码单元中训练你的模型。使用模型检查点（model checkpointing）来储存具有最低验证集 loss 的模型。

当然，你也可以对训练集进行 [数据增强](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 以优化模型的表现，不过这不是必须的步骤。



```python
### TODO: 训练模型
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5',
                                verbose=1, save_best_only=True)

Xception_model.fit(train_Xception, train_targets,
                validation_data=(valid_Xception,valid_targets),
               epochs=10, batch_size=20, callbacks=[checkpointer],verbose=1)
```
    Epoch 10/10
    6660/6680 [============================>.] - ETA: 8s - loss: 0.1982 - acc: 0.950 - ETA: 7s - loss: 0.2594 - acc: 0.900 - ETA: 7s - loss: 0.2090 - acc: 0.921 - ETA: 7s - loss: 0.1633 - acc: 0.945 - ETA: 7s - loss: 0.1569 - acc: 0.946 - ETA: 7s - loss: 0.1568 - acc: 0.953 - ETA: 7s - loss: 0.1803 - acc: 0.947 - ETA: 7s - loss: 0.1835 - acc: 0.942 - ETA: 7s - loss: 0.1802 - acc: 0.941 - ETA: 7s - loss: 0.1663 - acc: 0.944 - ETA: 7s - loss: 0.1600 - acc: 0.946 - ETA: 7s - loss: 0.1588 - acc: 0.946 - ETA: 7s - loss: 0.1548 - acc: 0.947 - ETA: 7s - loss: 0.1629 - acc: 0.943 - ETA: 6s - loss: 0.1671 - acc: 0.942 - ETA: 6s - loss: 0.1658 - acc: 0.944 - ETA: 6s - loss: 0.1646 - acc: 0.943 - ETA: 6s - loss: 0.1663 - acc: 0.941 - ETA: 6s - loss: 0.1600 - acc: 0.945 - ETA: 6s - loss: 0.1710 - acc: 0.944 - ETA: 6s - loss: 0.1665 - acc: 0.945 - ETA: 6s - loss: 0.1656 - acc: 0.945 - ETA: 6s - loss: 0.1690 - acc: 0.944 - ETA: 6s - loss: 0.1863 - acc: 0.943 - ETA: 6s - loss: 0.1957 - acc: 0.941 - ETA: 6s - loss: 0.1998 - acc: 0.939 - ETA: 6s - loss: 0.1969 - acc: 0.940 - ETA: 6s - loss: 0.1944 - acc: 0.941 - ETA: 6s - loss: 0.1985 - acc: 0.941 - ETA: 5s - loss: 0.2019 - acc: 0.941 - ETA: 5s - loss: 0.2045 - acc: 0.940 - ETA: 5s - loss: 0.2012 - acc: 0.940 - ETA: 5s - loss: 0.1965 - acc: 0.941 - ETA: 5s - loss: 0.1986 - acc: 0.940 - ETA: 5s - loss: 0.1970 - acc: 0.941 - ETA: 5s - loss: 0.1957 - acc: 0.941 - ETA: 5s - loss: 0.1922 - acc: 0.942 - ETA: 5s - loss: 0.1887 - acc: 0.943 - ETA: 5s - loss: 0.1866 - acc: 0.945 - ETA: 5s - loss: 0.1846 - acc: 0.944 - ETA: 5s - loss: 0.1869 - acc: 0.943 - ETA: 5s - loss: 0.1848 - acc: 0.944 - ETA: 5s - loss: 0.1828 - acc: 0.944 - ETA: 4s - loss: 0.1879 - acc: 0.944 - ETA: 4s - loss: 0.1845 - acc: 0.944 - ETA: 4s - loss: 0.1856 - acc: 0.944 - ETA: 4s - loss: 0.1849 - acc: 0.944 - ETA: 4s - loss: 0.1832 - acc: 0.944 - ETA: 4s - loss: 0.1808 - acc: 0.945 - ETA: 4s - loss: 0.1804 - acc: 0.945 - ETA: 4s - loss: 0.1817 - acc: 0.944 - ETA: 4s - loss: 0.1846 - acc: 0.944 - ETA: 4s - loss: 0.1829 - acc: 0.945 - ETA: 4s - loss: 0.1823 - acc: 0.945 - ETA: 4s - loss: 0.1856 - acc: 0.944 - ETA: 4s - loss: 0.1844 - acc: 0.945 - ETA: 4s - loss: 0.1835 - acc: 0.945 - ETA: 4s - loss: 0.1823 - acc: 0.946 - ETA: 4s - loss: 0.1809 - acc: 0.946 - ETA: 4s - loss: 0.1835 - acc: 0.946 - ETA: 4s - loss: 0.1827 - acc: 0.946 - ETA: 3s - loss: 0.1815 - acc: 0.946 - ETA: 3s - loss: 0.1812 - acc: 0.946 - ETA: 3s - loss: 0.1823 - acc: 0.946 - ETA: 3s - loss: 0.1830 - acc: 0.946 - ETA: 3s - loss: 0.1835 - acc: 0.945 - ETA: 3s - loss: 0.1842 - acc: 0.944 - ETA: 3s - loss: 0.1848 - acc: 0.944 - ETA: 3s - loss: 0.1849 - acc: 0.944 - ETA: 3s - loss: 0.1865 - acc: 0.944 - ETA: 3s - loss: 0.1851 - acc: 0.944 - ETA: 3s - loss: 0.1839 - acc: 0.945 - ETA: 3s - loss: 0.1828 - acc: 0.945 - ETA: 3s - loss: 0.1830 - acc: 0.945 - ETA: 3s - loss: 0.1824 - acc: 0.945 - ETA: 3s - loss: 0.1819 - acc: 0.945 - ETA: 3s - loss: 0.1834 - acc: 0.945 - ETA: 3s - loss: 0.1861 - acc: 0.945 - ETA: 2s - loss: 0.1851 - acc: 0.945 - ETA: 2s - loss: 0.1850 - acc: 0.945 - ETA: 2s - loss: 0.1852 - acc: 0.945 - ETA: 2s - loss: 0.1851 - acc: 0.944 - ETA: 2s - loss: 0.1850 - acc: 0.944 - ETA: 2s - loss: 0.1843 - acc: 0.944 - ETA: 2s - loss: 0.1852 - acc: 0.944 - ETA: 2s - loss: 0.1850 - acc: 0.944 - ETA: 2s - loss: 0.1857 - acc: 0.943 - ETA: 2s - loss: 0.1858 - acc: 0.942 - ETA: 2s - loss: 0.1854 - acc: 0.942 - ETA: 2s - loss: 0.1843 - acc: 0.943 - ETA: 2s - loss: 0.1839 - acc: 0.943 - ETA: 2s - loss: 0.1833 - acc: 0.943 - ETA: 2s - loss: 0.1831 - acc: 0.943 - ETA: 1s - loss: 0.1833 - acc: 0.943 - ETA: 1s - loss: 0.1863 - acc: 0.942 - ETA: 1s - loss: 0.1872 - acc: 0.942 - ETA: 1s - loss: 0.1876 - acc: 0.941 - ETA: 1s - loss: 0.1873 - acc: 0.941 - ETA: 1s - loss: 0.1891 - acc: 0.941 - ETA: 1s - loss: 0.1885 - acc: 0.941 - ETA: 1s - loss: 0.1902 - acc: 0.941 - ETA: 1s - loss: 0.1915 - acc: 0.940 - ETA: 1s - loss: 0.1932 - acc: 0.940 - ETA: 1s - loss: 0.1933 - acc: 0.940 - ETA: 1s - loss: 0.1948 - acc: 0.940 - ETA: 1s - loss: 0.1937 - acc: 0.940 - ETA: 1s - loss: 0.1932 - acc: 0.940 - ETA: 1s - loss: 0.1922 - acc: 0.940 - ETA: 1s - loss: 0.1930 - acc: 0.940 - ETA: 1s - loss: 0.1933 - acc: 0.940 - ETA: 0s - loss: 0.1935 - acc: 0.940 - ETA: 0s - loss: 0.1943 - acc: 0.940 - ETA: 0s - loss: 0.1960 - acc: 0.939 - ETA: 0s - loss: 0.1964 - acc: 0.939 - ETA: 0s - loss: 0.1977 - acc: 0.939 - ETA: 0s - loss: 0.1989 - acc: 0.939 - ETA: 0s - loss: 0.1980 - acc: 0.939 - ETA: 0s - loss: 0.1986 - acc: 0.938 - ETA: 0s - loss: 0.2002 - acc: 0.938 - ETA: 0s - loss: 0.1993 - acc: 0.938 - ETA: 0s - loss: 0.2000 - acc: 0.938 - ETA: 0s - loss: 0.2005 - acc: 0.938 - ETA: 0s - loss: 0.2000 - acc: 0.938 - ETA: 0s - loss: 0.1995 - acc: 0.938 - ETA: 0s - loss: 0.2007 - acc: 0.938 - ETA: 0s - loss: 0.2032 - acc: 0.937 - ETA: 0s - loss: 0.2037 - acc: 0.9375Epoch 00009: val_loss did not improve
    6680/6680 [==============================] - 8s - loss: 0.2035 - acc: 0.9376 - val_loss: 0.7865 - val_acc: 0.8216





    <keras.callbacks.History at 0x1dc3e4f5dd8>




```python
### TODO: 加载具有最佳验证loss的模型权重
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')
```

---

### 【练习】测试模型

<a id='question8'></a>  

### __问题 8:__ 

在狗图像的测试数据集上试用你的模型。确保测试准确率大于60%。


```python
### TODO: 在测试集上计算分类准确率
Xception_predictions = [np.argmax(Xception_model.predict(np.expand_dims(feature,axis=0))) for feature in test_Xception]

test_accuracy = 100*np.sum(np.array(Xception_predictions)==np.argmax(test_targets, axis=1))/len(Xception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 80.9809%


---

### 【练习】使用模型测试狗的品种


实现一个函数，它的输入为图像路径，功能为预测对应图像的类别，输出为你模型预测出的狗类别（`Affenpinscher`, `Afghan_hound` 等）。

与步骤5中的模拟函数类似，你的函数应当包含如下三个步骤：

1. 根据选定的模型载入图像特征（bottleneck features）
2. 将图像特征输输入到你的模型中，并返回预测向量。注意，在该向量上使用 argmax 函数可以返回狗种类的序号。
3. 使用在步骤0中定义的 `dog_names` 数组来返回对应的狗种类名称。

提取图像特征过程中使用到的函数可以在 `extract_bottleneck_features.py` 中找到。同时，他们应已在之前的代码块中被导入。根据你选定的 CNN 网络，你可以使用 `extract_{network}` 函数来获得对应的图像特征，其中 `{network}` 代表 `VGG19`, `Resnet50`, `InceptionV3`, 或 `Xception` 中的一个。
 
---

<a id='question9'></a>  

### __问题 9:__


```python
### TODO: 写一个函数，该函数将图像的路径作为输入
### 然后返回此模型所预测的狗的品种
import extract_bottleneck_features
def predict_dog_class(image_path):
    # 提取bottleneck特征
    bottleneck_Xception_feature = extract_Xception(path_to_tensor(image_path))
    # 获得预测向量
    predicted_vec = Xception_model.predict(bottleneck_Xception_feature)
    # 返回预测种类
    return dog_names[np.argmax(predicted_vec)]
```

---

<a id='step6'></a>
## 步骤 6: 完成你的算法



实现一个算法，它的输入为图像的路径，它能够区分图像是否包含一个人、狗或两者都不包含，然后：

- 如果从图像中检测到一只__狗__，返回被预测的品种。
- 如果从图像中检测到__人__，返回最相像的狗品种。
- 如果两者都不能在图像中检测到，输出错误提示。

我们非常欢迎你来自己编写检测图像中人类与狗的函数，你可以随意地使用上方完成的 `face_detector` 和 `dog_detector` 函数。你__需要__在步骤5使用你的CNN来预测狗品种。

下面提供了算法的示例输出，但你可以自由地设计自己的模型！

![Sample Human Output](images/sample_human_output.png)




<a id='question10'></a>  

### __问题 10:__

在下方代码块中完成你的代码。

---



```python
### TODO: 设计你的算法
### 自由地使用所需的代码单元数吧
def image_check(image_paths):
    
    if face_detector(image_paths):      #如果检测到的是人
        #返回最相像的狗品种
        looks_like_dog_type = predict_dog_class(image_paths)
        #加载彩色（通道顺序为BGR）图像
        human_img = cv2.imread(image_paths)
        # 将BGR图像进行灰度处理
        human_gray = cv2.cvtColor(human_img,cv2.COLOR_BGR2GRAY)
        # 在图像中找出脸
        faces = face_cascade.detectMultiScale(human_gray)
        # 获取每一个所检测到的脸的识别框
        for (x,y,w,h) in faces:
        # 在人脸图像中绘制出识别框
            cv2.rectangle(human_img,(x,y),(x+w,y+h),(255,0,0),2) 
        # 将BGR图像转变为RGB图像以打印
        cv_rgb = cv2.cvtColor(human_img, cv2.COLOR_BGR2RGB)
        # 展示含有识别框的图像
        plt.imshow(cv_rgb)
        plt.show()
        # 打印图像,并输出你最像的狗的种类是：
        print("You mostly looks like a {} for a dog breed..\n\n".format(looks_like_dog_type))
    elif dog_detector(image_paths):      #如果检测到狗狗,返回被预测的狗的品种
        print("Hey, awesome dog \n\n")
        dog_breed = predict_dog_class(image_paths)
        dog_img = cv2.imread(image_paths)
        plt.imshow(dog_img)
        plt.show()
        print("it's breed is {}.\n\n".format(dog_breed))
    else:
        print("sorry,it's a wrong image ,i don't know it is!")
```

---
<a id='step7'></a>
## 步骤 7: 测试你的算法

在这个部分中，你将尝试一下你的新算法！算法认为__你__看起来像什么类型的狗？如果你有一只狗，它可以准确地预测你的狗的品种吗？如果你有一只猫，它会将你的猫误判为一只狗吗？


<a id='question11'></a>  

### __问题 11:__

在下方编写代码，用至少6张现实中的图片来测试你的算法。你可以使用任意照片，不过请至少使用两张人类图片（要征得当事人同意哦）和两张狗的图片。
同时请回答如下问题：

1. 输出结果比你预想的要好吗 :) ？或者更糟 :( ？
2. 提出至少三点改进你的模型的想法。

1.更好

2.首先，可以降低学习率增大epochs；其次增加输出前的fc中的units数目；最后，可以尝试ResNet-50的迁移学习方案；


```python
## TODO: 在你的电脑上，在步骤6中，至少在6张图片上运行你的算法。
## 自由地使用所需的代码单元数吧
print("dog classifier demo")
for i in range(3):
    index = int(np.random.choice(len(human_files_short), 1, replace=True))
    image_check(human_files_short[index])
    image_check(dog_files_short[index])
```

    dog classifier demo



![png](output_64_1.png)


    ^___^,You mostly looks like a Smooth_fox_terrier for a dog breed..
    
    



![png](output_64_3.png)


    ^___^,You mostly looks like a Icelandic_sheepdog for a dog breed..
    
    



![png](output_64_5.png)


    ^___^,You mostly looks like a Smooth_fox_terrier for a dog breed..
    
    
    Hey, awesome dog 
    
    



![png](output_64_7.png)


    it's breed is American_staffordshire_terrier.
    
    



![png](output_64_9.png)


    ^___^,You mostly looks like a Petit_basset_griffon_vendeen for a dog breed..
    
    
    Hey, awesome dog 
    
    



![png](output_64_11.png)


    it's breed is Curly-coated_retriever.
    
    


**注意: 当你写完了所有的代码，并且回答了所有的问题。你就可以把你的 iPython Notebook 导出成 HTML 文件。你可以在菜单栏，这样导出File -> Download as -> HTML (.html)把这个 HTML 和这个 iPython notebook 一起做为你的作业提交。**
