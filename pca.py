import numpy as np
import cv2
import os, random
import sys  # 引用sys库
from tkinter import filedialog

from PyQt5 import QtWidgets  # 引用PyQt5库里QtWidgets类
from PyQt5.QtWidgets import *  # 导入PyQt5.QtWidgets里所有的方法
from PyQt5.QtGui import *  # 导入PyQt5.QtGui里所有的方法

bestten = np.zeros(10)


def get_images(path):
    image_list = []
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(('png')):
                image_list.append(os.path.join(parent, filename))
    return image_list


def load_images(path=u'./FaceDB_orl', split_rate=1.0):  # 加载图像集，随机选择sampleCount张图片用于训练
    data = []  # 单个文件夹下的数据集
    x_train = []  # 总训练集
    x_test = []  # 总测试集
    y_train = []  # 总训练集的标签
    y_test = []  # 总测试集的标签

    # 遍历40个文件夹
    for k in range(40):
        folder = os.path.join(path, '%03d' % (k + 1))  # 当前文件夹

        data = [cv2.imread(image, 0) for image in get_images(folder)]  # ① cv2.imread()读取灰度图，0表示灰度图模式

        data_train_num = int(np.array(data).shape[0] * split_rate)

        data_train_indexs = random.sample(range(10),
                                          data_train_num)  # random.data_train_indexs()从0-9中随机选择sampleCount个元素，return a new list

        x_train.extend([data[i].ravel() for i in range(10) if i in data_train_indexs])
        x_test.extend([data[i].ravel() for i in range(10) if i not in data_train_indexs])

        y_train.extend([k] * data_train_num)  # 将文件夹名作为标签
        y_test.extend([k] * (10 - data_train_num))

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def pca(x_train, dim):
    '''
    主成分分析，将10304维度的数据降维到100维
    :param x_train: 训练集
    :param dim: 降到k维
    :return:
    '''
    x_train = np.asmatrix(x_train, np.float32)  # 转换成矩阵
    num_train = x_train.shape[0]  # 取矩阵的维度 → (320, 10304)

    # 求每一行的均值
    data_mean = np.mean(x_train, axis=0)  # axis = 0：压缩行，对各列求均值 → 1 * n 矩阵

    # 零均值化：让矩阵X_train减去每一行的均值，得到零均值化后的矩阵Z
    Z = x_train - np.tile(data_mean, (num_train, 1))  # np.tile()用于复制，这里让data_mean.shape从(1, 10304) → (320, 10304)

    D, V = np.linalg.eig(Z * Z.T)  # 求协方差矩阵的特征值与特征向量

    # V1.shape - (320,100)
    V1 = V[:, 0:dim]  # 按列取前dim个特征向量（降到多少维就取前多少个特征向量）

    V1 = Z.T * V1  # 小矩阵特征向量向大矩阵特征向量过渡

    # 降维 - Z*V1
    return np.array(Z * V1), data_mean, V1


def predict(xTrain, yTrain, num_train, data_mean, x_test, V):
    # 降维处理
    x_test_low_dim = np.array((x_test - np.tile(data_mean, (1, 1))) * V)

    predict_result = yTrain[np.sum((xTrain - np.tile(x_test_low_dim, (num_train, 1))) ** 2, axis=1).argmin()]
    print(yTrain[np.sum((xTrain - np.tile(x_test_low_dim, (num_train, 1))) ** 2, axis=1).argsort()[:10]] + 1)
    global bestten
    bestten = np.sum((xTrain - np.tile(x_test_low_dim, (num_train, 1))) ** 2, axis=1).argsort()[:10] + 1
    print('bestten', bestten)

    print('欧式距离识别的编号为 %d' % (predict_result + 1))


x_train, y_train, x_test, y_test = load_images()  # (320,10304); (320); (80, 10304); (80);
num_train, num_test = x_train.shape[0], x_test.shape[0]

# 训练pca模型
print("Start Traning.")
x_train_low_dim, data_mean, V = pca(x_train, 300)  # shape(320, 100)
print("Finish Traning.")


def predict_test(filename):
    print("\nStart Predicting.")
    # test_img = "test\\" + filename[-8:]
    test_img = filename
    predict(x_train_low_dim, y_train, num_train, data_mean, cv2.imread(test_img, 0).ravel(), V)
    print("Finish Predicting.")


class Qt_Window(QWidget):  # 定义一个类，继承于QWidget
    def __init__(self):  # 构建方法
        self._app = QtWidgets.QApplication([])  # 创建QApplication示例
        super(Qt_Window, self).__init__()  # 固定格式
        self.file_path = ''

    def init_ui(self):  # 定义方法，在该方法里构建界面组件
        self.win = QMainWindow()

        # 定义组件
        self.open_Button_address = QPushButton(self.win)  # 选择地址
        self.open_Button = QPushButton(self.win)  # 退出按钮
        self.detect_image = QLabel(self.win)  # 图片（目前为标签控件，需要后续将其转换为图片控件）
        self.detect_image1 = QLabel(self.win)  # 图片（目前为标签控件，需要后续将其转换为图片控件）
        self.detect_image2 = QLabel(self.win)  # 图片（目前为标签控件，需要后续将其转换为图片控件）
        self.detect_image3 = QLabel(self.win)  # 图片（目前为标签控件，需要后续将其转换为图片控件）
        self.detect_image4 = QLabel(self.win)  # 图片（目前为标签控件，需要后续将其转换为图片控件）
        self.detect_image5 = QLabel(self.win)  # 图片（目前为标签控件，需要后续将其转换为图片控件）
        self.detect_image6 = QLabel(self.win)  # 图片（目前为标签控件，需要后续将其转换为图片控件）
        self.detect_image7 = QLabel(self.win)  # 图片（目前为标签控件，需要后续将其转换为图片控件）
        self.detect_image8 = QLabel(self.win)  # 图片（目前为标签控件，需要后续将其转换为图片控件）
        self.detect_image0 = QLabel(self.win)  # 图片（目前为标签控件，需要后续将其转换为图片控件）

        self.label1 = QLabel(self.win)  # 文字
        self.label2 = QLabel(self.win)  # 文字
        self.label3 = QLabel(self.win)  # 文字
        self.label4 = QLabel(self.win)  # 文字
        self.label5 = QLabel(self.win)  # 文字
        self.label6 = QLabel(self.win)  # 文字
        self.label7 = QLabel(self.win)  # 文字
        self.label8 = QLabel(self.win)  # 文字
        self.label9 = QLabel(self.win)  # 文字
        self.label0 = QLabel(self.win)  # 文字
        self.label_txt = QLabel(self.win)  # 文字

        # 设置控件
        self.open_Button_address.resize(200, 50)
        self.open_Button_address.move(600, 800)
        self.open_Button_address.setText("选择文件")
        self.open_Button_address.setCheckable(True)
        self.open_Button_address.clicked.connect(self.select_file)

        self.open_Button.resize(200, 50)
        self.open_Button.move(1000, 800)
        self.open_Button.setText("退出")
        self.open_Button.setCheckable(True)
        self.open_Button.clicked.connect(self.exit)

        self.detect_image.resize(92, 112)
        self.detect_image.move(450, 100)
        self.label0.move(500, 500)
        self.label_txt.move(450, 500)
        self.label_txt.setText('人脸应是')

        self.detect_image0.resize(92, 112)
        self.detect_image0.move(900, 100)
        self.detect_image1.resize(92, 112)
        self.detect_image1.move(1100, 100)
        self.detect_image2.resize(92, 112)
        self.detect_image2.move(1300, 100)

        self.label1.move(900, 220)
        self.label2.move(1100, 220)
        self.label3.move(1300, 220)

        self.detect_image3.resize(92, 112)
        self.detect_image3.move(900, 300)
        self.detect_image4.resize(92, 112)
        self.detect_image4.move(1100, 300)
        self.detect_image5.resize(92, 112)
        self.detect_image5.move(1300, 300)

        self.label4.move(900, 420)
        self.label5.move(1100, 420)
        self.label6.move(1300, 420)

        self.detect_image6.resize(92, 112)
        self.detect_image6.move(900, 500)
        self.detect_image7.resize(92, 112)
        self.detect_image7.move(1100, 500)
        self.detect_image8.resize(92, 112)
        self.detect_image8.move(1300, 500)

        self.label7.move(900, 620)
        self.label8.move(1100, 620)
        self.label9.move(1300, 620)

        self.win.showMaximized()
        sys.exit(self._app.exec_())

    def exit(self):  # 定义关闭事件
        if self.open_Button.isChecked():
            sys.exit(0)  # sys.exit(0)为正常退出  sys.exit(1)为异常退出

    def select_file(self):
        self.file_path = filedialog.askopenfilename()
        print("选择的文件路径：" + self.file_path)
        predict_test(self.file_path)
        self.show_img()

    def show_img(self):
        pix = QPixmap(self.file_path)
        self.detect_image.setPixmap(pix)

        file_path1 = 'FaceDB_orl\\0'
        print(file_path1 + '{:0>2}'.format(bestten[1] // 10) + "\\" + '{:0>2}'.format(bestten[1] % 10))
        pix = QPixmap(
            file_path1 + "{:0>2}".format((bestten[0] + 9) // 10) + "\\" + "{:0>2}".format(bestten[0] % 10 + 1) + ".png")
        self.detect_image0.setPixmap(pix)
        pix = QPixmap(
            file_path1 + "{:0>2}".format((bestten[1] + 9) // 10) + "\\" + "{:0>2}".format(bestten[1] % 10 + 1) + ".png")
        self.detect_image1.setPixmap(pix)
        pix = QPixmap(
            file_path1 + "{:0>2}".format((bestten[2] + 9) // 10) + "\\" + "{:0>2}".format(bestten[2] % 10 + 1) + ".png")
        self.detect_image2.setPixmap(pix)
        pix = QPixmap(
            file_path1 + "{:0>2}".format((bestten[3] + 9) // 10) + "\\" + "{:0>2}".format(bestten[3] % 10 + 1) + ".png")
        self.detect_image3.setPixmap(pix)
        pix = QPixmap(
            file_path1 + "{:0>2}".format((bestten[4] + 9) // 10) + "\\" + "{:0>2}".format(bestten[4] % 10 + 1) + ".png")
        self.detect_image4.setPixmap(pix)
        pix = QPixmap(
            file_path1 + "{:0>2}".format((bestten[5] + 9) // 10) + "\\" + "{:0>2}".format(bestten[5] % 10 + 1) + ".png")
        self.detect_image5.setPixmap(pix)
        pix = QPixmap(
            file_path1 + "{:0>2}".format((bestten[6] + 9) // 10) + "\\" + "{:0>2}".format(bestten[6] % 10 + 1) + ".png")
        self.detect_image6.setPixmap(pix)
        pix = QPixmap(
            file_path1 + "{:0>2}".format((bestten[7] + 9) // 10) + "\\" + "{:0>2}".format(bestten[7] % 10 + 1) + ".png")
        self.detect_image7.setPixmap(pix)
        pix = QPixmap(
            file_path1 + "{:0>2}".format((bestten[8] + 9) // 10) + "\\" + "{:0>2}".format(bestten[8] % 10 + 1) + ".png")
        self.detect_image8.setPixmap(pix)

        self.label0.setText("{:0>2}".format((bestten[0] + 9) // 10))
        self.label1.setText('1th  ' + "{:0>2}".format((bestten[0] + 9) // 10))
        self.label2.setText('2nd  ' + "{:0>2}".format((bestten[1] + 9) // 10))
        self.label3.setText('3rd  ' + "{:0>2}".format((bestten[2] + 9) // 10))
        self.label4.setText('4th  ' + "{:0>2}".format((bestten[3] + 9) // 10))
        self.label5.setText('5th  ' + "{:0>2}".format((bestten[4] + 9) // 10))
        self.label6.setText('6th  ' + "{:0>2}".format((bestten[5] + 9) // 10))
        self.label7.setText('7th  ' + "{:0>2}".format((bestten[6] + 9) // 10))
        self.label8.setText('8th  ' + "{:0>2}".format((bestten[7] + 9) // 10))
        self.label9.setText('9th  ' + "{:0>2}".format((bestten[8] + 9) // 10))


s = Qt_Window()
s.init_ui()
