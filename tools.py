import os
from xml.dom.minidom import parse
import xml.dom.minidom
import matplotlib.pyplot as plt
import cv2
from utils.utils import *
from models import *

# hyp = {'giou': 1.582,  # giou loss gain
#        'xy': 4.688,  # xy loss gain
#        'wh': 0.1857,  # wh loss gain
#        'cls': 27.76,  # cls loss gain
#        'cls_pw': 1.446,  # cls BCELoss positive_weight
#        'obj': 21.35,  # obj loss gain
#        'obj_pw': 3.941,  # obj BCELoss positive_weight
#        'iou_t': 0.2635,  # iou training threshold
#        # 'lr0': 0.002324,  # initial learning rate
#        'lr0': 0.001000,
#        'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.97,  # SGD momentum
#        'weight_decay': 0.0004569,  # optimizer weight decay
#        'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
#        'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
#        'degrees': 1.113,  # image rotation (+/- deg)
#        'translate': 0.06797,  # image translation (+/- fraction)
#        'scale': 0.1059,  # image scale (+/- gain)
#        'shear': 0.5768}  # image shear (+/- deg)


"获取检测图像的ground truth的左上角和右下角坐标值"
def xml_to_txt(xmlPath):

   DOMTree = xml.dom.minidom.parse(xmlPath)
   collection = DOMTree.documentElement

   coordinate = []
   label = []

   objects = collection.getElementsByTagName("object")
   for obj in objects:
       x1 = obj.getElementsByTagName('xmin')[0].firstChild.data
       y1 = obj.getElementsByTagName('ymin')[0].firstChild.data
       x2 = obj.getElementsByTagName('xmax')[0].firstChild.data
       y2 = obj.getElementsByTagName('ymax')[0].firstChild.data
       name = obj.getElementsByTagName('name')[0].firstChild.data
       coordinate.append([x1,y1,x2,y2])
       label.append([name])

   return coordinate,label

"计算预测框和原始框的交并比(暂时不用)"
def calIOU_V2(rec1, rec2):
    """
    computing IoU
    :param rec1: x1,y1,x2,y2
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    cx1 = int(rec1[0])
    cy1 = int(rec1[1])
    cx2 = int(rec1[2])
    cy2 = int(rec1[3])
    gx1 = int(rec2[0])
    gy1 = int(rec2[1])
    gx2 = int(rec2[2])
    gy2 = int(rec2[3])
    # cx1, cy1, cx2, cy2 = rec1
    # gx1, gy1, gx2, gy2 = rec2
    # 计算每个矩形的面积
    S_rec1 = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    S_rec2 = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    # 计算相交矩形
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = area / (S_rec1 + S_rec2 - area)
    return iou

# with open('PR.txt', 'r') as f:
#     context = [x for x in f.read().splitlines()]
# # print(list)
# precision = context[0].split(' ')
# precision = list(map(float, precision))
# recall = context[1].split(' ')
# recall = list(map(float, recall))
# mAP = context[2].split(' ')
# mAP = list(map(float, mAP))
# F1 = context[3].split(' ')
# F1 = list(map(float, F1))

"绘制训练结果"
def train_plot(result1):

    p1_list = []
    r1_list = []
    map1_list = []
    f1_list = []
    # p2_list = []
    # r2_list = []
    # map2_list = []
    # f2_list = []
    x = []
    # y = []#epoch:180
    # for i in range(180):
    #     y.append(int(i))
    with open(result1, 'r') as f:
        list = [x for x in f.read().splitlines()]
        # print(list)
    for i in range(len(list)):
        p1_list.append(float(list[i][93:101]))
        r1_list.append(float(list[i][104:112]))
        map1_list.append(float(list[i][115:123]))
        f1_list.append(float(list[i][126:134]))
        x.append(int(i))
    # with open(result2, 'r') as f:
    #     list = [x for x in f.read().splitlines()]
        # print(list)
    # for i in range(len(list)):
    #     p2_list.append(float(list[i][93:101]))
    #     r2_list.append(float(list[i][104:112]))
    #     map2_list.append(float(list[i][115:123]))
    #     f2_list.append(float(list[i][126:134]))

    titles = ['Precision', 'Recall', 'mAP', 'F1 score']
    "Add section"
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 5))
    plt.rc('font', family = 'Times New Roman')
    plt.tight_layout(3.0)
    "The first image for the p"
    plt.sca(ax1)
    ax1.plot(x, p1_list, '-', lw=1, color='b', label='YOLOv3-SPP')
    # ax1.plot(y, p2_list, '-', lw=1, color='r', label='YOLOv3-SPP')
    plt.xlabel('epoch', fontweight='semibold')
    plt.ylabel('accuracy', fontweight='semibold')
    plt.legend(loc='best')
    plt.title(titles[0], fontweight='heavy')
    "The second image for the r"
    plt.sca(ax2)
    ax2.plot(x, r1_list, '-', lw=1, color='b', label='YOLOv3-SPP')
    # ax2.plot(y, r2_list, '-', lw=1, color='r', label='YOLOv3-SPP')
    plt.xlabel('epoch', fontweight='semibold')
    plt.ylabel('accuracy', fontweight='semibold')
    plt.legend(loc='best')
    plt.title(titles[1], fontweight='heavy')
    "The third image for the mAP"
    plt.sca(ax3)
    ax3.plot(x, map1_list, '-', lw=1, color='b', label='YOLOv3-SPP')
    # ax3.plot(y, map2_list, '-', lw=1, color='r', label='YOLOv3-SPP')
    plt.xlabel('epoch', fontweight='semibold')
    plt.ylabel('accuracy', fontweight='semibold')
    plt.legend(loc='best')
    plt.title(titles[2], fontweight='heavy')
    "The forth image for the f1"
    plt.sca(ax4)
    ax4.plot(x, f1_list, '-', lw=1, color='b', label='YOLOv3-SPP')
    # ax4.plot(y, f2_list, '-', lw=1, color='r', label='YOLOv3-SPP')
    plt.xlabel('epoch', fontweight='semibold')
    plt.ylabel('accuracy', fontweight='semibold')
    plt.legend(loc='best')
    plt.title(titles[3], fontweight='heavy')
    # plt.plot(x, p_list, '-', lw=1, color='r', label='yolov3-spc2')
    # plt.plot(x, r_list, '-', lw=1, color='g', label='Recall')
    # plt.plot(x, map_list, '-', lw=1, color='b', label='mAP')
    # plt.plot(x, f1_list, '-', lw=1, color='y', label='f1 score')
    # plt.xlabel("epoch")
    # plt.ylabel("accuracy")
    # plt.legend(loc="best")
    plt.savefig('pr.jpg', bbox_inches = 'tight', pad_inches = 0, dpi=600)
    # plt.savefig('pr.pdf')
    # plt.savefig('pr.eps')
    # plt.show()

# train_plot('results.txt')


# with open('../yolov3-master1/loss.txt', 'r') as f:
#     context = [x for x in f.read().splitlines()]
# print(list)
# iou = context[0].split(' ')
# iou = list(map(float, iou))
# conf = context[1].split(' ')
# conf = list(map(float, conf))
# cls = context[2].split(' ')
# cls = list(map(float, cls))

def loss_plot(start, stop):
    x = [] #epoch:180
    for i in range(stop):
        x.append(int(i))
    # y = []#epoch:140
    # for i in range(140):
    #     y.append(int(i))
    titles = ['IoU_loss', 'Conf_loss', 'Cls_loss']
    iou_loss, conf_loss, cls_loss = plot_loss(start,stop)
    iou_loss = iou_loss.tolist()
    conf_loss = conf_loss.tolist()
    cls_loss = cls_loss.tolist()


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,5))
    plt.rc('font', family='Times New Roman')
    "The first image for the IoU Loss"
    plt.sca(ax1)
    ax1.plot(x, iou_loss, '-', lw=1, color='b', label='YOLOv3-SPP')
    # ax1.plot(x, iou, '-', lw=1, color='r', label='YOLOv3-SPP')
    plt.xlabel('epoch',fontweight='semibold')
    plt.ylabel('loss',fontweight='semibold')
    plt.legend(loc='best')
    plt.title(titles[0],fontweight='heavy')
    "The second image for the Conf Loss"
    plt.sca(ax2)
    ax2.plot(x, conf_loss, '-', lw=1, color='b', label='YOLOv3-SPP')
    # ax2.plot(x, conf, '-', lw=1, color='r', label='YOLOv3-SPP')
    plt.xlabel('epoch',fontweight='semibold')
    plt.ylabel('loss',fontweight='semibold')
    plt.legend(loc='best')
    plt.title(titles[1], fontweight='heavy')
    "The third image for the Cls Loss"
    plt.sca(ax3)
    ax3.plot(x, cls_loss, '-', lw=1, color='b', label='YOLOv3-SPP')
    # ax3.plot(x, cls, '-', lw=1, color='r', label='YOLOv3-SPP')
    plt.xlabel('epoch',fontweight='semibold')
    plt.ylabel('loss',fontweight='semibold')
    plt.legend(loc='best')
    plt.title(titles[2], fontweight='heavy')
    # plt.savefig('loss.pdf')
    # plt.savefig('loss.eps')
    plt.savefig('loss.jpg', bbox_inches = 'tight', pad_inches = 0, dpi=600)
    # plt.show()

# loss_plot(0,250)


"计算检测结果的平均置信度"
def Conf(predict_result):
    budaipo = 0
    count0 = 0
    jietou = 0
    count1 = 0
    small = 0
    count2 = 0
    # bujiao = 0
    # count4 = 0
    # lianglapian = 0
    # count5 = 0
    with open(predict_result, 'r') as f:
        list = [x for x in f.read().splitlines()]
    for i in range(len(list)):
        j = 1
        predict_list = str(list[i]).split(' ')
        while len(predict_list[j]) != 0:
            if predict_list[j] == str(0):
                budaipo += float(predict_list[j + 1])
                count0 += 1
            elif predict_list[j] == str(1):
                jietou += float(predict_list[j + 1])
                count1 += 1
            elif predict_list[j] == str(2):
                small += float(predict_list[j + 1])
                count2 += 1
            # elif predict_list[j] == str(4):
            #     bujiao += float(predict_list[j + 1])
            #     count4 += 1
            # else:
            #     lianglapian += float(predict_list[j + 1])
            #     count5 += 1
            j += 2
    print("布带破的平均置信度为:", budaipo / count0)
    print("接头的平均置信度为:", jietou / count1)
    print("text_region的平均置信度为:", small / count2)
    # print("布胶的平均置信度为:", bujiao / count4)
    # print("拉片的平均置信度为:", lianglapian / count5)


"从测试图像中将text_region裁剪出来"
def imgCrop(indir):

    context_list = os.listdir(indir)
    for file in context_list:
        if file[-1] == 't':
            img = cv2.imread('data/testImages/' + file.replace('.txt','') , cv2.IMREAD_GRAYSCALE)
            with open(indir + '/' + file,'r') as f:
                context = [x for x in f.read().splitlines()]
            for i in range(len(context)):
                list = str(context[i]).split(' ')
                if list[4] == '2':
                    xmin = int(list[0])
                    ymin = int(list[1])
                    xmax = int(list[2])
                    ymax = int(list[3])
                    cropImg = img[ymin:ymax , xmin:xmax]
                    name = 'small_test/' + file.replace('.jpg.txt','_crop'+str(i)) + '.jpg'
                    cv2.imwrite(name,cropImg)


"xml转txt（类别+边界框中心坐标关于整张图的比例+宽高关于整张图的比例）,数据标记好之后调用"
"将图像的标记信息转化为YOLO格式"
def xml_to_txt1(xmlPath):

   DOMTree = xml.dom.minidom.parse(xmlPath)
   collection = DOMTree.documentElement

   objects = collection.getElementsByTagName("object")
   size = collection.getElementsByTagName("size")

   for s in size:
       width = s.getElementsByTagName("width")[0].firstChild.data
       height = s.getElementsByTagName("height")[0].firstChild.data

   for obj in objects:
       x1 = obj.getElementsByTagName('xmin')[0].firstChild.data
       y1 = obj.getElementsByTagName('ymin')[0].firstChild.data
       x2 = obj.getElementsByTagName('xmax')[0].firstChild.data
       y2 = obj.getElementsByTagName('ymax')[0].firstChild.data
       # name = obj.getElementsByTagName('name')[0].firstChild.data
       "将坐标转换为yolo格式：中心点坐标+宽高"
       x = ((int(x1) + int(x2)) / 2) / int(width)
       x = round(x, 6)
       y = ((int(y1) + int(y2)) / 2) / int(height)
       y = round(y, 6)
       w = (int(x2) - int(x1)) / int(width)
       w = round(w, 6)
       h = (int(y2) - int(y1)) / int(height)
       h = round(h, 6)
       "将yolo格式的数据写入txt文件"
       txt_path = xmlPath.replace('XML', 'Labels').replace('xml', 'txt')
       file = open(txt_path, 'a+')
       file.write('0' + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')

# xmlPath = os.listdir('data/testXML')
# for file in xmlPath:
#     xml_to_txt('data/testXML/' + file)


# kmeans_targets('data/kmeans.txt', 9, 608)
# kmean_anchors('data/kmeans.txt', 9, (544,544), thr=0.2635)
# plot_results(0,200)

# titles = ['IoU_loss', 'Conf_loss', 'Cls_loss']
# iou_loss, conf_loss, cls_loss = plot_loss(0,200)
# iou_loss = iou_loss.tolist()
# conf_loss = conf_loss.tolist()
# cls_loss = cls_loss.tolist()
#
# with open('yolo-spp-spc.txt', 'a+') as f:
#     for file in iou_loss:
#         f.write(str(file) + ',')
#     f.write('\n')
#     for file in conf_loss:
#         f.write(str(file) + ',')
#     f.write('\n')
#     for file in cls_loss:
#         f.write(str(file) + ',')
# f.close()

# img = os.listdir('data/kmeans_img')
# with open('data/kmeans.txt', 'a+') as f:
#     for file in img:
#         f.write('/media/hdc/data2/xmj/yolov3-master/data/kmeans_img/' + file + '\n')

def get_info_from_txt(txt_path, categories=None, img_size=(918, 960)): # width * height
    with open(txt_path) as file:
        lines = file.readlines()

    coordinates = []
    classes = []

    for line in lines:
        line = line.split(' ')
        classes.append(categories[int(line[0])]) if categories is not None else classes.append('object')

        ltx = int((float(line[1]) - float(line[3]) / 2) * img_size[0])
        lty = int((float(line[2]) - float(line[4]) / 2) * img_size[1])
        brx = int((float(line[1]) + float(line[3]) / 2) * img_size[0])
        bry = int((float(line[2]) + float(line[4]) / 2) * img_size[1])
        coordinates.append([ltx, lty, brx, bry])

    return coordinates, classes