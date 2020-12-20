from utils.parse_config import *
from utils.utils import *
from pathlib import Path

import torch.nn.functional as F

ONNX_EXPORT = False

#根据cfg文件中的模块用pytorch创建相关层，例如Conv2d等
#在调用该函数时，对于Conv层和BN层会随机初始化。在train.py中模型加载之后，backbone会使用预训练的权重文件darknet53.conv.74
def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)   #只保留cfg文件中的[net]模块，该模块定义网络的超参数
    output_filters = [int(hyperparams['channels'])]   #output_filters = 3,是输入图像的通道数
    module_list = nn.ModuleList()      #nn.ModuleList()可以定义一个存储网络结构信息的列表，例如卷积大小，步长等，它是无序列表
    yolo_index = -1

    for mdef in module_defs:
        modules = nn.Sequential()      #nn.Sequential()定义一个按顺序级联的网络架构，可以实现forward前向传播

        if mdef['type'] == 'convolutional':         #如果该模块为卷积层
            bn = int(mdef['batch_normalize'])       #存储批量归一化参数
            filters = int(mdef['filters'])          #存储卷积核数量
            kernel_size = int(mdef['size'])         #存储卷积核大小
            if len(mdef) == 8:
                dilation = int(mdef['dilation'])
                dilated_kernal_size = kernel_size + 2*(dilation - 1)
                pad = int((dilated_kernal_size - 1) // 2)
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=int(mdef['stride']),
                                                   padding=pad,
                                                   dilation=dilation,
                                                   bias=not bn))
            else:
                pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0    #存储卷积核的填充，3*3卷积的pad为1； 1*1卷积的pad为0
                #在modules中创建一个卷积层，参数使用cfg文件中的卷积参数
                #in_channels：卷积层的输入图像的通道数； out_channels：卷积层的输出图像的通道数；bias = False
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=kernel_size,
                                                       stride=int(mdef['stride']),
                                                       padding=pad,
                                                       bias=not bn))
            #如果bn=1，添加一个批量归一化层
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            #添加ReLU损失函数层
            if mdef['activation'] == 'leaky':
                # modules.add_module('activation', nn.PReLU(num_parameters=filters, init=0.1))
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'ReLU':
                modules.add_module('activation', nn.ReLU(inplace=True))
            else:
                pass

        elif mdef['type'] == 'RL':
            modules = nn.ReLU(inplace=True)

        elif mdef['type'] == 'maxpool':             #如果该模块为池化层（SPP中的maxpool）
            kernel_size = int(mdef['size'])         #存储池化操作的卷积核大小
            stride = int(mdef['stride'])            #存储池化操作卷积核的步长
            #添加一个池化层,SPP得到的子特征图大小均为13*13
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            if kernel_size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':            #如果该模块为上采样
            #添加一个上采样层（双线性插值），yolov3-spp均为2倍上采样
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

        elif mdef['type'] == 'route':           #如果该模块为连接层      # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]        #获取要连接使用的特征图所在的层数，数量可能为1,2,4
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])      #计算特征图拼接后得到的特征图的通道数
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(scale_factor=1/float(mdef[i+1]['stride']), mode='nearest')  # reorg3d

        elif mdef['type'] == 'shortcut':        #如果该模块为跳转连接层       # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]         #计算残差块输出的特征图的通道数

        elif mdef['type'] == 'integrate' or mdef['type'] == 'add':
            filters = output_filters[-1]  #256

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate dimensions 2 and 3 to cat with prior layer
            pass

        elif mdef['type'] == 'yolo':            #如果该模块为yolo检测层
            yolo_index += 1                     #记录当前为第几个yolo检测层，起始yolo_index=-1
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask，获取使用的anchor编号
            a = [float(x) for x in mdef['anchors'].split(',')]  # anchor
            a = [(a[i], a[i + 1]) for i in range(0, len(a), 2)] #a是一个列表，依次存储每个anchor的尺寸
            #添加yolo检测层
            modules = YOLOLayer(anchors=[a[i] for i in mask],  # anchor list，获取要使用的3个锚框大小
                                nc=int(mdef['classes']),  # number of classes，获取分类数目
                                img_size=hyperparams['height'],  # 416，获取图像大小
                                yolo_index=yolo_index)  # 0, 1 or 2，获取当前yolo检测层的编号
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index):
        super(YOLOLayer, self).__init__()

        self.anchors = torch.Tensor(anchors)        #将不同大小的anchor由numpy转化成tensor
        self.na = len(anchors)  # number of anchors (3)，使用的锚框数量
        self.nc = nc  # number of classes (80)，分类的类别数目
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints

        if ONNX_EXPORT:  # grids must be computed in __init__
            stride = [32, 16, 8][yolo_index]  # stride of this layer
            nx = int(img_size[1] / stride)  # number x grid points
            ny = int(img_size[0] / stride)  # number y grid points
            create_grids(self, max(img_size), (nx, ny))

    def forward(self, p, img_size, var=None):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, ny, nx = p.shape[0], p.shape[-2], p.shape[-1]
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device, p.dtype)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.nc + 5, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Constants CAN NOT BE BROADCAST, ensure correct shape!
            ngu = self.ng.repeat((1, self.na * self.nx * self.ny, 1))
            grid_xy = self.grid_xy.repeat((1, self.na, 1, 1, 1)).view((1, -1, 2))
            anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view((1, -1, 2)) / ngu

            # p = p.view(-1, 5 + self.nc)
            # xy = torch.sigmoid(p[..., 0:2]) + grid_xy[0]  # x, y
            # wh = torch.exp(p[..., 2:4]) * anchor_wh[0]  # width, height
            # p_conf = torch.sigmoid(p[:, 4:5])  # Conf
            # p_cls = F.softmax(p[:, 5:85], 1) * p_conf  # SSD-like conf
            # return torch.cat((xy / ngu[0], wh, p_conf, p_cls), 1).t()

            p = p.view(1, -1, 5 + self.nc)
            xy = torch.sigmoid(p[..., 0:2]) + grid_xy  # x, y
            wh = torch.exp(p[..., 2:4]) * anchor_wh  # width, height
            p_conf = torch.sigmoid(p[..., 4:5])  # Conf
            p_cls = p[..., 5:5 + self.nc]
            # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
            # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
            p_cls = torch.exp(p_cls).permute((2, 1, 0))
            p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
            p_cls = p_cls.permute(2, 1, 0)
            return torch.cat((xy / ngu, wh, p_conf, p_cls), 2).squeeze().t()

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride

            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
            # io[..., 5:] = F.softmax(io[..., 5:], dim=4)  # p_cls

            if self.nc == 1:
                io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 5 + self.nc), p

#构造指定网络架构并实现前向传播，yolov3-spp
class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg, img_size=(416, 416)):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)   #self.module_defs存放cfg文件中的每个模块以及内部参数，它是一个列表
        self.module_defs[0]['cfg'] = cfg          #给module_defs的第一行添加一个键值对 cfg：yolov3-spp
        self.module_defs[0]['height'] = img_size  #设置图像高度
        # 按照cfg文件构造网络架构，hyperparams存储网络训练的相关参数，module_list存储模块（卷积，上采样，route...）
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = get_yolo_layers(self)   #获取yolo检测层所在的层数

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, x, var=None):
        img_size = max(x.shape[-2:])
        layer_outputs = []
        output = []

        #遍历module_defs（cfg文件中每个模块的内容）和module_list（将对应的模块用pytorch实现），两者为一一对应关系
        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            # 获取cfg文件中存在的模块类型，yolov3-spp中有net，convolutional，shortcut，route，maxpool，upsample，yolo
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool', 'RL']:
                x = module(x)
            elif mtype == 'route':
                #layer_i存储要进行拼接的两个特征图所在的层数
                layer_i = [int(x) for x in mdef['layers'].split(',')]
                #如果只有一个特征图，则不进行拼接，只是拿来用
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                #如果不止一个特征图，那就进行拼接
                else:
                    try:
                        #将两层特征图在通道上拼接，Concat方式
                        x = torch.cat([layer_outputs[i] for i in layer_i], 1)
                    except:  # apply stride 2 for darknet reorg layer
                        #如果两层特征图尺寸不同，则将前面浅层特征图进行降维，再进行拼接
                        layer_outputs[layer_i[1]] = F.interpolate(layer_outputs[layer_i[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layer_i], 1)
                    # print(''), [print(layer_outputs[i].shape) for i in layer_i], print(x.shape)
            elif mtype == 'shortcut':
                # 如果是残差连接，则将残差块的输入X和残差项F(X)直接逐元素相加（横向拼接），Add方式
                layer_i = int(mdef['from'])
                if len(mdef) == 4:
                    x = layer_outputs[-1] * layer_outputs[layer_i]
                else:
                    x = layer_outputs[-1] + layer_outputs[layer_i]


            elif mtype == 'integrate' or mtype == 'add':
                layer_i = [int(x) for x in mdef['layers'].split(',')]
                x = layer_outputs[layer_i[0]] + layer_outputs[layer_i[1]]

            elif mtype == 'yolo':
                x = module(x, img_size)
                output.append(x)
            layer_outputs.append(x)

        if self.training:
            return output
        elif ONNX_EXPORT:
            output = torch.cat(output, 1)  # cat 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            nc = self.module_list[self.yolo_layers[0]].nc  # number of classes
            return output[5:5 + nc].t(), output[:4].t()  # ONNX scores, boxes
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            for i, b in enumerate(a):
                if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                    # fuse this bn layer with the previous conv2d layer
                    conv = a[i - 1]
                    fused = torch_utils.fuse_conv_and_bn(conv, b)
                    a = nn.Sequential(fused, *list(a.children())[i + 1:])
                    break
            fused_list.append(a)
        self.module_list = fused_list
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers


#获取yolo检测层所在的层数
def get_yolo_layers(model):
    #enumerate函数将一个列表中的元素逐个赋予下标，对于module_defs，它是从[net]开始计数，且下标从0开始计数
    # [82, 94, 106] for yolov3； [90, 102, 114] for yolov3-spp； [90, 102, 114, 126] for yolov3-spp-xmj
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng  # x and y grid size
    self.img_size = img_size
    self.stride = img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    self.anchor_vec = self.anchors.to(device) / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device).type(type)
    self.ng = torch.Tensor(ng).to(device)
    self.nx = nx
    self.ny = ny

#model(yolov3-spp.cfg)，weights = weights/darknet53.conv.74
def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    file = Path(weights).name       #file = darknet53.conv.74

    # Try to download weights if not available locally
    msg = weights + ' missing, download from https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI'
    if not os.path.isfile(weights):
        try:
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            os.system('curl -f ' + url + ' -o ' + weights)
        except IOError:
            print(msg)
    assert os.path.exists(weights), msg  # download missing weights from Google Drive

    # Establish cutoffs
    if file == 'darknet53.conv.74':
        print('yolov3-spp')
        cutoff = 75      #yolov3的网络模型最后一层卷积层为第75层（从1开始计数）；如果使用LSN，则截止到76
    elif file == 'yolov3-tiny.conv.15':
        print('yolov3-tiny')
        cutoff = 15
    elif file == 'yolov4.conv.137':
        print('yolov4')
        cutoff = 105

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    ptr = 0
    "yolov3在训练阶段，使用yolov3基础网络，即darknet53（cfg文件中的前75层），所以加载的是darknet53.conv.74" \
    "如果使用了辅助网络LSN，则预训练权重从1到cutoff"
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv_layer = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w
    "yolov4"
    # for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
    #     if mdef['type'] == 'convolutional':
    #         conv = module[0]
    #         if mdef['batch_normalize']:
    #             # Load BN bias, weights, running mean and running variance
    #             bn = module[1]
    #             nb = bn.bias.numel()  # number of biases
    #             # Bias
    #             bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
    #             ptr += nb
    #             # Weight
    #             bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
    #             ptr += nb
    #             # Running Mean
    #             bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
    #             ptr += nb
    #             # Running Var
    #             bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
    #             ptr += nb
    #         else:
    #             # Load conv. bias
    #             nb = conv.bias.numel()
    #             conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
    #             conv.bias.data.copy_(conv_b)
    #             ptr += nb
    #         # Load conv. weights
    #         nw = conv.weight.numel()  # number of weights
    #         conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
    #         ptr += nw

    return cutoff


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')
