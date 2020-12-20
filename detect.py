import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *
from tools import *

def detect(cfg,
           data,
           weights,
           images='data/testImages',  # input folder
           output='output',  # output folder
           fourcc='mp4v',  # video codec
           img_size=416,
           conf_thres=0.5,
           nms_thres=0.5,
           save_txt=True,
           save_images=True):
    # Initialize
    device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    opt.half = opt.half and device.type != 'cpu'  # half precision only supported on cuda
    if opt.half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if opt.webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size, half=opt.half)
    else:
        dataloader = LoadImages(images, img_size=img_size, half=opt.half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    sumTime = 0
    t0 = time.time()
    try:
        predict_result = 'data/predict_result.txt'
        with open(predict_result,'r+') as f:
            f.truncate()
        f.close()
    except:
        pass

    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)  #output/***.jpg

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]


        "将图像的原始标签添加到预测图像上"
        # image = cv2.imread(im0)
        # coordinate, label = xml_to_txt('data/testLabelsXML/' + save_path.split('/')[1].replace("bmp", "xml"))
        try:
            coordinate, label = get_info_from_txt('data/testLabels/' + save_path.split('/')[1].replace('bmp', 'txt'), classes)
            for i in range(len(coordinate)):
                c1, c2 = (int(coordinate[i][0]), int(coordinate[i][1])), (int(coordinate[i][2]), int(coordinate[i][3]))
                name = label[i]
                cv2.rectangle(im0, c1, c2, (0, 255, 0), 2)
                cv2.putText(im0, name[0], (c2[0], c2[1] - 2), 0, 0.5, [0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
        except:
            print(f'The image {save_path.split("/")[1]} doesn\'t have targets on itself.')

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen
            print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            # predict_coordinate = []
            count = 0
            result = save_path.split('/')[1]+' '
            for *xyxy, conf, cls_conf, cls in det:
                # IOU = 0
                if save_txt:  # Write to file，将检测框的左上角和右下角坐标，预测类别和分类置信度存到对应图像的txt中
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, cls_conf))

                with open(save_path + '.txt','r') as f:
                    text = [x for x in f.read().splitlines()]
                # print("The conf is:",text[count].split(' ')[5])
                result = result + text[count].split(' ')[4] + ' ' + text[count].split(' ')[5] + ' '
                # if float(text[count].split(' ')[5]) < 0.7:
                #     count += 1
                #     continue
                # else:
                #     xmlPath = "data/testLabelsXML/" + save_path.split('/')[1].split('.')[0] +'.xml'  #data/testLabelsXML/****.xml
                #     print("xmlPath is:", xmlPath)
                #     real_coordinate = xml_to_txt(xmlPath)
                #     print("real_coordinate:",real_coordinate)
                #     predict_coordinate.append([text[count].split(' ')[0],text[count].split(' ')[1],text[count].split(' ')[2],text[count].split(' ')[3]])  #x1,y1,x2,y2
                #     print("predict_coordinate:",predict_coordinate)
                #     if len(real_coordinate) == 1:
                #         IOU = calIOU_V2(predict_coordinate[count],real_coordinate[count])
                    # if len(real_coordinate) == 1:
                    #     IOU = calIOU_V2(predict_coordinate[count],real_coordinate[count])
                    #     print("The IOU = ",IOU)
                    # else:
                    #     if count == 0:
                    #         IOU = calIOU_V2(predict_coordinate[count], real_coordinate[count + 1])
                    #         print("The IOU = ",IOU)
                    #     else:
                    #         IOU = calIOU_V2(predict_coordinate[count], real_coordinate[count - 1])
                    #         print("The IOU = ",IOU)
                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], cls_conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                count += 1
        # with open(predict_result,'a+') as f:
        #     f.write(result+'\n')
        print('Done. (%.3fs)' % (time.time() - t))
        sumTime += time.time() - t

        if opt.webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                vid_writer.write(im0)

    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)

    #print('Done. (%.3fs)' % (time.time() - t0))
    "计算测试所用时长"
    print('All Images Done in (%.3fs)' % sumTime)
    f.close()




if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    exp_num = input('Experiment Number: ')
    output = f"output/{exp_num}_{time.strftime('%Y年%m月%d日%H时%M分%S秒', time.localtime())}"

    # weights / yolov3 - spp.weights
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/cfg/yolov3-spp-1cls.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/testImages', help='path to images')
    # parser.add_argument('--images', type=str, default='train/validImages', help='path to images')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default='output', help='specifies the output path for images and videos')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--webcam', action='store_true', help='use webcam')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.cfg,
               opt.data,
               opt.weights,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               fourcc=opt.fourcc,
               output=opt.output)

    import shutil, os
    results = os.listdir('output')
    os.makedirs(output)

    for result in results:
        shutil.move(f'output/{result}', f'{output}/{result}')