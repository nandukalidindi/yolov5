# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import os
import argparse
import sys
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel

from torchvision import transforms
from effdet import create_model
from timm.models.layers import set_layer_config

torch.backends.cudnn.benchmark = True

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, non_max_suppression_fast, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

UNDETECTED_COUNT_THRESHOLD = 600
MAX_DETECTION_FRAME_COUNT = 1800
# UNDETECTED_COUNT_THRESHOLD = 60
# MAX_DETECTION_FRAME_COUNT = 180


def load_sy4(weights, device, half):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    return model


def load_effdet(weights, imgsz, device):
    with set_layer_config(scriptable=False):
        model = create_model(
            'tf_efficientdet_d2',
            bench_task='predict',
            num_classes=2,
            pretrained=True,
            checkpoint_path=weights,
            image_size=imgsz,
        ).to(device)
    return model


def separate_predictions(pred):
    pred = pred[0]
    pred_fire, pred_fire_prob = [], 0
    pred_smoke, pred_smoke_prob = [], 0
    for p in pred:
        if p[-1] == 0:
            pred_fire.append(p.unsqueeze(0))
        else:
            pred_smoke.append(p.unsqueeze(0))

    if len(pred_fire) > 0:
        pred_fire = torch.cat(pred_fire)
        pred_fire_prob = torch.sum(pred_fire).item() / len(pred_fire)

    if len(pred_smoke) > 0:
        pred_smoke = torch.cat(pred_smoke)
        pred_smoke_prob = torch.sum(pred_smoke).item() / len(pred_smoke)

    return pred_fire, pred_fire_prob, pred_smoke, pred_smoke_prob


def get_predictions(
    img, model_sy4, model_effdet, conf_thres, iou_thres, classes,
    agnostic_nms, max_det, augment, imgsz, resize_transform
):
    # Inference - Scaled Yolo v4
    pred_sy4 = model_sy4(img, augment=augment)[0]
    pred_sy4 = non_max_suppression(pred_sy4, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Inference - Efficientdet D2
    if list(img.shape[-2:]) != imgsz:
        img = resize_transform(img)
    pred_effdet = model_effdet(img)[0].cpu()
    pred_effdet[:, 5] = pred_effdet[:, 5] - 1
    pred_effdet = [non_max_suppression_fast(pred_effdet, iou_thres, conf_thres)]

    # Separate fire and smoke predictions
    pred_sy4_fire, pred_sy4_fire_prob, pred_sy4_smoke, pred_sy4_smoke_prob = separate_predictions(pred_sy4)
    pred_effdet_fire, pred_effdet_fire_prob, pred_effdet_smoke, pred_effdet_smoke_prob = separate_predictions(pred_effdet)

    # Get the final predictions
    pred_fire = pred_sy4_fire if pred_sy4_fire_prob > pred_effdet_fire_prob else pred_effdet_fire
    pred_smoke = pred_sy4_smoke if pred_sy4_smoke_prob > pred_effdet_smoke_prob else pred_effdet_smoke
    pred = []
    if len(pred_fire) > 0:
        pred.append(pred_fire.cpu())
    if len(pred_smoke) > 0:
        pred.append(pred_smoke.cpu())

    if len(pred) > 0:
        pred = torch.cat(pred)

    # FIXME: scale the algo to batch size

    # pred will be [tensor(shape=(n, 6), info=(n is num bboxes, 6 is - coords (4), prob, class_idx))]
    return [pred]


@torch.no_grad()
def run(weight_sy4='scaled_yolov4.pt',  # model.pt path(s)
        weight_effdet='efficientdet_d2.pth.tar',
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        save_stream_detection_image='stream/detect',
        save_stream_detection_video='stream/detect',
        verbose=False,
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    stride, names = 64, ['fire', 'smoke']  # assign defaults

    model_sy4 = load_sy4(weight_sy4, device, half)
    model_effdet = load_effdet(weight_effdet, imgsz, device)

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    resize_transform = transforms.Compose([transforms.Resize(imgsz)])  # for resizing images

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, verbose=verbose)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if device.type != 'cpu':
        model_sy4(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model_sy4.parameters())))  # run once
        model_effdet(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model_effdet.parameters())))  # run once
    t0 = time.time()

    # Video Snippets
    build_video = False
    fire_smoke_detected = False
    current_video_frame_count = 0
    undetected_count = 0

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred = get_predictions(
            img, model_sy4, model_effdet, conf_thres, iou_thres, classes,
            agnostic_nms, max_det, augment, imgsz, resize_transform
        )

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    if names[int(c)] == 'fire' or names[int(c)] == 'smoke':
                        fire_smoke_detected = True
                        build_video = True
                        undetected_count = 0

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_width=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            else:
                fire_smoke_detected = False
                undetected_count = undetected_count + 1

            # Print time (inference + NMS)
            if verbose:
                print(f'{s}Done.')

            if fire_smoke_detected:
                cv2.imwrite(f'{save_stream_detection_image}/fire-detection.jpg', im0)

            if build_video and undetected_count >= UNDETECTED_COUNT_THRESHOLD:
                current_video_frame_count = 0
                build_video = False
                vid_writer[i].release()

            if build_video:
                current_video_frame_count = current_video_frame_count + 1
                if save_img:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

                if current_video_frame_count >= MAX_DETECTION_FRAME_COUNT:
                    current_video_frame_count = 0
                    vid_path[i] = None
                    vid_writer[i].release()
                    shutil.copy(save_path, save_stream_detection_video)
                    print("===============================")
                    print("===============================")
                    print("SENDING VIDEO!!!")
                    print("===============================")
                    print("===============================")


            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_sy4', type=str, default=os.path.join(BASE_DIR, 'weights/scaled_yolov4.pt'), help='scaled yolov4 path')
    parser.add_argument('--weight_effdet', type=str, default=os.path.join(BASE_DIR, 'weights/efficientdet_d2.pth.tar'), help='efficientdet d2 path')
    parser.add_argument('--source', type=str, default=os.path.join(BASE_DIR, 'data/video.mp4'), help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--save-stream-detection-image', default='stream/detect', help='Directory to save the stream snapshot for each frame')
    parser.add_argument('--save-stream-detection-video', default='stream/detect', help='Directory to save the video containing the detections across the specified time')
    parser.add_argument('--verbose', action='store_true', help='display intermediate logs')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
