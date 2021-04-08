import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
import streamlit as st
from pathlib import Path
from GD_download import download_file_from_google_drive

def get_obj_det_model(local=False):
    # download or load the model from disk
    if local:
        torch.hub.set_dir('.')
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=800).eval()

def crop_beers(image, model, threshold, GPU=True):
    #  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    boxes, classes, labels, preds = find_bottles(image, model, detection_threshold=threshold, GPU=GPU)
    if len(boxes) > 0:
        image_cropped = image.crop(tuple(boxes[0]))  # crop image: select only relevant part of pic
        # todo correct als er 2 boxes zijn (nu pak degene met hoogste pred, boxes is al gesorteerd op pred)
    else:
        image_cropped = image
    # image = draw_boxes(boxes, classes, labels, image)
    return image_cropped, len(boxes)


def find_bottles(image, model, detection_threshold=0.8, GPU=True):
    coco_names = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    device = torch.device('cuda' if GPU else 'cpu')
    model.to(device)

    # define the torchvision image transforms
    transform = transforms.Compose([
        transforms.ToTensor()])

    # transform the image to tensor
    image = transform(image)
    image = image.unsqueeze(0)  # add a batch dimension
    if GPU:
        image = image.cuda()
    outputs = model(image) # get the predictions on the image

    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]

    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # which labels are bottles?
    bottles = []
    for i in pred_classes:
        bottles.append(i == 'bottle')
    bottles = np.array(bottles)

    boxes = pred_bboxes.astype(np.int32)

    # get boxes above the threshold score & which are bottles
    relevant_outputs = (pred_scores >= detection_threshold) & bottles

    if sum(relevant_outputs) > 0:
        return boxes[relevant_outputs], \
               list(np.array(pred_classes)[relevant_outputs]), \
               outputs[0]['labels'][relevant_outputs], \
               pred_scores[relevant_outputs]
    else:
        return[[], [], [], []]

#@st.cache(show_spinner=False)
def get_obj_det_model_Drive():
    save_dest = Path('checkpoints')
    save_dest.mkdir(exist_ok=True)
    f_checkpoint = Path("checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")
    #if not f_checkpoint.exists():
    download_file_from_google_drive('1mJ_rRsGYDplNab-gkBeObEBspWGVNfg6', f_checkpoint)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=800).eval()
    return model

