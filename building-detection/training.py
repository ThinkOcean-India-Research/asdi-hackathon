import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# import torchvision.transforms as T
# from torchvision import utils

import utils.transforms as T
from utils.engine import train_one_epoch, evaluate
import utils.utils as utils
# print(torch.cuda.is_available())

# generated in create_masks.py
SATELLITE_IMAGES_PATH = 'data/AOI_1_rio/imgs'
MASKS_PATH = 'data/AOI_1_rio/masks_nonzero'
IMG_SIZE = (406, 438)


class BuildingFootprintDataset(Dataset):
    def __init__(self, transforms=None) -> None:
        super().__init__()
        self.transforms = transforms
        # self.imgs = sorted(os.listdir(SATELLITE_IMAGES_PATH))
        self.filenames = sorted(os.listdir(MASKS_PATH))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(SATELLITE_IMAGES_PATH,
                         self.filenames[idx])).convert("RGB").resize(IMG_SIZE)
        # print(np.array(img).shape)
        mask = Image.open(os.path.join(
            MASKS_PATH, self.filenames[idx])).resize(IMG_SIZE)
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # to ensure nonzero bounding boxes
            if xmax == xmin:
                xmax += 0.1
            if ymax == ymin:
                ymax += 0.1

            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class, the building. Has to be one as zero refers to the background
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * \
            (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else 0

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = BuildingFootprintDataset(get_transform(train=True))

# img, target = dataset[0]
# print(img.shape, target['masks'].shape)
# img, target = dataset[1200]
# print(img.shape, target['masks'].shape)
# print(dataset[1200])


data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn
)


images, targets = next(iter(data_loader))
images = list(image for image in images)
# print(images[0].shape)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)   # Returns losses and detections
# # For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions
