import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

# print(torch.cuda.is_available())

# generated in create_masks.py
SATELLITE_IMAGES_PATH = 'data/AOI_1_rio/imgs'
MASKS_PATH = 'data/AOI_1_rio/masks'


class BuildingFootprintDataset(Dataset):
    def __init__(self, transforms=None) -> None:
        super().__init__()
        self.transforms = transforms
        self.imgs = sorted(os.listdir(SATELLITE_IMAGES_PATH))
        self.masks = sorted(os.listdir(MASKS_PATH))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index):
        img = Image.open(os.path.join(SATELLITE_IMAGES_PATH,
                         self.imgs[index])).convert("RGB")
        print(np.array(img).shape)
        mask = Image.open(os.path.join(MASKS_PATH, self.masks[index]))
        mask = np.array(mask)
        # a tensor of masks. We have only one per image in this case
        masks = torch.as_tensor(mask, dtype=torch.uint8).unsqueeze(0)
        print(masks.shape)
        return img, masks


dataset = BuildingFootprintDataset()
print(dataset[4])
