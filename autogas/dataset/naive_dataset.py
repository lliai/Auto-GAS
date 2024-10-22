import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NaiveImageDataset(Dataset):
    """ Naive Image Dataset"""

    def __init__(self, img_folder, trans=None):
        self.img_folder = img_folder
        self.img_names = os.listdir(self.img_folder)

        if trans is not None:
            self.trans = trans
        else:
            self.trans = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path)
        # import pdb; pdb.set_trace()
        img = self.trans(img.convert('RGB'))
        return img
