import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor, Compose, ConvertImageDtype, RandomHorizontalFlip
from utils import ProblemClasses
from torchvision.transforms import functional as F

torch.manual_seed(17)

class CardDatsSet(DataLoader):
    def __init__(self, root, mode="train"):
        super().__init__(self)
        self.mode = mode
        self.root = os.path.join(root, mode)
        self.imgs = list(sorted(os.listdir(self.root)))
        # filename,width,height,class,xmin,ymin,xmax,ymax
        self.df = pd.read_csv(os.path.join(root, mode + "_labels.csv"))
        #self.df["boxes"] = self.df[["xmin", "ymin", "xmax", "ymax"]].apply(list, axis=1)
        self.transform = self.make_transform()

    def make_transform(self):
        transforms = []
        transforms.append(PILToTensor())
        transforms.append(ConvertImageDtype(torch.float))
        return Compose(transforms)

    def __getitem__(self, item):

        flip_augmentation = False if self.mode == "test" or np.random.random_sample(1) < 0.5 else True
        flip_augmentation2 = False if self.mode == "test" or np.random.random_sample(1) < 0.5 else True
        img_path = os.path.join(self.root, self.imgs[item])
        img = Image.open(img_path)
        if flip_augmentation:
            img = F.vflip(img)
        if flip_augmentation2:
            img = F.hflip(img)

        d = self.df[self.df.filename == self.imgs[item]]
        boxes = []
        labels = []
        for n in d.index:
            if flip_augmentation and not(flip_augmentation2):
                #Vertical flip
                box = [d.loc[n].xmin, img.height - d.loc[n].ymax, d.loc[n].xmax, img.height - d.loc[n].ymin]
            elif not(flip_augmentation) and flip_augmentation2:
                #Horisontal flip
                box = [img.width - d.loc[n].xmax, d.loc[n].ymin, img.width - d.loc[n].xmin, d.loc[n].ymax ]
            elif flip_augmentation and flip_augmentation2:
                #both vertical and horizontal
                box = [img.width - d.loc[n].xmax, img.height - d.loc[n].ymax, img.width - d.loc[n].xmin, img.height - d.loc[n].ymin]
            else:
                box = [d.loc[n].xmin, d.loc[n].ymin, d.loc[n].xmax, d.loc[n].ymax]
            boxes.append(box)
            labels.append(ProblemClasses.class_num[d.loc[n]["class"]])

        target = {}
        target["boxes"] = torch.Tensor(boxes).to(torch.float)
        target["labels"] = torch.Tensor(labels).to(torch.int64)

        return self.transform(img), target, (flip_augmentation, flip_augmentation2),d.filename

    def __len__(self):
        return len(self.imgs)


def get_boxes_rectangle(tg):
    boxes = []
    for box in tg["boxes"]:
        boxes.append(Rectangle((box[0].detach().numpy(), box[1].detach().numpy()),
                               (box[2] - box[0]).detach().numpy(), (box[3] - box[1]).detach().numpy()))

    pc = PatchCollection(boxes, edgecolors="b", facecolors="none", alpha=1)
    return pc


if __name__ == "__main__":
    ds = CardDatsSet(os.path.join(os.getcwd(), "images"), "train")
    print(len(ds))
    fig, ax = plt.subplots(2, 5, figsize=(10, 4), layout="constrained")
    img_index_change=200
    for i in range(5):

        data1 = ds[i+img_index_change]

        img = data1[0]
        ax[0][i].imshow(img.permute(1, 2, 0))
        class_num = data1[1]["labels"][0].detach().numpy()
        ax[0][i].set_title(str(data1[3].values[0])+":"+str(data1[2])+" flip;"+str(class_num) + "-" + str(ProblemClasses.class_name[int(class_num)]))
        # ax[0][i].set_axis_off()
        ax[0][i].add_collection(get_boxes_rectangle(data1[1]))

        data2 = ds[i+5+img_index_change]
        img = data2[0]
        ax[1][i].imshow(img.permute(1, 2, 0))
        class_num = data2[1]["labels"][0].detach().numpy()
        ax[1][i].set_title(str(data1[3].values[0])+":"+str(data1[2])+" flip;"+str(class_num) + "-" + str(ProblemClasses.class_name[int(class_num)]))
        ax[1][i].set_axis_off()
        ax[1][i].add_collection(get_boxes_rectangle(data2[1]))
        plt.axis("off")

    plt.show()

    print("end")
