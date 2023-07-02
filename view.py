import os
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from data import CardDatsSet
from model_fasterRCNN import create_model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from utils import collate_fn, ProblemClasses


model_name = "fastrrcnn4"
num_classes = 7
num_workers = 0
batch_size = 1
model = create_model(num_classes, pretrained=True)
model.load_state_dict(torch.load(model_name + ".net"))
model.eval()

ds = CardDatsSet(os.path.join(os.getcwd(), "images"), mode="test")
test_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
# im, tr, _, _ = next(iter(test_dataloader))
find_num = 33
for i, d in enumerate(test_dataloader):
    if i == find_num:
        im = d[0]
        tr = d[1]
        flip_flag = d[2]
        file_name = d[3][0].values[0]
        break


predict = model(im)

iou_threshold = 0.1
threshold = 0.35
print(max( predict[0]["scores"]))
ind = nms(predict[0]["boxes"], predict[0]["scores"], iou_threshold).detach().cpu().numpy()
print(list(zip(predict[0]["scores"][ind], predict[0]["labels"][ind])))

boxes = []
labels = []
labels_xy = []
for i, box in enumerate(predict[0]["boxes"][ind]):
    if predict[0]["scores"][i] > threshold:
        boxes.append(Rectangle((box[0].detach().cpu().numpy(), box[1].detach().cpu().numpy()),
                               (box[2] - box[0]).detach().cpu().numpy(), (box[3] - box[1]).detach().cpu().numpy()))
        labels.append(ProblemClasses.class_name[int(predict[0]["labels"][ind][i].detach().numpy())])
        labels_xy.append([box[0].detach().cpu().numpy(), box[1].detach().cpu().numpy()])
pc = PatchCollection(boxes, facecolors="none", edgecolors="g", alpha=0.8)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(im[0].permute(1, 2, 0))

ax.set_title(str(file_name) + ";" + str(flip_flag) + " flip")
ax.add_collection(pc)
for i in range(len(labels)):
    xy = labels_xy[i]
    l = labels[i]
    ax.text(xy[0]+6, xy[1]-11, l, bbox={"edgecolor":"g", "facecolor":"none"}, color="w")

model.train()
loss_dict = model(im, tr)
loss = sum(loss for loss in loss_dict.values())
print("loss:",loss)
plt.show()



