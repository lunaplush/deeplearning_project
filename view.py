import os
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from data import CardDatsSet
from model_fasterRCNN import create_model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from utils import collate_fn

model_name = "fastrrcnn1"
num_classes = 6
num_workers = 4
batch_size = 1
model = create_model(num_classes)
model.load_state_dict(torch.load(model_name + ".net"))
model.eval()

ds = CardDatsSet(os.path.join(os.getcwd(), "images"), mode="test")
test_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
im, tr = next(iter(test_dataloader))
predict = model(im)

iou_threshold = 0.1
threshold = 0.8
ind = nms(predict[0]["boxes"], predict[0]["scores"], iou_threshold).detach().cpu().numpy()
boxes = []
for i, box in enumerate(predict[0]["boxes"][ind]):
    if predict[0]["scores"][i] > threshold:
        boxes.append(Rectangle((box[0].detach().cpu().numpy(), box[1].detach().cpu().numpy()),
                               (box[2]-box[0]).detach().cpu().numpy(), (box[3]-box[1]).detach().cpu().numpy()))
pc = PatchCollection(boxes, facecolors="none", edgecolors="g", alpha=0.8)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(im[0].permute(1, 2, 0))
ax.add_collection(pc)
plt.show()
