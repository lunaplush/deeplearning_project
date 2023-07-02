import os
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from data import CardDatsSet
from model_fasterRCNN import create_model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from utils import collate_fn, ProblemClasses
from pprint import pprint
import adjust
import gc
model_name = "fastrrcnn4"
num_classes = ProblemClasses.num_classes
num_workers = adjust.num_workers
batch_size = 1
model = create_model(num_classes, pretrained=False)
model.load_state_dict(torch.load(model_name + ".net"))
iou_threshold = 0.1
threshold = 0.35
model.eval()

ds = CardDatsSet(os.path.join(os.getcwd(), "images"), mode="test")
N = len(ds)
test_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
# im, tr, _, _ = next(iter(test_dataloader))

metric = MeanAveragePrecision()
score_map_all = 0
score_map_nms_all = 0

for i, d in enumerate(test_dataloader):
    im = d[0]
    tr = d[1]
    flip_flag = d[2]
    file_name = d[3][0].values[0]
    predict = model(im)
    metric.update(predict, tr)
    scores = metric.compute()
    score_map_all += scores["map"] / N
    ind = nms(predict[0]["boxes"], predict[0]["scores"], iou_threshold).detach().cpu().numpy()
    predict_nms = [{}]
    for k in predict[0].keys():
        predict_nms[0][k] = predict[0][k][ind]
    metric.update(predict_nms, tr)
    scores_nms = metric.compute()
    score_map_nms_all += scores_nms["map"] / N
    gc.collect()
    print("num", i)
print("SCORES", score_map_all, "with NMS", score_map_nms_all)

#
#
#
# print(max( predict[0]["scores"]))
#
# print(list(zip(predict[0]["scores"][ind], predict[0]["labels"][ind])))
#
# boxes = []
# labels = []
# labels_xy = []
# for i, box in enumerate():
#     if predict[0]["scores"][i] > threshold:
#         boxes.append(Rectangle((box[0].detach().cpu().numpy(), box[1].detach().cpu().numpy()),
#                                (box[2] - box[0]).detach().cpu().numpy(), (box[3] - box[1]).detach().cpu().numpy()))
#         labels.append(ProblemClasses.class_name[int(predict[0]["labels"][ind][i].detach().numpy())])
#         labels_xy.append([box[0].detach().cpu().numpy(), box[1].detach().cpu().numpy()])
# pc = PatchCollection(boxes, facecolors="none", edgecolors="g", alpha=0.8)
# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.imshow(im[0].permute(1, 2, 0))
#
# ax.set_title(str(file_name) + ";" + str(flip_flag) + " flip")
# ax.add_collection(pc)
# for i in range(len(labels)):
#     xy = labels_xy[i]
#     l = labels[i]
#     ax.text(xy[0]+6, xy[1]-11, l, bbox={"edgecolor":"g", "facecolor":"none"}, color="w")
#
# model.train()
# loss_dict = model(im, tr)
# loss = sum(loss for loss in loss_dict.values())
# print("loss:",loss)
# plt.show()
#
#
#
