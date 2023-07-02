import os
import pickle
import pandas as pd
import time
import torch
import adjust

from data import CardDatsSet
from model_fasterRCNN import create_model
from torch.utils.data import DataLoader
from utils import collate_fn, ProblemClasses

if adjust.MAC:
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device is", DEVICE)

FIRST_STEP = adjust.FIRST_STEP
model_name = adjust.model_name
batch_size = adjust.batch_size
max_epochs = adjust.max_epochs
num_workers = adjust.num_workers

dataset_train = CardDatsSet(os.path.join(os.getcwd(), "images"))
dataset_test = CardDatsSet(os.path.join(os.getcwd(), "images"), mode="test")
train_data = DataLoader(dataset_train,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=collate_fn)

test_data = DataLoader(dataset_test,
                       batch_size=batch_size,
                       shuffle=False,
                       num_workers=num_workers,
                       collate_fn=collate_fn)


def train(model, start_epoch, max_epochs, optim, sheduler):
    time_start = time.time()

    model.to(DEVICE)
    losses_train = []
    losses_test = []

    for epoch in range(max_epochs):

        scores = []
        loss_train = 0
        loss_test = 0
        #
        #
        #     model.train()
        #     for imgs, targets, _, _ in train_data:
        #
        #         optim.zero_grad()
        #         imgs = list(img.to(DEVICE) for img in imgs)
        #         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        #         loss_dict = model(imgs, targets)
        #         loss = sum(loss for loss in loss_dict.values())
        #         loss_train += loss.item()
        #         loss.backward()
        #         optim.step()

        for imgs, targets, _, _ in test_data:
            with torch.no_grad():
                loss_dict = model(imgs, targets)
                print("loss_dict", loss_dict)
            loss = sum(loss for loss in loss_dict.values())
            loss_test += loss.item()


        sheduler.step()
        loss_train = loss_train / len(train_data.dataset)
        loss_test = loss_test / len(test_data.dataset)
        losses_train.append(loss_train)
        losses_test.append(loss_test)
        print("Epoch {}/{}. Loss {}".format(start_epoch + epoch, start_epoch + max_epochs, loss_train))


        print(time.time() - time_start, " c ; current loss on ", i, " : ", loss_train)
    analysis_data = {"losses_train": losses_train, "losses_test": losses_test}
    return analysis_data

model = create_model(ProblemClasses.num_classes, pretrained=FIRST_STEP)
params = [p for p in model.parameters() if p.requires_grad]
if adjust.optimizer_dict["type"] == "Adam":
    optim = torch.optim.Adam(model.parameters(), lr=adjust.optimizer_dict["lr"],
                             weight_decay=adjust.optimizer_dict["weight_decay"])
elif adjust.optimizer_dict["type"] == "SGD":
    optim = torch.optim.SGD(params, lr=adjust.optimizer_dict["lr"],
                            momentum=adjust.optimizer_dict["momentum"],
                            weight_decay=adjust.optimizer_dict["weight_decay"])
else:
    print("Неизвестныей тип оптимизатора. ")
    exit()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                               step_size=3,
                                               gamma=0.1)

if FIRST_STEP:
    f = open("epoch_" + model_name + ".num", "wb")
    epoch_num = 0
    pickle.dump(epoch_num, f)
    if not os.path.exists(model_name + "_epochs"):
        os.mkdir(model_name + "_epochs")
    f.close()
else:
    f = open("epoch_" + model_name + ".num", "rb")
    epoch_num = pickle.load(f)
    model.load_state_dict(torch.load(model_name + ".net"))
    f.close()

analysis_data = train(model, epoch_num, max_epochs, optim, lr_scheduler)
df = pd.DataFrame(analysis_data)

# ----------------Завершение цикла обучения
f = open("epoch_" + model_name + ".num", "wb")
step = epoch_num + max_epochs
pickle.dump(step, f)
f.close()
model = model.to("cpu")
torch.save(model.state_dict(), model_name + ".net")

if FIRST_STEP:
    df.to_csv(model_name + ".csv")
else:
    df.to_csv(model_name + ".csv", mode="a", header=False)

# images, targets = next(iter(train_data))
# images = [img for img in images]
# targets = [{k:v for k, v in t.items()} for t in targets]


# output = model(images, targets)
# model.eval()
# x = [torch.rand(3, 504, 378), torch.rand(3, 100, 40)]
# predictions = model(x)           # Returns predictions
#
# print(output)
# print(predictions)
"""
, {'boxes': tensor([[1.9240e+01, 3.4788e+01, 2.3888e+01, 3.9677e+01],
        [2.9226e+01, 2.6600e+01, 3.5909e+01, 4.0763e+01],     
       grad_fn=<StackBackward0>), 'labels': 'scores':]

"""
