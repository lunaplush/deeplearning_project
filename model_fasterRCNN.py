from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def create_model(num_classes=6, pretrained=False):
    if pretrained:
        pretrained_znach = "DEFAULT"
    else:
        pretrained_znach = None
    model = fasterrcnn_resnet50_fpn(weights=pretrained_znach)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
