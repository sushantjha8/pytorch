from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

def get_istance_segmentation(num_classes):
    # load an instance model pretrained on coco
    model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of features for classifier
    in_features=model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
    return model