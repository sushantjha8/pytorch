from calendar import c
import torch
import os
from PIL import Image
from pycocotools.coco import COCO
import torchvision

class customDataset(torch.utils.data.Dataset):
    def __init__(self,root,annotation,transforms=None):
        #image root path
        self.root=root
        self.transforms=transforms
        self.coco = COCO(annotation)
        self.ids =list(sorted(self.coco.imgs.keys()))
    def __getitem__(self,index):
        # own coco file
        coco=self.coco
        #Image ID
        img_id = self.ids[index]
        # list: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        #Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        #path for input image
        path =coco.loadImgs(img_id)[0]['file_name']

        # image 
        img=Image.open(os.path.join(self.root,path))

        # Bounding boxes in coco is [xmin,ymin,w,h]
        # But in pytorch we use [xmin,ymin,xmax,ymax] madatory
        boxes = []
        for i in range(len(coco_annotation)):
            xmin=coco_annotation[i]['bbox'][0]
            ymin=coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin,ymin,xmax,ymax])
        bboxes = torch.as_tensor(boxes,dtype=torch.float32)
        labels =torch.ones((len(coco_annotation),),dtype=torch.int64)
        img_id = torch.tensor([img_id])
        area=[]
        for in in range(len(coco_annotation)):
            area.append(coco_annotation[i]['area'])
        area=torch.as_tensor(area,dtype=torch.float32)
        iscrowd = torch.zeros((len(coco_annotation),),dtype=torch.int64)


        #convert to dictonanry format

        my_annotation={}
        my_annotation["boxes"]=boxes
        my_annotation["labels"]=labels
        my_annotation["image_id"]=img_id
        my_annotation["area"]=area
        my_annotation["iscrowd"]=iscrowd


        # transform image

        if self.transforms is not None:
            img=self.transforms(img)
        return img,my_annotation
    def __len__(self):
        return len(self.ids)

def get_transform():
    custom_trans=[]
    custom_trans.append(torchvision.transform.ToTensor())
    return torchvision.transforms.Compose(custom_trans)