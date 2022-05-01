from cmath import pi
import os
from pyexpat import model
import torch
import yaml
from Model import get_istance_segmentation
from ptcoco_dataset import get_transform,customDataset
import torchvision
from PIL import Image

def collect_fn(batch):
    return tuple(zip(*batch))


def cocopipeline_train(pipeline):
    print(pipeline)
    input_image_cocodir=pipeline['image_root']
    input_json_coco =pipeline["input_annoation"]
    mm_detection = True
    num_classes = 4
    num_epoch = 2
    model= get_istance_segmentation(num_classes)
    device=torch.device('cpu')
    model.to(device)
    params=[p for p in model.parameters() if p.requires_grad]

    optimizer= torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=0.0005) 
    if os.path.exists(os.path.abspath(input_image_cocodir)):
        print("hiii")
        sample_dataset=customDataset(os.path.abspath(input_image_cocodir),os.path.abspath(input_json_coco),transforms=get_transform())
        #print(sample_dataset)

        data_loader=torch.utils.data.DataLoader(sample_dataset,batch_size=1,shuffle=True,num_workers=4,collate_fn=collect_fn)
        len_dataloader=len(data_loader)
        for epoch in range(num_epoch):
            model.train()
            i = 0 
            for imgs,annos in data_loader:
                i +=1
                imgs=list(img.to(device) for img in imgs)
                #print(imgs)
                annos=[{k: v.to(device) for k,v in anno.items()} for anno in annos]
                loss_dict= model(imgs,annos)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                print(f'Iteration : {i}/{len_dataloader}, loss: {losses}')

        path=os.path.abspath("")
        torch.save(model.state_dict(),'model.pth')


def evaluator(pipeline):
    ################################
    ####   model evaluation    #####
    ################################
    model = torch.load('model.pth')
    model.eval()
    transform = torchvision.transforms.ToTensor()
    img = Image.open('./object_detection/data/14321263043_b76ef054d3_k.jpg')
    input = transform(img)
    




if __name__ =='__main__':
    torch.multiprocessing.freeze_support
    with open(os.path.abspath(os.path.join(os.getcwd(),'object_detection','config.yaml')),'r')as fd:
        params =yaml.safe_load(fd)
    pipelines=params["pipeline"]
    for pipeline in pipelines:
        
        if pipeline == 'train':
            print(pipelines['train'])
            cocopipeline_train(pipelines['train'])
        elif pipeline == 'test':
            evaluator(pipelines['test'])
