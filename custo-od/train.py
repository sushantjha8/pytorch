from cmath import pi
import os
from pyexpat import model
import torch
import yaml
from ptcoco_dataset import get_transform,customDataset,collater,AspectRatioBasedSampler
import torchvision
from PIL import Image
import torch.optim as optim
from retinanet import model
import collections
def retinanet_net_train(pipeline):
    print(pipeline)
    input_image_cocodir=pipeline['image_root']
    input_json_coco =pipeline["input_annoation"]
    depth =int(pipeline["depths"])
    mm_detection = True
    num_classes = 4
    num_epoch = 2

    
    sample_dataset=customDataset(os.path.abspath(input_image_cocodir),os.path.abspath(input_json_coco),transforms=get_transform())
    print(sample_dataset)
    sampler = AspectRatioBasedSampler(sample_dataset, batch_size=2, drop_last=False)
    data_loader=torch.utils.data.DataLoader(sample_dataset,batch_size=1,shuffle=True,num_workers=4,collate_fn=collater)#, batch_sampler=sampler)
    len_dataloader=len(data_loader)
    print(len_dataloader)

     # Create the model
    if depth == 18:
        retinanet = model.resnet18(num_classes=1, pretrained=True)
    elif depth == 34:
        retinanet = model.resnet34(num_classes=1, pretrained=True)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=1, pretrained=True)
    elif depth == 101:
        retinanet = model.resnet101(num_classes=1, pretrained=True)
    elif depth == 152:
        retinanet = model.resnet152(num_classes=1, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')


    device=torch.device('cpu')
    try:
        retinanet.to(device)
        print(f"model loadded to device {retinanet.parameters()}")
    except Exception as e:
        print("issue in loading model")
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)














if __name__ =='__main__':
    torch.multiprocessing.freeze_support
    with open(os.path.abspath(os.path.join(os.getcwd(),'custo-od','config.yaml')),'r')as fd:
        params =yaml.safe_load(fd)
    pipelines=params["pipeline"]
    for pipeline in pipelines:
        
        if pipeline == 'train':
            print(pipelines['train'])
            retinanet_net_train(pipelines['train'])
        # elif pipeline == 'test':
        #     evaluator(pipelines['test'])
