from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as T
import timm, os, torch, numpy as np
from PIL import Image
# from utils import tr_2_im
import torch, os, cv2, random
def inferance(dl, model, device, num_im , save_inference_images = None, data_name =None ):
    os.makedirs(save_inference_images, exist_ok = True)
    cols = 3
    rows = (num_im + cols -1)//cols
    count = 1
    ims, gts, preds = [],[],[]
    for idx, data in enumerate(dl):
        im, gt = data
        with torch.no_grad():
            pred = torch.argmax(model(im.to(device)), dim=1)
        ims.append(im.cpu().numpy())
        gts.append(gt.cpu().numpy())
        preds.append(pred.cpu().numpy())
    plt.figure(figsize = (25, 20))
    while count <= num_im:
        for idx, (im, gt, pred) in enumerate(zip(ims, gts, preds)):
            if count > num_im: break
            plt.subplot(rows, cols, count)
            plt.imshow(im.squeeze(0).transpose(2, 1, 0))
            plt.title("Input Images")
            plt.axis('off')
            count+=1
            
            plt.subplot(rows, cols, count)
            plt.imshow(gt.transpose(2,1,0))
            plt.title("GT image")
            plt.axis("off")
            count+=1
            
            plt.subplot(rows, cols, count)
            plt.imshow(pred.transpose(2, 1, 0))
            plt.title("Prediction Mask")
            plt.axis("off")
            count+=1
            plt.savefig(f"{save_inference_images}/{data_name}.png")