from matplotlib import pyplot as plt
from torchvision import transforms as T
import os
import random
import numpy as np

def tn_2_np(t):
    transform = T.Compose([
        T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
    ])
    rgb = True if len(t) == 3 else False
    return (transform(t) * 255).detach().cpu().permute(2, 1, 0).numpy().astype(np.uint8) if rgb else (t.squeeze() * 255).detach().cpu().numpy().astype(np.uint8)

def visualization(ds, im_number, save_file=None, data_type=None, img_types=['.png', 'jpg']):
    os.makedirs(save_file, exist_ok=True)
    plt.figure(figsize=(25, 20))
    rows = im_number // 4
    cols = im_number // rows
    count = 1
    index = [random.randint(0, len(ds)-1) for _ in range(im_number)]

    for i, idx in enumerate(index):
        if count == im_number + 1:
            break
        im, gt = ds[idx]
        im, gt = im.float(), gt.float() if gt is not None else None  # Convert to float32

        # Plot original image
        plt.subplot(rows, cols, count)
        plt.imshow(tn_2_np(im))
        plt.axis("off")
        plt.title("Original Image")
        count += 1

        # Plot ground truth if available
        if gt is not None:
            plt.subplot(rows, cols, count)
            plt.imshow(tn_2_np(gt))
            plt.axis("off")
            plt.title('GT')
            count += 1

    plt.savefig(f"{save_file}/{data_type}.{'.'.join([img_type for img_type in img_types])}")
def plots(r, save_file = None, data_name = None):
    os.makedirs(save_file, exist_ok =True)
    plt.figure(figsize=(8,4))
    plt.plot(r["tr_loss"], label = "Train Loss")
    plt.plot(r["val_loss"], label = "Validation Loss")
    plt.title("Train and Validation Losses")
    plt.xticks(np.arange(len(r["val_loss"])), [i for i in range(1, len(r["val_loss"])+1)])
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()
    plt.savefig(f"{save_file}/{data_name}_loss.png")
    # plt.ylim(0, 0.1)


    plt.figure(figsize=(8,4))
    plt.plot(r["tr_pa"], label = "Train PA")
    plt.plot(r["val_pa"], label = "Validation PA")
    plt.title("Train and Validation PA")
    plt.xticks(np.arange(len(r["val_loss"])), [i for i in range(1, len(r["val_loss"])+1)])
    plt.xlabel("Epochs")
    plt.ylabel("PA")
    # plt.ylim(0.97, 1)
    plt.legend()
    plt.savefig(f"{save_file}/{data_name}_PA.png")
   
    
    plt.figure(figsize=(8,4))
    plt.plot(r["tr_ion"], label = "Train mioU")
    plt.plot(r["val_ion"], label = "Validation mioU")
    plt.title("Train and Validation mioU")
    plt.xticks(np.arange(len(r["val_loss"])), [i for i in range(1, len(r["val_loss"])+1)])
    plt.xlabel("Epochs")
    plt.ylabel("mIoU")
    plt.legend()
    plt.savefig(f"{save_file}/{data_name}_mIoU.png")




  