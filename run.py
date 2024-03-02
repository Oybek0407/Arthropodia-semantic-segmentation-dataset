import argparse, timm, torch, random, pandas as pd, numpy as np,  pickle as p
from matplotlib import pyplot as plt
from torchvision import transforms as T
import albumentations as A
from albumentations.pytorch import  ToTensorV2
from utils import visualization, plots
import segmentation_models_pytorch as smp
from model import set_up
from data import get_dl
from tqdm import tqdm
from train import train
from inference import inferance
def run(args):

        mean, std= [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        trsform = A.Compose([A.Resize(224,224),
                            A.Normalize(mean =mean, std=std), 
                             ToTensorV2(transpose_mask=True)], is_check_shapes=False)

        
        tr_dl, val_dl, ts_dl = get_dl(root = args.data_path, bs = args.batch_size , transformatios=trsform) 
        
        data_name = {tr_dl: "train", val_dl: "valid", ts_dl: "test"}
        for data, name in data_name.items():
                visualization(ds=data.dataset, im_number=args.im_num, save_file=args.save_sample, data_type=name )
        print(f"\n Sample images are being saved in a file named {args.save_sample}!\n")

        
        model = smp.Unet(encoder_name = args.model_name, classes = args.classes, encoder_depth = args.encoder_depth, 
                 encoder_weights = args.encoder_weights, activation = None, decoder_channels= args.decoder_channels)
        print("model yuklanib olindi!")
        
        device, model, optimazer, loss_fn, epochs = set_up(model)
        result = train(model = model, tr_dl = tr_dl, val_dl = val_dl, epochs = epochs, device = device,
                       opt = optimazer, loss_fn = loss_fn, save_prefix = args.save_prefix , save_file = args.save_model)
        print("Model yakunlandi!")
        
        plots(r = result, save_file =args.save_file, data_name = args.data_path)
        print(f"\n학습률 조정이 완료되고 결과가  {args.save_file}에 저장 되었습니다\n")
        # inference
        model.load_state_dict(torch.load(args.save_name_model))
        model.eval()
        inferance(dl = ts_dl, model = model, device = device, num_im = 15, save_inference_images = args.inf_file, data_name = args.data_path)
        
        print(f"\nInference 과정이 완료되었고 GrandCAM의 결과를 {args.inf_file}  파일에서  확인 가능합니다!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lentils Types Classification Demo")

    parser.add_argument("-dp", "--data_path", type=str, default="arthropodia", help="path of dataset")
    parser.add_argument("-nm", "--im_num", type=str, default= 20 , help="number of images")
    parser.add_argument("-sf", "--save_file", type=str, default= "Learning rate" , help="number of images") 
    parser.add_argument("-fs", "--save_sample", type=str, default= "Sample images" , help="number of images")# visu
        
    parser.add_argument("-bs", "--batch_size", type=str, default= 64, help="path of dataset")
        
    parser.add_argument("-md", "--model_name", type=str, default= "resnet18", help="Model name for training")
    parser.add_argument("-cl", "--classes", type=str, default=  2 , help=" number of classes")
    parser.add_argument("-ed", "--encoder_depth", type=str, default= 5, help="encoder depth of model")
    parser.add_argument("-ew", "--encoder_weights", type=str, default= "imagenet", help="encoder depth of model")
    parser.add_argument("-dc", "--decoder_channels", type=str, default= [256, 128, 64, 32, 16] , help="encoder depth of model")
        
    parser.add_argument("-sp", "--save_prefix", type=str, default= "arthropodia", help="File for saving")
    parser.add_argument("-sm", "--save_model", type=str, default= "arthropodia_file", help="File for saving") # for model
    parser.add_argument("-if", "--inf_file", type=str, default= "Inference_file", help="File for saving inference results")

    parser.add_argument("-mn", "--save_name_model", type=str, default= "arthropodia_file/arthropodia_best.pth", help="File for saving")
        
  
    # parser.add_argument("-my", "--model_yulagi", type=str, default="saved_models/Clouds_best_model.pth", help="Trained model")

    args = parser.parse_args()
    run(args)
