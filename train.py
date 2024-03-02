from model import Metrics
from tqdm import tqdm
import timm, os, torch, numpy as np
def train(model, tr_dl, val_dl, epochs, device, opt, loss_fn, save_prefix, save_file =None):
    tr_loss, tr_pa, tr_ion = [],[],[]
    val_loss, val_pa, val_ion = [],[],[]
    tr_lens, val_lens = len(tr_dl), len(val_dl)
    best_loss = np.inf
    model.to(device)
    print("Train 시작 할 예정입니다.......")
    for epoch in range(epochs):
        print(f"{epoch+1} - Train 시작중이다")
        tr_losses, tr_PA, tr_IOU =0,0,0
        model.train()
        for idx , batch in enumerate(tqdm(tr_dl)):
            im, gt = batch
            im, gt  = im.to(device), gt.to(device)
            pred = model(im)
            met = Metrics(pred, gt, loss_fn)
            loss_ = met.loss()
            

            tr_IOU += met.mIoU()
            tr_PA+= met.PA()
            tr_losses+=loss_.item()
            opt.zero_grad()
            loss_.backward()
            opt.step()
            
        tr_IOU /=tr_lens
        tr_PA /= tr_lens
        tr_losses /= tr_lens
        tr_loss.append(tr_losses); tr_pa.append(tr_PA); tr_ion.append(tr_IOU)
        print("\n ------------------------------------------")
        print(f"{epoch+1} - epoch train result: \n")
        print(f"Train loss                 --> {tr_losses:.3f}")
        print(f"Train PA                   --> {tr_PA:.3f}")
        print(f"Train mIoU                 --> {tr_IOU:.3f}\n")
        
        print(f"{epoch+1} - Validation 시작중이다")
        val_losses, val_PA, val_IOU =0,0,0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dl)):
                im, gt = batch
                im, gt  = im.to(device), gt.to(device)
                pred = model(im)
                met = Metrics(pred, gt, loss_fn)
                val_losses += met.loss().item()
                val_PA += met.PA()
                val_IOU+= met.mIoU()
            val_losses /= val_lens
            val_PA /=val_lens
            val_IOU /= val_lens
            
            print(f"Validation {epoch+1} - epoch  results\n")
            print(f"Validation loss             --> {val_losses:.3f}")
            print(f"Validation PA               --> {val_PA:.3f}")
            print(f"Validation mIoU             --> {val_IOU:.3f}")
            val_loss.append(val_losses); val_pa.append(val_PA); val_ion.append(val_IOU)
            if val_losses < best_loss:
                best_loss = val_losses
                os.makedirs(save_file, exist_ok= True)
                torch.save(model.state_dict(), f"{save_file}/{save_prefix}_best.pth")
                
    return {"tr_loss": tr_loss, "tr_pa": tr_pa , "tr_ion": tr_ion,
           "val_loss": val_loss, "val_pa": val_pa, "val_ion": val_ion}  