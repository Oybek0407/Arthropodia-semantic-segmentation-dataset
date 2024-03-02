import torch, numpy as np
def set_up(model): 
        return "cuda" if torch.cuda.is_available() else "cpu" , model.to('cuda'), torch.optim.Adam(params=model.parameters(), lr=3e-4), torch.nn.CrossEntropyLoss(), 10
# device, model, optimazer, loss_fn, epochs = set_up(model)
class Metrics():
    def __init__(self, pred, gt, loss_fn, eps = 3e-4, number_class = 2):
        self.pred = torch.argmax(torch.nn.functional.softmax(pred, dim =1), dim =1)
        self.pred_ = pred
        self.gt = gt
        self.eps = eps
        self.loss_fn = loss_fn
        self.number_class = number_class

    def to_contiguous(self, inp): return inp.contiguous().view(-1)

    def PA(self):
        with torch.no_grad():
            pa = torch.eq(self.pred, self.gt).int()
        return float(pa.sum())/float(pa.numel())
    
    def mIoU(self):
        pred, gt = self.to_contiguous(self.pred), self.to_contiguous(self.gt)
        IoU_class = []
        for a in range(self.number_class):
            match_pred = pred ==a
            match_gt = gt == a
            if match_gt.long().sum().item() == 0: IoU_class.append(np.nan)
            else:
                intersection = torch.logical_and(match_pred, match_gt).sum().float().item()
                union = torch.logical_or(match_pred, match_gt).sum().float().item()
                iou = intersection/(union+self.eps)
                IoU_class.append(iou)
            return np.nanmean(IoU_class)
    def loss(self):
        return self.loss_fn(self.pred_, self.gt)