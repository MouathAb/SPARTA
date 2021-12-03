import torch
from typing import TypeVar
from tqdm import tqdm
import numpy as np
import pandas as pd
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.T2TViT.models.t2t_vit import t2t_vit_14,t2t_vit_24,t2t_vit_t_14,t2t_vit_t_24
from models import T2TViT
import argparse
import glob
from loader import load_dataset
from config import load_model
from config2 import load_model2
from models.from_timm2 import *

torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)
#torch.autograd.set_detect_anomaly(True)    


class squeezeExcitaion(nn.Module):
    def __init__(self, n_classes, in_features,reduction_ratio=2):
        super(squeezeExcitaion, self).__init__()
        
        #self.transformer_out = layer
        self.in_features = in_features
        self.reduction_ratio = reduction_ratio
        self.num_features_reduced = self.in_features // reduction_ratio
        
        self.fc1 = nn.Linear(self.in_features, self.num_features_reduced, bias=True)
        self.fc2 = nn.Linear(self.num_features_reduced, self.in_features, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #norm_layer = None or partial(nn.LayerNorm, eps=1e-6)
        #self.norms = norm_layer(self.in_features)

        self.cls = nn.Linear(self.in_features,n_classes, bias=True)
    def forward(self, squeeze_tensor):
        #squeeze and excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        output_tensor = torch.mul(squeeze_tensor, fc_out_2)
        
        #classificatioin
        y = self.cls(output_tensor)
        
        return y
    
def mixup_data(x, y, alpha=1.0, device=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# focal loss
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __inti__(self, weight=None,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.weight = weight
    
    def forward(self, x, target):
        
        ce_loss = F.cross_entropy(x,target,reduction= self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1-pt) ** 2 * ce_loss).mean()
        return focal_loss
    
# IoU loss
SMOOTH = 1e-6
class GIoU(nn.modules.loss._WeightedLoss):
    """Generalized IoU Loss. Paper: https://arxiv.org/abs/1902.09630
    """
    def __inti__(self, reduction='mean'):
        super(GIoU, self).__init__(reduction=reduction)
        self.reduction = reduction
        #self.threshold = 0.5
        
    def forward(self, os,ol,ts,tl):
        intersection =torch.clamp(torch.minimum(os+ol,ts+tl) - torch.maximum(os,ts),min=0)
        union = ol + tl - intersection 
        iou = intersection/(union+ SMOOTH)
        
        ac = torch.max(os+ol,ts+tl) - torch.min(os,ts) 
        
        giou = iou - (ac-union)/(ac + SMOOTH)
        loss = 1 - giou
        #giou = torch.clamp(20*(giou-self.threshold),0,10).ceil()/10
        
        if self.reduction == 'sum':
            return torch.sum(loss)
        if self.reduction =='mean':
            return torch.mean(loss)
        
        return  loss
    
def train_one_epoch(model, train_loader, criterion, reg, optimizer, epoch,device):
    # keep track of training loss
    epoch_loss = 0.0
    epoch_loss_giou = 0.0
    epoch_loss_cls = 0.0
    epoch_accuracy = 0.0
    epoch_loss_mses = 0.0
    epoch_loss_msel = 0.0
    total = 0
    ###################
    # train the model #
    ###################
    model.train()
    with tqdm (train_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            total += data.shape[0]
            # move tensors to GPU if CUDA is available
            target_exp, target_start, target_length = target
            data, target_exp, target_start, target_length = data.to(device), target_exp.to(device), target_start.to(device), target_length.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            if False:
                #regulazation mixup
                data, targets_a, targets_b, lam = mixup_data(data, target_exp,1., device)
                data, targets_a, targets_b = map(Variable, (data,targets_a, targets_b))
                
            # forward pass: compute predicted outputs by passing inputs to the model
            output_exp, output_start, output_length = model.forward(data)
            
            # Calculate Accuracy
            accuracy = (output_exp.argmax(dim=1) == target_exp).float().sum() 
            
            # Calculate loss
            if False:
                loss = mixup_criterion(criterion, output_exp, targets_a, targets_b, lam).to(device, dtype=torch.float32)
            else:
                loss_giou = GIoU(reduction='mean')(output_start.squeeze(1),output_length.squeeze(1),target_start,target_length)
                loss_mse_start = nn.MSELoss(reduction='mean')(output_start.squeeze(1)/(WIN-1),target_start/(WIN-1))
                loss_mse_length = nn.MSELoss(reduction='mean')(output_length.squeeze(1)/(WIN-1),target_length/(WIN-1))
                loss_spot = loss_giou + loss_mse_start + loss_mse_length
                loss_cls = criterion(output_exp,target_exp)
                loss = loss_cls + loss_spot
            # loss = criterion(output_exp,target_exp)+ GIoU(output_start.squeeze(1),output_length.squeeze(1),target_start,target_length)
            # MSE : nn.MSELoss()(output_start.squeeze(1),target_start) + nn.MSELoss()(output_length.squeeze(1),target_length)
            # Backpropagation
            loss.backward()       
    
            # update training loss and accuracy
            epoch_loss += loss.item()
            epoch_loss_giou += loss_giou.item()
            epoch_loss_cls += loss_cls.item()
            epoch_loss_mses += loss_mse_start.item()
            epoch_loss_msel += loss_mse_length.item()
            epoch_accuracy += accuracy.item()
    
            
            # perform a single optimization step (parameter update)
            optimizer.step()   
            
            #set the postfix for progress bar options
            tepoch.set_postfix(loss=loss.item(),
                               loss_giou=loss_giou.item(),
                               loss_cls=loss_cls.item(), 
                               loss_mse_length=loss_mse_length.item(),
                               loss_mse_start=loss_mse_start.item(), 
                               accuracy=100. * (accuracy.item()/data.shape[0]),
                               )

    return epoch_loss / len(train_loader), epoch_accuracy / total, epoch_loss_giou/len(train_loader), epoch_loss_cls/len(train_loader), epoch_loss_mses/len(train_loader), epoch_loss_msel/len(train_loader)

#def alpha(ptar,tar,i):
#    return (ptar==tar).float()
def alpha(tar,j):
    if j != 0:
        return (tar[j-1]==tar[j]).float()
    else:
        return 1
    
def validate_one_epoch(model, valid_loader, criterion, device):
    # keep track of training loss
    valid_loss = 0.0
    valid_loss_giou = 0.0
    valid_loss_cls = 0.0
    valid_accuracy = 0.0
    valid_loss_mse_start = 0.0
    valid_loss_mse_length = 0.0
    valid_accuracy = 0.0
    total = 0
    ######################
    # prevalid the model #
    ######################
    model.eval()

    for i, (data, target) in enumerate(valid_loader):
        total += data.shape[0]
        # move tensors to GPU if CUDA is available
        target_exp, target_start, target_length = target
        data, target_exp, target_start, target_length = data.to(device), target_exp.to(device), target_start.to(device), target_length.to(device)
        
        with torch.no_grad():
            # forward pass: compute predicted outputs by passing inputs to the model
            output_exp, output_start, output_length = model.forward(data)
            
            # Calculate Accuracy
            accuracy = (output_exp.argmax(dim=1) == target_exp).float().sum()
            
            # Calculate loss
            loss_giou = GIoU(reduction='mean')(output_start.squeeze(1),output_length.squeeze(1),target_start,target_length)
            loss_mse_start = nn.MSELoss(reduction='mean')(output_start.squeeze(1)/(WIN-1),target_start/(WIN-1))
            loss_mse_length = nn.MSELoss(reduction='mean')(output_length.squeeze(1)/(WIN-1),target_length/(WIN-1))
            loss_cls = criterion(output_exp,target_exp)
            loss = loss_cls + loss_giou + loss_mse_start + loss_mse_length #criterion(output_exp,target_exp) + GIoU(reduction='mean')(output_start.squeeze(1),output_length.squeeze(1),target_start,target_length)
            
            # update training loss and accuracy
            valid_loss += loss.item()
            valid_loss_giou += loss_giou.item()
            valid_loss_mse_start += loss_mse_start.item()
            valid_loss_mse_length += loss_mse_length.item()
            valid_loss_cls += loss_cls.item()
            valid_accuracy += accuracy.item()
            

    return valid_loss / len(valid_loader), valid_accuracy / total, valid_loss_giou/len(valid_loader), valid_loss_cls/len(valid_loader), valid_loss_mse_start/len(valid_loader), valid_loss_mse_length/len(valid_loader)


def one_epoch(epoch, model, optimizer, criterion, reg, device, train_loader,
              valid_loader):
    gc.collect()

    print("=" * 20)
    print(f"EPOCH {epoch} TRAINING...")
    train_loss, train_acc = 0, 0
    train_loss, train_acc, train_loss_giou, train_loss_cls, train_loss_mse_start, train_loss_mse_length = train_one_epoch(model, train_loader, criterion, reg,
                                                                                                                          optimizer,epoch, device)
    train_acc = np.mean(train_acc)
    print(
        f"[TRAIN] EPOCH {epoch} - LOSS: {train_loss:2.4f}, ACCURACY:{train_acc:2.4f}, LOSS_GIoU: {train_loss_giou:2.4f}, LOSS_CLS:{train_loss_cls:2.4f}, LOSS_MSE_START: {train_loss_mse_start:2.4f}, LOSS_MSE_LENGTH:{train_loss_mse_length:2.4f} "
    )
    gc.collect()
    valid_loss, valid_acc = 0, 0
    
    if valid_loader is not None:
        gc.collect()
        print(f"EPOCH {epoch} VALIDATING...")
        valid_loss, valid_acc, valid_loss_giou, valid_loss_cls, valid_loss_mse_start, valid_loss_mse_length = validate_one_epoch(model, valid_loader,
                                                                                                                                 criterion, device)
        valid_acc = np.mean(valid_acc)      
        print(f"[VALID] LOSS: {valid_loss:2.4f} ,ACCURACY: {valid_acc:2.4f}, LOSS_GIoU: {valid_loss_giou:2.4f}, LOSS_CLS: {valid_loss_cls:2.4f}, LOSS_MSE_START: {valid_loss_mse_start:2.4f}, LOSS_MSE_LENGTH:{valid_loss_mse_length:2.4f} ")

        gc.collect()
    
    return train_loss, train_loss_giou, train_loss_cls, train_loss_mse_start, train_loss_mse_length, train_acc, valid_loss, valid_loss_giou, valid_loss_cls, valid_loss_mse_start, valid_loss_mse_length , valid_acc



def fit(
    model: torch.nn.Module,
    model_name: str,
    epochs: int,
    criterion,
    optimizer,
    scheduler,
    reg,
    device,
    train_loader,
    valid_loader,
    name,
) -> None:

    # keeping track of losses as it happen
    train_losses = []
    train_losses_giou = []
    train_losses_cls = []
    train_losses_mse_start = []
    train_losses_mse_length = []
    valid_losses = []
    valid_losses_giou = []
    valid_losses_cls = []
    valid_losses_mse_start = []
    valid_losses_mse_length = []
    train_accs = []
    valid_accs = []
    best_valid_acc = 0.
    epc = 0
    # To froze the model

    for epoch in range(1, epochs + 1):
        train_loss, train_loss_giou, train_loss_cls, train_loss_mse_start, train_loss_mse_length, train_acc, valid_loss, valid_loss_giou, valid_loss_cls, valid_loss_mse_start, valid_loss_mse_length, valid_acc = one_epoch(
            epoch,
            model,
            optimizer,
            criterion,
            reg,
            device,
            train_loader,
            valid_loader,
        )
        if scheduler is not None:
            print(f'LR={scheduler.get_last_lr()}')
        
        if valid_acc > best_valid_acc:
            print('--- ACC has improved ---')
            #torch.save(model.state_dict(),
            #           './src2/models/trained/'+name+'/' + model_name + '_best_fold_'+str(SUB)+'.pth')
            best_valid_acc = valid_acc
            epc = epoch
        #torch.save(model.state_dict(), './src2/models/trained/'+name+'/'+model_name+'_fold_'+str(SUB)+'.pth')
        valid_losses.append(valid_loss)
        valid_losses_giou.append(valid_loss_giou)
        valid_losses_cls.append(valid_loss_cls)
        valid_losses_mse_start.append(valid_loss_mse_start)
        valid_losses_mse_length.append(valid_loss_mse_length)
        valid_accs.append(valid_acc)
        
        train_losses.append(train_loss)
        train_losses_giou.append(train_loss_giou)
        train_losses_cls.append(train_loss_cls)
        train_losses_mse_start.append(train_loss_mse_start)
        train_losses_mse_length.append(train_loss_mse_length)
        train_accs.append(train_acc)
        if scheduler is not None:
            scheduler.step()
    torch.save(model.state_dict(), './src2/models/trained/'+ name + '/'+ model_name +'_fold_'+str(SUB)+'.pth')
    print('\n\t the best accuracy is : ',best_valid_acc , '\tat epoch number : ', epc) #just uncomment the paragraph and change that to best_valid_acc
    df = pd.DataFrame([],columns=['train_losses','train_losses_giou','train_losses_cls','train_losses_mse_start','train_losses_mse_length','train_accs','valid_losses','valid_losses_giou','valid_losses_cls','valid_losses_mse_start','valid_losses_mse_length','valid_accs'])
    df['train_losses'] = train_losses
    df['train_losses_giou'] = train_losses_giou
    df['train_losses_cls'] = train_losses_cls
    df['train_losses_mse_start'] = train_losses_mse_start
    df['train_losses_mse_length'] = train_losses_mse_length
    df['train_accs'] = train_accs
    df['valid_losses'] = valid_losses
    df['valid_losses_giou'] = valid_losses_giou
    df['valid_losses_cls'] = valid_losses_cls
    df['valid_losses_mse_start'] = valid_losses_mse_start
    df['valid_losses_mse_length'] = valid_losses_mse_length
    df['valid_accs'] = valid_accs
    df.to_csv('./RESULT/TRAIN_VALID/[TRAIN_VALID]_data_'+name+'_model_'+model_name +'_fold_'+str(SUB)+ '_best_valid_acc_'+str(best_valid_acc)+'_Epoch_'+str(epc)+'.csv',index_label=False)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-m",
        "--model",
        default="vit_b_16",
        type=str,
        help="Which model to train",
    )
    
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="learning rate",
    )
    
    parser.add_argument(
        "-b",
        "--batch",
        default=8,
        type=int,
        help="batch size",
    )
    
    parser.add_argument(
        "-e",
        "--epoch",
        default=30,
        type=int,
        help="number of epochs",
    )
    
    parser.add_argument(
        "-se",
        "--squeezeexcitation",
        default='False',
        type=str,
        help="Add a squeeze and excitation block on top of the model",
    )
    
    parser.add_argument(
        "--data",
        default='sfew',
        type=str,
        help="Path to the global content of all data",
    )
    
    parser.add_argument(
        "--scheduler",
        default='lr_scheduler',
        type=str,
        help="Scheduler",
    )
    
    parser.add_argument(
        "--dataAug",
        default='True',
        type=str,
        help="Train with Data Augmentation",
    )
    
    parser.add_argument(
        "-reg",
        "--Regulizer",
        default='True',
        type=str,
        help="Train with Cutout and Mixup regulizer",
    )
    
    parser.add_argument(
        "--pretrained",
        default='True',
        type=str,
        help="use the pretrained model on MAE for cls_tkn",
    )
    
    parser.add_argument(
        "--vit",
        default='True',
        type=str,
        help="use the vit model on the first frame",
    )
    
    parser.add_argument(
        "-beta",
        "--tokenLoss",
        default=0.0,
        type=float,
        help="distance between cls_token of two successive frames",
    )
    
    parser.add_argument(
        "-sub",
        "--subject",
        default=1,
        type=int,
        help="subject - LOSO",
    )
    
    parser.add_argument(
        "-w",
        "--window",
        default=60,
        type=int,
        help="length of the slidingWindow ",
    )
    
    args = parser.parse_args()
    
    WIN = args.window
    MODEL = args.model
    SE = args.squeezeexcitation == 'True'
    pretrained = 'True' == args.pretrained and len(glob.glob('./src/models/rafdb/*')) > 0
    DATA = args.data
    LR = args.lr
    beta = args.tokenLoss
    EPOCH = args.epoch
    BATCH_SIZE = args.batch
    SCHD = args.scheduler
    DA = args.dataAug == 'True'
    REG = args.Regulizer == 'True'
    VIT = args.vit == 'True'
    SUB = args.subject
    mixup_args = dict(
            mixup_alpha=0.0, cutmix_alpha=0.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=3)
    #mixup_fn = Mixup(**mixup_args)
    #mixup_fn.mixup_enabled = False
     
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model2(MODEL)
    
    if SE:
        model.head = squeezeExcitaion(n_classes=3,in_features=model.head.in_features,reduction_ratio=2)
    OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=LR)
    CRITERIONi = nn.CrossEntropyLoss()
    weights = torch.FloatTensor([1.0,1.0,1.0]).to(DEVICE)
    CRITERION = FocalLoss(weight=weights,reduction='mean')
    if SCHD == 'lr_scheduler':
        SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER, 10, gamma=0.2)
    else:
        #print('No used Scheduler')
        SCHEDULER = None
    
    print('[ TRAIN ]')    
    print('MODEL       : ', MODEL)
    print('WINDOW      : ', WIN)
    print('SE attn     : ', SE)
    print('PRETRAIN    : ', pretrained)
    print('LR          : ', LR)
    print('BATCH_SIZE  : ', BATCH_SIZE)
    print('EPOCH       : ', EPOCH)
    print('TOKEN LOSS  : ', beta)
    print('CRITERION   : ', CRITERION)
    print('DATA AUG    : ', DA)
    print('REGULIZER   : ', REG)
    print('DATA TRAIN  : ', DATA)
    print('SCHEDULER   : ', SCHEDULER)
    print('OPTIMIZER   : ', OPTIMIZER)
    
    print('\n[LOAD] ...')
    train_loader, valid_loader, _ = load_dataset("./data",DATA, model.img_size[0],BATCH_SIZE,DA,REG,SUB,WIN)
    print('[DONE!]\n')
    if SE:
        MODEL += '_SE'

    model.to(DEVICE)
    
    fit(
        model,
        MODEL,
        EPOCH,
        CRITERION,
        OPTIMIZER,
        SCHEDULER,
        REG,
        DEVICE,
        train_loader,
        valid_loader,
        DATA,
    )
    
    print('\n[FINISH]')
