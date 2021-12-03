import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

import argparse
import gc

from loader import load_test
from config import load_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import glob
from config import load_model
from config2 import load_model2
from models.from_timm2 import *
from copy import deepcopy


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
    
def get_prediction(model, device, test_loader):
    
    model.eval()
    with tqdm (test_loader, unit="batch") as tepoch:
        for i,(data, target) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            # move tensors to GPU if CUDA is available
            target_exp, target_start, target_length = target
            data, target_exp, target_start, target_length = data.to(device), target_exp.to(device), target_start.to(device), target_length.to(device)
            label = [(target_exp.detach().cpu().numpy(), target_start.detach().cpu().numpy(), target_length.detach().cpu().numpy())]
            with torch.no_grad():
                # forward pass: compute predicted outputs by passing inputs to the model
                output_exp,output_start, output_end = model.forward(data)
                output = [(output_exp.argmax(dim=1).detach().cpu().numpy(),torch.round(output_start.squeeze(1)).detach().cpu().numpy(), torch.round(output_end.squeeze(1)).detach().cpu().numpy())]
                if i == 0:
                    outputs = output
                    labels = label
                else:
                    outputs += output
                    labels += label
    #print(outputs.size())
    return outputs, labels


def accuracy(outputs, labels):
    return (outputs.argmax(dim=1) == labels).float().mean()


def clean_predictions(outputs):
    threshold = 3
    clean_outputs = deepcopy(outputs)
    
    
    for b, output in enumerate(outputs):
        # Replace the few zeros between intervals
        
        dis = 0
        pos = 0
        ici = []
        
        """GET the dominant class"""
        if output.count(1) < output.count(2):
            c = 2
            c_ = 1
        else:
            c = 1
            c_ = 2    
            
        for n,i in enumerate(output):
            if i == 0:
                dis += 1
                pos = n - dis + 1
            else:
                if pos != 0 :
                    ici += [(dis,pos)]
                dis = 0
                pos = 0
            
        for d,p in ici:
            if d < threshold:
                clean_outputs[b][p:p+d] = [c]*d 
            
        # Remove very small intervals
        dis = 0
        pos = 0
        labas = []
        for n,i in enumerate(clean_outputs[b]):
            if i != 0:
                dis += 1
                pos = n - dis + 1
            else:
                if dis != 0:
                    labas += [(dis,pos)]
                dis = 0
                pos = 0
        
        for d,p in labas:
            if d < threshold:
                clean_outputs[b][p:p+d] = [0]*d
        

            
        # Remove the misclass intervals : the misclass is the class with less frequency
        """ ! it has to be one class per sequence """            
        dis = 0
        pos = 0
        labas = []
        
        """GET the dominant class"""
        if clean_outputs[b].count(1) < clean_outputs[b].count(2):
            c = 2
            c_ = 1
        else:
            c = 1
            c_ = 2
            
        for n,i in enumerate(clean_outputs[b]):
            if i == c_:
                dis += 1
                pos = n - dis + 1
            else:
                if dis != 0:
                    labas += [(dis,pos)]
                dis = 0
                pos = 0
        
        for d,p in labas:
            clean_outputs[b][p:p+d] = [c]*d
                
    return clean_outputs

def howmany(outputs,exp='all'):
    alle = 0
    macroe = 0
    microe = 0
    for o in outputs:
        if 1 in o:
            alle += 1
            microe += 1
        if 2 in o:
            alle += 1
            macroe += 1
    
    if exp == 'all':
        return alle
    if exp == 'macro':
        return macroe
    if exp == 'micro':
        return microe

def true_false_positive(p_interval,gt_interval,exp='all'):
    tp = 0.0
    fp = 0.0
    for b,m in enumerate(p_interval):
        n = gt_interval[b]
        if exp =='all':
            continue
        if exp == 'macro':
            if 1 in m:
                m = []
            if 1 in n:
                n = []
        if exp == 'micro':
            if 2 in m:
                m = []
            if 2 in n:
                n = []
                
        if len(m) != 0 and len(n)!= 0:
            I = (np.array(m) & np.array(n)).sum()
            U = (np.array(m) | np.array(n)).sum()
            
            if U == 0:
                iou =0
            else:
                iou = I/U
                
            if iou >= 0.5:
                tp += 1
            else:
                fp += 1
        else:
            if len(m)!=0:
                fp += 1
            else:
                pass
    return tp

def unify(vector):
    z = 60 #length of slidingwindoww
    B = len(vector) # Batch size
    
    sequence = np.zeros((B,z),dtype=int)
    for _ in range(B):
        exp, start, length = vector[_]
        for i in range(z):
            if i in range(int(start),int(start+length)):
                sequence[_,i] = int(exp)
            else:
                sequence[_,i] = 0
    return list(sequence.reshape(B*z,))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="vit_s_16",
        type=str,
        help="Which model to train",
    )
    parser.add_argument(
        "-b",
        "--batch",
        default=1,
        type=int,
        help="batch size",
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
        default='casme2_lv',
        type=str,
        help="Path to the global content of all data",
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
        "-sub",
        "--subject",
        default=1,
        type=int,
        help="subject - LOSO ",
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
    beta = 1
    BATCH_SIZE = args.batch
    VIT = args.vit == 'True'
    SUB = args.subject
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model2(MODEL)
    name = args.data
    if SE:
        model.head = squeezeExcitaion(n_classes=3,in_features=model.head.in_features,reduction_ratio=2)
    
    print('[ TEST ]')    
    print('MODEL       : ', MODEL)
    print('SE attn     : ', SE)
    print('PRETRAIN    : ', pretrained)
    print('BATCH_SIZE  : ', BATCH_SIZE)
    print('DATA TRAIN  : ', DATA)
    print('WINDOW     : ', WIN)
    print('SUBJECT     : ', SUB)
    
    print('\n[LOAD] && [INFERENCE] ...')
            
    if SE:
        MODEL += '_SE'

    
    model.load_state_dict(torch.load('./src2/models/trained/'+name+'/' + MODEL + '_fold_'+str(SUB)+'.pth'))

    model.to(DEVICE)
    gc.collect()
    
    df =  pd.read_csv('./data/test/'+DATA+'_test.csv') 
    df = df[df.Subject.isin([SUB])].reset_index()
    SEQ = len(df)
    outputs = []
    labels = []
    for i in range(SEQ):
        print(i)
        test_loader, _ = load_test("./data",DATA, model.img_size[0],BATCH_SIZE,SUB,i,WIN)        
        outputs_, labels_ = get_prediction(model, DEVICE, test_loader)
        outputs_, labels_ = unify(outputs_), unify(labels_)
        outputs += [outputs_]
        labels += [labels_]
    print('[DONE!]\n')
    
    # CLEAN predection
    print('[CLEAN PRED]')
    clean_outputs = clean_predictions(outputs)
    print('[DONE!]\n')
    print('outputs')
    print(outputs[0])
    print('\n')
    n = howmany(clean_outputs,'all')
    n1 = howmany(clean_outputs,'micro')
    n2 = n -n1 # howmany(outputs,'macro')
    print('clean_outputs')
    print(clean_outputs[0])
    print('\n')
    print('labels')
    print(labels[0])
    m = howmany(labels,'all')
    m1 = howmany(labels,'micro')
    m2 = m -m1 # howmany(outputs,'macro')
    
    gc.collect()
    
    '''
    assert(m != 0)
    tp, fp = true_false_positive(pred_all_intervals,target_all_intervals)
    assert(fp == n-tp)
    fn = m - tp
    recall = tp / m
    try:
        precision = tp / n
    except:
        print("you have a problem here! check the predicted intervals ")
        precision = 0.0
    F_score = 2*tp/(m+n)
    '''
        #Evaluation for overall
    tp = true_false_positive(clean_outputs,labels,exp='all')
    recall = tp/m
    try:
        precision = tp/n
    except:
        print("[Overall] you have a problem here! check the predicted intervals ")
        precision = 0.0
    try:
        F1_score = 2*(recall*precision)/(recall+precision)
    except:
        F1_score = 0.0
    
    print('*'*50)
    print('Spotting Macro/Micro-Expression overall performance')
    print('Recall          : ', recall)
    print('Precision       : ', precision)
    print('F1-score        : ', F1_score)
    
    gc.collect()
    
    #Evaluation for Micro-expression
    try:
        assert(m1 != 0)
    except:
        print(m1)
        assert(m1 != 0)
    tp1 = true_false_positive(clean_outputs,labels,exp='micro')
    fp2 = n1 - tp1
    fn1 = m1 - tp1
    recall1 = tp1 / m1
    try:
        precision1 = tp1 / n1
    except:
        print("[MiE] you have a problem here! check the predicted intervals ")
        precision1 = 0.0
    try : 
        F1_score1 = 2*(recall1*precision1)/(recall1+precision1)
    except:
        F1_score1 = 0.0
    
    print('*'*50)
    print('Spotting Micro-Expression performance')
    print('True positive   : ', tp1)
    #print('False positive  : ', fp1)
    #print('False negative  : ', fn1)
    print('Recall          : ', recall1)
    print('Precision       : ', precision1)
    print('F1-score        : ', F1_score1)
    
    gc.collect()
    
    #Evaluation for Macro-expression
    assert(m2 != 0)
    #tp1 = true_false_positive(pred_macro_intervals,target_macro_intervals)
    #fp1 = n1 - tp1
    #fn1 = m1 - tp1
    tp2 = tp - tp1
    recall2 = tp2 / m2
    try:
        precision2 = tp2 / n2
    except:
        print("[MaE] you have a problem here! check the predicted intervals ")
        precision2 = 0.0
    try:
        F1_score2 = 2*(recall2*precision2)/(recall2+precision2)
    except:
        F1_score2 = 0.0
    
    print('*'*50)
    print('Spotting Macro-Expression performance')
    print('True positive   : ', tp2)
   #print('False positive  : ', fp2)
   # print('False negative  : ', fn2)
    print('Recall          : ', recall2)
    print('Precision       : ', precision2)
    print('F1-score        : ', F1_score2)
    
    gc.collect()
    

    df = pd.DataFrame([],columns=['Predicted_all','Predicted_micro','Predicted_macro',
                                  'Target_all','Target_micro','Target_macro','Reacll_all',
                                  'Precision_all','F1_score_all','TP_micro','Reacll_micro',
                                  'Precision_micro','F1_score_micro','TP_macro','Reacll_macro',
                                  'Precision_macro','F1_score_macro'])
    
    df['Predicted_all'] = [n]
    df['Predicted_micro'] = [n1]
    df['Predicted_macro'] = [n2]
    df['Target_all'] = [m]
    df['Target_micro'] = [m1]
    df['Target_macro'] = [m2]
    
    df['Reacll_all'] = [recall]
    df['Precision_all'] = [precision]
    df['F1_score_all'] = [F1_score]
    
    df['TP_micro'] = [tp1]
    df['Reacll_micro'] = [recall1]
    df['Precision_micro'] = [precision1]
    df['F1_score_micro'] = [F1_score1]
    
    df['TP_macro'] = [tp2]
    df['Reacll_macro'] = [recall2]
    df['Precision_macro'] = [precision2]
    df['F1_score_macro'] = [F1_score2]
    
    
    
    df.to_csv('./RESULT/TEST/[TEST]_data_'+name+'_model_'+MODEL + '_best_fold_'+str(SUB)+'.csv',index_label=False) 
    
    '''
    assert(m != 0)
    tp, fp = true_false_positive(pred_all_intervals,target_all_intervals)
    assert(fp == n-tp)
    fn = m - tp
    recall = tp / m
    try:
        precision = tp / n
    except:
        print("you have a problem here! check the predicted intervals ")
        precision = 0.0
    F_score = 2*tp/(m+n)
    '''
    """
    pred_micro_intervals,n2 = get_micro_intervals(outputs)
    pred_macro_intervals,n1 = get_macro_intervals(outputs)
    pred_all_intervals,n = get_all_intervals(outputs)
    
    gc.collect()
    
    target_micro_intervals,m2 = get_micro_intervals(labels)
    target_macro_intervals,m1 = get_macro_intervals(labels)
    target_all_intervals,m = get_all_intervals(labels)
    
    gc.collect()
    #Evaluation for Macro-expression
    assert(m1 != 0)
    tp1 = true_false_positive(pred_macro_intervals,target_macro_intervals)
    fp1 = n1 - tp1
    fn1 = m1 - tp1
    recall1 = tp1 / m1
    try:
        precision1 = tp1 / n1
    except:
        print("[MaE] you have a problem here! check the predicted intervals ")
        precision1 = 0.0
    try:
        F1_score1 = 2*(recall1*precision1)/(recall1+precision1)
    except:
        F1_score1 = 0.0
    
    print('*'*50)
    print('Spotting Macro-Expression performance')
    print('True positive   : ', tp1)
    print('False positive  : ', fp1)
    print('False negative  : ', fn1)
    print('Recall          : ', recall1)
    print('Precision       : ', precision1)
    print('F1-score        : ', F1_score1)
    
    gc.collect()
    
    #Evaluation for Micro-expression
    assert(m2 != 0)
    tp2 = true_false_positive(pred_micro_intervals,target_micro_intervals)
    fp2 = n2 - tp2
    fn2 = m2 - tp2
    recall2 = tp2 / m2
    try:
        precision2 = tp2 / n2
    except:
        print("[MiE] you have a problem here! check the predicted intervals ")
        precision2 = 0.0
    try : 
        F1_score2 = 2*(recall2*precision2)/(recall2+precision2)
    except:
        F1_score2 = 0.0
    
    print('*'*50)
    print('Spotting Micro-Expression performance')
    print('True positive   : ', tp2)
    print('False positive  : ', fp2)
    print('False negative  : ', fn2)
    print('Recall          : ', recall2)
    print('Precision       : ', precision2)
    print('F1-score        : ', F1_score2)
    
    gc.collect()
    
    #Evaluation for overall
    recall = (tp1+tp2)/(m1+m2)
    try:
        precision = (tp1+tp2)/(n1+n2)
    except:
        print("[Overall] you have a problem here! check the predicted intervals ")
        precision = 0.0
    try:
        F1_score = 2*(recall*precision)/(recall+precision)
    except:
        F1_score = 0.0
    
    print('*'*50)
    print('Spotting Macro/Micro-Expression overall performance')
    print('Recall          : ', recall)
    print('Precision       : ', precision)
    print('F1-score        : ', F1_score)
    
    gc.collect()
    """
    
    
    """
    def get_micro_intervals(outputs):
    intervals = []
    count = 0
    for output in outputs:
        dis = 0
        pos = 0
        interval = []
        for n,i in enumerate(output):
            if i == 1:
                dis += 1
                pos = n - dis + 1
            else:
                if dis != 0:
                    count += 1
                    interval += [(int(output[pos]),pos,pos+dis-1)] # (1,pos_mis,pos_max)
                    dis = 0
                    pos = 0
        intervals += [interval]
    return intervals, count

def get_macro_intervals(outputs):
    intervals = []
    count = 0
    for output in outputs:
        dis = 0
        pos = 0
        interval = []
        for n,i in enumerate(output):
            if i == 2:
                dis += 1
                pos = n - dis + 1
            else:
                if dis != 0:
                    count += 1
                    interval += [(int(output[pos]),pos,pos+dis-1)] # (2,pos_mis,pos_max)
                    dis = 0
                    pos = 0
        intervals += [interval]
    return intervals, count
"""
"""
def get_all_intervals(outputs):
    intervals = []
    count = 0
    intervals_mi = []
    count_mi = 0
    for output in outputs:
        dis = 0
        pos = 0
        interval = []
        interval_mi = []
        for n,i in enumerate(output):
            if i != 0:
                dis += 1
                pos = n - dis + 1
            else:
                if dis != 0:
                    count += 1
                    interval += [(int(output[pos]),pos,pos+dis-1)] # (class,pos_mis,pos_max)
                    dis = 0
                    pos = 0
                    if dis <= 15:
                        count_mi += 1
                        interval_mi += [(int(output[pos]),pos,pos+dis-1)] # (class,pos_mis,pos_max)
        intervals += [interval]
        intervals_mi += [interval_mi]
    return intervals, count, intervals_mi, count_mi
"""
