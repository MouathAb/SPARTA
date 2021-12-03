#import pandas as pd
#import numpy as np

import os
import gc
import torch
torch.cuda.empty_cache()

print('*'*200)
print('\n\t\t\t\t\t\t\t\t\t\t\tRun it all in once :)\n')
print('*'*200)

#models = ['vit_s_16','vit_b_16','vit_res_b_16','tnt_s_16','t2t_14',
#         't2t_24','t2t_t_14','t2t_t_24','swin_s_p4w7','swin_b_p4w7','swin_l_p4w7']


    
models = ['vit_s_16']

#datas = ['fer13','sfew','rafdb']
data = 'casme2_lv'


print('\n')
print('\t\t\t\t\t\t\t\t\t\t\tdata : ', data)
print('\n')
print('/\\'*100)
#subject =  np.unique(pd.read_csv('./data/train/casme2_lv_train.csv').Subject.values)
for model in models: 
    print('\n')
    print('\t\t\t\t\t\t\t\t\t\t\tmodel : ', model)
    print('\n')
    print('/\\'*100)
    try:
        os.system('python3 ./src/models/from_timm2.py --window 10 -m '+model)
    except:
        print(' [FIX] Error on building some models ...')
        
    for s in range(1,23):
        print('\n')
        print('\t\t\t\t\t\t\t\t\t\t\tsubject : ', s)        
        print('\n')
        os.system('python3 ./src/train.py --epoch 50 --data '+data+' -m '+model+' -se False -sub '+str(s) +' --window 10')
        gc.collect()
        os.system('python3 ./src/test.py  --data '+data+'  -m '+model+' -se False -sub '+str(s) +' --window 10')
        
        #os.system('python src2/train.py --epoch 15 --data '+data+' --scheduler nope -m '+model+' -se True')
        print('\n')
        print('#'*200)

    print('\n')
    print('/\\'*100)
