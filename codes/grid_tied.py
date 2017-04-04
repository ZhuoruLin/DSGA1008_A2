import os
import numpy as np
emb_hid_sizes = [300,500]
pdropouts = [0.0,0.2,0.5]
model = 'LSTM'
for size in emb_hid_sizes:
	for pdropout in pdropouts:
	    save_path = '../../datasets/saved_models/%s_%s_%s_tied_%s.pt' %(model,size,size,pdropout)
	    info_path = '../../datasets/info_dict/%s_%s_%s_tied_%s.pk' %(model,size,size,pdropout)
	    cmdLine = 'python main.py --cuda --tied --batch-size 20 --pdropout %s --emsize %s --nhid %s\
	    --save %s --infopath %s'%(pdropout,size,size,save_path,info_path)
	    print(cmdLine)
	    os.system(cmdLine)

