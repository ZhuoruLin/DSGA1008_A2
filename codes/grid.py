import os
emsizes = [100,500,1000]
nhids = [100,500,1000]
nlayers = [2,3]
models = ['LSTM','GRU']
for model in models:
    for emsize in emsizes:
        for nhid in nhids:
            for nlayer in nlayers:
                save_path = 'saved_models/%s_%s_%s_%s.pt' %(model,nlayer,emsize,nhid)
                info_path = 'info_dict/%s_%s_%s_%s.pk' %(model,nlayer,emsize,nhid)
                cmdLine = 'python main.py --cuda --epochs 20 --batch-size 200 --model %s --nlayers %s --emsize %s --nhid %s\
                --save %s --infopath %s'%(model,nlayer,emsize,nhid,save_path,info_path)
                print(cmdLine)
                os.system(cmdLine)