import os
emsizes = [100,500,1000]
nhids = [100,500,1000]
nlayers = [2,3]
models = ['LSTM']
epochs = 30
for model in models:
    for emsize in emsizes:
        for nhid in nhids:
            for nlayer in nlayers:
                save_path = 'saved_models/%s_%s_%s_%s_%sepcs.pt' %(model,nlayer,emsize,nhid,epochs)
                info_path = 'info_dict/%s_%s_%s_%s_%sepcs.pk' %(model,nlayer,emsize,nhid,epochs)
                cmdLine = 'python main.py --cuda --epochs %s --batch-size 200 --model %s --nlayers %s --emsize %s --nhid %s\
                --save %s --infopath %s'%(epochs,model,nlayer,emsize,nhid,save_path,info_path)
                print(cmdLine)
                os.system(cmdLine)