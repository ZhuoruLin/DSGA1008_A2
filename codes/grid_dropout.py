import os
emsize = 300
nhids = 300
nlayers = 2
epochs = 30
model = 'LSTM'
for pdropout in np.arange(0.5,1,0.1):
    save_path = '../../datasets/%s_%s_%s_%s_%sepcs_%sdrop.pt' %(model,nlayer,emsize,nhid,epochs,pdropout)
    info_path = '../../datasets/%s_%s_%s_%s_%sepcs_%sdrop.pk' %(model,nlayer,emsize,nhid,epochs,pdropout)
    cmdLine = 'python main.py --cuda --pdropout %s --epochs %s --batch-size 200 --model %s --nlayers %s --emsize %s --nhid %s\
    --save %s --infopath %s'%(pdropout,epochs,model,nlayer,emsize,nhid,save_path,info_path)
    print(cmdLine)
    os.system(cmdLine)