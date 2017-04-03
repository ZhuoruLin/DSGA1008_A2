import os
for bptt in np.arange(20,45,5):
    save_path = '../../datasets/$sbptt.pt' %(bptt)
    info_path = '../../datasets/%sbptt.pk' %(bptt)
    cmdLine = 'python main.py --cuda --save %s --infopath %s'%(model,nlayer,emsize,nhid,save_path,info_path)
    print(cmdLine)
    os.system(cmdLine)