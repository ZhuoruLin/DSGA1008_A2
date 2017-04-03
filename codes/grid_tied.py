import os
emb_hid_sizes = [100,300,500,1000,1500]
epochs = 30
bptts = [20,25,30,35]
for model in models:
    for size in emb_hid_size:
        for bptt in bptts:
            save_path = '../../datasets/saved_models/%s_%s.pt' %(size,bptt)
            info_path = '../../datasets/info_dict/%s_%s.pk' %(size,bptt)
            cmdLine = 'python main.py --cuda --tied --batch-size 200 --bptt %s --emsize %s --nhid %s\
            --save %s --infopath %s'%(bptt,model,size,size,save_path,info_path)
            print(cmdLine)
            os.system(cmdLine)