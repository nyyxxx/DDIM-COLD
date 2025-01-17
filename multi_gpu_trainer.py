import os,sys
get_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(get_path)
sys.path.append('/home/ailanyy1/anaconda3/lib/python3.8/site-packages')
from diffusion_loader import ColdDownSampleDataset   #nyy change dataset
from ViT_draft2drawing import DiffusionVisionTransformer
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

def printLog(string,fileName):
    f = open(fileName,'a')
    f.write(string+'\n')
    f.close()
    #print(string)
    return 0

def init_process_group(world_size, rank):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:16666',
        world_size=world_size,
        rank=rank)

def evaluate(model, dataloader, device):
    model.eval()
    datagenerator = iter(dataloader)
    loss_arr = []
    for _ in range(len(dataloader)):
        noisy_img,img,t = next(datagenerator)
        noisy_img = noisy_img.to(device)
        img = img.to(device)
        t = t.to(device)
        with torch.no_grad():
            denoised_img = model(noisy_img,t)
            loss = F.smooth_l1_loss(denoised_img,img)
        loss_arr.append(loss.item())
    return np.array(loss_arr).mean()

def main(
    rank, world_size,
    initializing,amp,batch_size,epoch_num,lr,
    resume,datadir,SavedDir,log,CheckpointDir,
    image_size,diff_step,patch_size,embed_dim,depth,head):

    loss_rec = 5.0
    steps = 0
    best_loss = 5.0

    init_process_group(world_size, rank)
       
    train_set = ColdDownSampleDataset(datadir[0],imgSize=image_size) #nyy change dataset
    test_set = ColdDownSampleDataset(datadir[1],imgSize=image_size)  #nyy change dataset
    train_sampler = DistributedSampler(train_set,world_size,rank,True,seed=42,drop_last=True)
    test_sampler = DistributedSampler(test_set,world_size,rank,False,drop_last=False)
    train_dataloader = DataLoader(train_set,batch_size=batch_size,sampler=train_sampler,num_workers=8)
    test_dataloader = DataLoader(test_set,batch_size=batch_size,sampler=test_sampler,num_workers=8)
    train_batchs = len(train_dataloader)
    test_batchs = len(test_dataloader)
    device = torch.device('cuda:{:d}'.format(rank))
    torch.cuda.set_device(device)
    model = DiffusionVisionTransformer(img_size=image_size,patch_size=patch_size,embed_dim=embed_dim,depth=depth,num_heads=head)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    if rank == 0 and not(os.path.isfile(SavedDir+initializing)):
        torch.save(model.state_dict(),SavedDir+initializing)
        time.sleep(5)
    else:
        time.sleep(5)
    try:
        state = torch.load(SavedDir+initializing)
        model.load_state_dict(state,strict=True)
    except Exception as e:
        print(e)
    if rank == 0:
        localtime = time.asctime( time.localtime(time.time()))
        printLog(f'Date: {localtime}',log)
        printLog("TrainSet batchs:"+str(train_batchs),log)
        printLog("TestSet batchs:"+str(test_batchs),log)
            
    # Assume the GPU ID to be the same as the process ID
    model = DDP(model, device_ids=[device], output_device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epoch_num[1], steps_per_epoch=train_batchs, pct_start=5/epoch_num[1],div_factor=1e4,cycle_momentum=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,train_batchs*epoch_num[1],0.0,verbose=False)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.6,patience=3,min_lr=0.0,verbose=True)
    if not(resume == 'none'):
        ckpt = torch.load(resume)
        epoch_num[0] = ckpt['epoch'] + 1
        loss_rec = ckpt['loss_rec']
        if rank == 0:
            printLog(f'resuming from epoch {epoch_num[0]:8d} of '+resume,log)
        scheduler.load_state_dict(ckpt['scheduler'])
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        steps = ckpt['steps']
        best_loss = ckpt['metric']
        if rank == 0:
            printLog(f'recovering best_loss {best_loss:4f}',log)
    time_start=time.time()   
    writer = SummaryWriter()  #nyy tensorboard     
    for epoch in range(*epoch_num):
        model.train()
        # The line below ensures all processes use a different
        # random ordering in data loading for each epoch.
        train_sampler.set_epoch(epoch)
        train_generator = iter(train_dataloader)                                                             
        for _ in range(len(train_dataloader)):  
                                                             #NOTE看到这
            noisy_img,img,t = next(train_generator)
            noisy_img = noisy_img.to(device)
            img = img.to(device)
            t = t.to(device)

            with torch.cuda.amp.autocast(enabled=amp):
                denoised_img = model(noisy_img,t)             #BUG CUDA out of memory
                loss = F.smooth_l1_loss(denoised_img,img)
                print(loss)                
            loss_rec = loss_rec*0.99+loss.item()*0.01
            optimizer.zero_grad(set_to_none=False)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            scheduler.step()
            if steps % 100 == 0 and rank == 0:
                time_end=time.time()
                printLog(f'steps: {steps:8d} loss: {loss_rec:.4f} time_cost: {time_end-time_start:.2f}',log) 
                time_start=time.time()
        del noisy_img,img,t,loss
        
        test_sampler.set_epoch(epoch)
        vloss = evaluate(model,test_dataloader,device)#gamma_array[epoch])
        vloss =  torch.tensor(vloss,device=device)
        dist.all_reduce(vloss, op=dist.ReduceOp.SUM)
        vloss /= torch.distributed.get_world_size()
        vloss = vloss.cpu().numpy()
        #scheduler.step(vloss)
        if rank == 0:
            #print(np.around(val_hist).astype(np.int32))
            printLog(f'epoch: {epoch:4d}    loss: {vloss:.5f}    time:{time.asctime(time.localtime(time.time()))}',log) 
            writer.add_scalar("loss",vloss,epoch)                  #nyy tensorboard
            if vloss<best_loss:
                best_loss = vloss
                torch.save(model.module.state_dict(),CheckpointDir+"bestloss.pkl")
            torch.save({
                        'epoch': epoch,
                        'steps': steps,
                        'loss_rec': loss_rec,
                        'metric': best_loss,
                        'state_dict': model.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }, CheckpointDir+"lastepoch.pkl")  
    writer.close()           
    dist.destroy_process_group()

if __name__ == '__main__':
    import torch.multiprocessing as mp
    import os,sys
    import yaml
    import shutil
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    ExpName = sys.argv[1]
    #ExpName = '20220822'
    with open(current_path+'/'+ExpName+'.yaml') as f:
        training_parameters = yaml.full_load(f)

    TrainDir = training_parameters['dataStorage']
    SavedDir=current_path+"/Saved_Models/"

    try:
        os.mkdir(SavedDir)
    except:
        pass
    initializing = training_parameters["initializing"]
    amp = training_parameters['AMP']
    modelName = training_parameters["framework"]
    num_gpus = training_parameters["num_gpus"]
    resume = training_parameters["resume"]
    if amp:
        batch_size=training_parameters['batch_size']*2
    else:
        batch_size = training_parameters['batch_size']
    epoch_num = training_parameters["epoch"]
    lr = training_parameters["base_lr"]*batch_size*num_gpus/512
    try:
        os.mkdir(SavedDir+ExpName+modelName)
    except:
        print('Warning!Current folder already exist!')
    shutil.copy(current_path+'/'+ExpName+'.yaml',SavedDir+ExpName+modelName)
    CheckpointDir=SavedDir+ExpName+modelName+"/"
    log=SavedDir+ExpName+modelName+"/train.log"
    
    image_size = training_parameters["image_size"]
    diff_step = training_parameters["diff_step"] 
    patch_size = training_parameters["patch_size"] 
    embed_dim = training_parameters["embed_dim"] 
    depth = training_parameters["depth"] 
    head = training_parameters["head"]    

    torch.multiprocessing.set_start_method('spawn')
    procs = []
    for rank in range(num_gpus):
        p = mp.Process(target=main, args=(rank,num_gpus,initializing,amp,batch_size,epoch_num,lr,resume,TrainDir,SavedDir,log,CheckpointDir,image_size,diff_step,patch_size,embed_dim,depth,head))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
