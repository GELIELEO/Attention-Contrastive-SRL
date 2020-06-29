import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import os


def learner(model, 
            storage, 
            mini_batch, 
            epochs, 
            num_iter,
            num_workers, 
            device,
            queue, 
            ready_to_works, 
            exit_flag
            ):

    print('Learner starting...')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), 
    #                             lr=0.0003)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000000], gamma=0.1)
    writer = SummaryWriter()
    
    path = './models'
    if not os.path.exists(path):
        os.mkdir(path)

    _ = [e.set() for e in ready_to_works]
    
    
    # switch to train mode
    model.train()

    for epoch in range(epochs):
        for i in range(num_workers):
            id = queue.get()
            print(f'Learner recieves worker-{id} done signal and reaches {i}th wokers')  

        batch_gen = storage.batch_generator(mini_batch) 
        
        for i in range(num_iter):
            for batch in batch_gen:
                obs_1, obs_2 = batch
                # print(obs.shape, act.shape, rst.shape)
                obs_1 = torch.autograd.Variable(obs_1).to(device)
                obs_2 = torch.autograd.Variable(obs_2).to(device)

                output, target = model(obs_1, obs_2)
                # print('output', output)
                loss = criterion(output, target)
                # print('target, output', target.dtype, output.dtype)
                
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
        if epoch == epochs-1:
            # set exit flag to 1, and notify workers to exit
            exit_flag.value = 1
        _ = [e.set() for e in ready_to_works]

        lr_scheduler.step()
        print('>>>>>>>>>>>>>>>>>>>> Current learning rate:',  optimizer.param_groups[0]['lr'])
        
        writer.add_scalar('loss', loss, epoch)
        if epoch%200 == 0:
            torch.save(model.state_dict(), os.path.join(path, 'model'+str(epoch)+'.pth'))
            torch.save(model.encoder_q.state_dict(), os.path.join(path, 'query'+str(epoch)+'.pth'))
                    