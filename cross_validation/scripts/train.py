import numpy as np
import time

import torch
import torch.optim as optim
import torch.nn.functional as F

def train(net, loader, I, checkpoint_path, save_step, class_weights, I_reduce_lr):

    net.train(True)
    torch.backends.cudnn.benchmark = True # Faster training

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    losses = np.zeros(I)

    start_time = time.time()

    #
    i = 0
       
    # 
    while i < I:

        for X, Y, name in loader:

            if i >= I:
                # if this causes exceptions, use continue instead
                break


            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True)

            output = net(X)

            loss = F.cross_entropy(output, Y, weight=class_weights.cuda())
            losses[i] = loss.item()
            
            print("Training iteration {}: loss: {}".format(i, losses[i]))

            # Perform update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save checkpoints
            if i > 0 and ((i+1) % save_step) == 0:
                print("Storing snapshot")

                state = {
                    'iteration': i+1,
                    'state_dict': net.state_dict(),
                    'optimizer' : optimizer.state_dict()}

                index = "00000{}".format(int(i)+1)[-6:]
                torch.save(state, checkpoint_path + "iteration_{}.pth.tar".format(index))

            # Final learning rate reduction
            if i == (I_reduce_lr - 1):
             
                print("Reducing learning rate by factor 10")
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

            i = i + 1

    end_time = time.time()

    print("Elapsed training time: {}".format(end_time - start_time))

    return (end_time - start_time)
