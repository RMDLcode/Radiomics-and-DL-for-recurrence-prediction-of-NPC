# coding:utf-8
from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
import random
from models import UNet3D

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
    
def Recompone_result(result, information):
        #information[self.padding_shape[0], self.cut_stride, self.ori_shape[0],self.ori_shape[1], self.ori_shape[2], self.n_labels]
        patch_s = result.shape[2]
        N_patches_img = torch.div((information[0][0] - patch_s), information[0][1], rounding_mode='floor')+1

        assert (result.shape[0] == N_patches_img)
        
        full_prob = torch.zeros((information[0][5], information[0][0],information[0][3], information[0][4])).to(device)
        full_sum = torch.zeros((information[0][5], information[0][0], information[0][3], information[0][4])).to(device)

        for s in range(N_patches_img):
            full_prob[:, s * information[0][1]:s * information[0][1] + patch_s] += result[s]
            full_sum[:, s * information[0][1]:s * information[0][1] + patch_s] += 1

        #assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum

        assert (torch.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (torch.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        img = final_avg[:, :information[0][2], :information[0][3], :information[0][4]]

        return img.unsqueeze(0)    
        
def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx,(data, target, information,idd) in tqdm(enumerate(val_loader),total=len(val_loader)):
            result=None
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)#转onehot
            #information[self.padding_shape[1], self.cut_stride, self.ori_shape[0],self.ori_shape[1], self.ori_shape[2], self.n_labels]
            data, target, information = data.to(device), target.to(device), information.to(device)
            for k in range(data.size(0)):
                inputd = data[k]
                output = model(inputd)

                if result is not None:
                    result = torch.cat((result, output), dim=0)
                else:
                    result = output
            #print(result.shape)

            pred = Recompone_result(result, information)

            loss=loss_func(pred, target, idd)
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(pred, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice': val_dice.avg[1]})
    return val_log

def train(model, train_loader, optimizer, loss_func, n_labels):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target, data_name) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target,n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output[3], target, data_name)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(),data.size(0))
        train_dice.update(output[3], target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice': train_dice.avg[1]})
    return val_log

def draw_figure(save_path):
    train_log_csv = pd.read_csv(os.path.join(save_path, 'train_log.csv'))
    x_axis = train_log_csv.values[:, 0].astype(np.int32)
    Train_Loss_axis = train_log_csv.values[:, 1]
    Train_dice_axis = train_log_csv.values[:, 2]
    Val_Loss_axis = train_log_csv.values[:, 3]
    Val_dice_axis = train_log_csv.values[:, 4]
    plt.title('training process')
    plt.plot(x_axis, Train_Loss_axis, '-', color='r', label='Train_Loss_axis')
    plt.plot(x_axis, Val_Loss_axis, '-', color='g', label='Val_Loss_axis')
    plt.plot(x_axis, Train_dice_axis, '-', color='b', label='Train_dice_axis')
    plt.plot(x_axis, Val_dice_axis, '-', color='y', label='Val_dice_axis')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Loss and Dice')
    plt.savefig(os.path.join(save_path,'process.png'))
    plt.clf()
    
if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda:%s'%args.gpu_id[0])
    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args),batch_size=args.batch_size,num_workers=args.n_threads, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=Val_Dataset(args),batch_size=1,num_workers=args.n_threads, shuffle=False, drop_last=False)

    # model info
    model = UNet3D.UNet3D(in_channels=2, out_channel=args.n_labels, training=True).to(device)

    #The default optimizer is SGD
    optimizer = optim.SGD(model.parameters(), momentum=0.99, lr=args.lr, weight_decay=1e-4, nesterov=True)

    start_epoch = 0

    #kaiming initialize
    model.apply(weights_init.init_model)
    
    common.print_network(model)
    loss = loss.DCLoss(n=1)

    log = logger.Train_Logger(save_path,"train_log")

    best = [0,0,0] 
    trigger = 0
    epoch_alpha = start_epoch
    val_alpha = 0.9
    val_cumulate_dice = 0

    for epoch in range(start_epoch+1, args.epochs + 1):
        #common.adjust_learning_rate(optimizer, epoch, args, num_loops=80)
        common.adjust_learning_rate_V2(optimizer, epoch, args, num_loops=args.epochs+1)#
        train_log = train(model, train_loader, optimizer, loss, args.n_labels)
        state = {'state_dict': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch,train_log,val_log)
        draw_figure(save_path)

        
        if val_cumulate_dice==0:
          val_cumulate_dice = val_log['Val_dice']
          val_cumulate_dice_old = val_cumulate_dice
        else:
          val_cumulate_dice_old = val_cumulate_dice
          val_cumulate_dice = val_alpha * val_cumulate_dice_old + (1 - val_alpha) * val_log['Val_dice']
        # Save checkpoint.

        trigger += 1
        quanzhong = 0.9
        if val_cumulate_dice>val_cumulate_dice_old:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice']
            best[2] = train_log['Train_dice']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))