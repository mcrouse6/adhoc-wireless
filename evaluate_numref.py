import argparse
import os
import glob
from tqdm import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True 

from torch.autograd import Variable
import torch.nn.functional as F


import datasets
import net
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from util import cdf, simulate_args_from_namespace
#plt.ion()


def test(models, test_loader):
    for _, model in models.items():
        model.eval()
    model_losses = [0]*2
    model_acc = [[], []]
    pdist = torch.nn.PairwiseDistance(p=1)

    for idx, sample_batch in enumerate(tqdm(test_loader, leave=False)):
        if torch.cuda.is_available():
            ref_data, target = sample_batch['data'].cuda(), sample_batch['target'].cuda()
        ref_data, target = Variable(ref_data),  Variable(target)


        ref_predictions, _ = models['orientation_model'](ref_data)
        ref_loss = F.mse_loss(ref_predictions, target)
        # ref_loss = F.l1_loss(ref_predictions, target)
       
        model_losses[0] += ref_loss

        model_acc[0].append(pdist(ref_predictions, target).detach().cpu().numpy() )




    N = len(test_loader.dataset)
   
    print('Test  Loss:: ref: {:.4f}'.format(model_losses[0] / N))
    print('Test  Acc:: ref-mean: {:.4f} ref-median: {:.4f} ref-max[{}]: {:.4f}'.format(np.mean(model_acc[0]),  
                                                                                   np.median(model_acc[0]),
                                                                                   np.argmax(model_acc[0]),
                                                                                   np.max(model_acc[0])))

    return model_losses, model_acc

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--noise-scale', type=float, default=0.0, metavar='NS',
                        help='noise_scale - angle noise (default: 0.0)')
    parser.add_argument('--ref-scale', type=float, default=1.0, metavar='RS',
                        help='ref_scale - ref location scale (default: 1.0)')
    parser.add_argument('--output', default='models/',
                        help='output directory')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--quantization', type=float, default=0, metavar='Q',
                        help='Quantization (0, 1, 10)')
    parser.add_argument('--rejectionsampling', type=int, default=0, metavar='R',
                        help='use rejection sampling (0 for no/1 for yes)')
    parser.add_argument('--numreferencenodes', type=int, default=2, metavar='RN',
                        help='number of reference nodes')
    parser.add_argument('--train-size', type=int, default=1000, metavar='TS',
                        help='training dataset size')
    parser.add_argument('--test-size', type=int, default=1000, metavar='TS',
                        help='training dataset size')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    #model_paths = glob.glob("models/noise_refs_30/*")
    # model_paths = glob.glob("models/orientation_num_refs_noise_03/*")
    model_paths = glob.glob("models/orientation_num_refs_noise_05/*")

    loss_l, acc_l = [], []
    acc_l = dict()
    # plt.figure()
    for model_str in model_paths:
        args = model_str.split("&")
        args[-1] = args[-1].split(".")[0]
        print(args)

      
                                                                                    
        train_set, train_loader, test_set, test_loader = datasets.get_orientation_dataset("", 
                                                                                int(args[1]),
                                                                                    {'train' : int(args[-1]) ,
                                                                                     'test' : int(args[-3])},
                                                                                int(args[10]), 
                                                                                0, 
                                                                                noise_scale=0.05,
                                                                                ref_scale=1.0)



        adhoc_model = torch.load(model_str)
        models = {'orientation_model' : adhoc_model}

        _, acc = test(models, test_loader)
        acc_l[float(args[10])] = np.median(acc[0]) 
        # cdf(np.hstack(acc[0]))

    #plt.figure() 
    #plt.plot(acc_l.keys(), [v*360. for v in list(acc_l.values())], 'o')    
    plt.plot(sorted(acc_l), [v*360 for v in sorted(acc_l.values(), reverse=True)], '-o')    
    plt.xlabel('Number of Reference Nodes')
    plt.ylabel('Median Error in Degrees')
    plt.title("Median Orientation Error in Degrees")
    # plt.show()
    #plt.tight_layout()
    #plt.grid()
    #plt.savefig("figs/increasing_numref_.03.png", dpi=300)
    #plt.savefig("figs/increasing_numref_all.png", dpi=300)

    #plt.hold()

