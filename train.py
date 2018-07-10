from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True 

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import datasets_expanded as datasets
import net
import matplotlib.pyplot as plt

from util import cdf, simulate_args_from_namespace


def train(models, train_loader, optimizers, sisr_thres=np.inf):
    for _, model in models.items():
        model.train()
    model_losses = [0]*3
    for idx, sample_batch in enumerate(tqdm(train_loader, leave=False)):
        if torch.cuda.is_available():
            ref_data, target = sample_batch['ref_angles'].cuda(), sample_batch['target'].cuda()
        ref_data, target = Variable(ref_data),  Variable(target)


        for optimizer in optimizers:
            optimizer.zero_grad()

        ref_predictions, _ = models['adhoc_model'](ref_data)

        # ref_loss = F.mse_loss(ref_predictions, target)
        ref_loss = F.l1_loss(ref_predictions, target)

        ref_loss = ref_loss.clamp(max=sisr_thres)

        ref_loss.backward()


        model_losses[0] += ref_loss#.detach().cpu().numpy()

        for optimizer in optimizers:
            optimizer.step()

    N = len(train_loader.dataset)
    print('Train Loss:: ref: {:.4f}'.format(model_losses[0] / N ))

    return model_losses



def test(models, test_loader):
    for _, model in models.items():
        model.eval()
    model_losses = [0]*2
    model_acc = [[], []]
    pdist = torch.nn.PairwiseDistance(p=1)

    for idx, sample_batch in enumerate(tqdm(test_loader, leave=False)):
        if torch.cuda.is_available():
            ref_data, target = sample_batch['ref_angles'].cuda(), sample_batch['target'].cuda()
        ref_data, target = Variable(ref_data),  Variable(target)


        ref_predictions, _ = models['adhoc_model'](ref_data)
        # ref_loss = F.mse_loss(ref_predictions, target)
        ref_loss = F.l1_loss(ref_predictions, target)
       
        model_losses[0] += ref_loss

        model_acc[0].append(pdist(ref_predictions, target).detach().cpu().numpy() )




    N = len(test_loader.dataset)
   
    print('Test  Loss:: ref: {:.4f}'.format(model_losses[0] / N))
    print('Test  Acc:: ref-mean: {:.4f} ref-median: {:.4f} ref-max[{}]: {:.4f}'.format(np.mean(model_acc[0]),  
                                                                                   np.median(model_acc[0]),
                                                                                   np.argmax(model_acc[0]),
                                                                                   np.max(model_acc[0])))

    return model_losses, model_acc


def train_model(models, model_path, train_loader, test_loader, lr, epochs, sisr_thres=np.inf):
    optimizers, schedulers = [], []
    for _, model in models.items():
        ps = filter(lambda x: x.requires_grad, model.parameters())
        optimizers.append(optim.SGD(ps, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4))
        schedulers.append( optim.lr_scheduler.CosineAnnealingLR(optimizers[-1], epochs))



    for epoch in range(1, epochs):
        print('[Epoch {}]'.format(epoch))
        train(models, train_loader, optimizers, sisr_thres=sisr_thres)
        model_losses, model_acc = test(models, test_loader)
        if epoch % 10 == 9:
            print("Outputing model")
            torch.save(models['adhoc_model'], model_path)
        for scheduler in schedulers:
            scheduler.step()
    
    # cdf(np.hstack(model_acc[0]))
    # plt.savefig("figs/hist.png")
    # plt.close()
    return model_acc[0]

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Example')
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
    parser.add_argument('--sisr-thres', type=float, default=np.inf, metavar='RS',
                        help='sisr-thres - (default: inf)')
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

    if args.rejectionsampling == 0:
        args.rejectionsampling = False
    elif args.rejectionsampling == 1:
        args.rejectionsampling = True
    else:
        assert False, "Invalid choice for rejection sampling"


    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_set, train_loader, test_set, test_loader = datasets.get_randomref_dataset("", 
                                                                                    args.batch_size, 
                                                                                    {'train' : args.train_size,
                                                                                     'test' : args.test_size},
                                                                                    args.numreferencenodes, 
                                                                                    args.quantization, 
                                                                                    args.rejectionsampling,
                                                                                    args.noise_scale,
                                                                                    args.ref_scale)

    adhoc_model = net.MLP(2*args.numreferencenodes, 1, activation=True)
    models = {'adhoc_model' : adhoc_model}

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    out_str = "&".join(simulate_args_from_namespace(args, positional=['output']))
    out_str = args.output+out_str+".pth"  
    print(out_str)
    if args.cuda:
        for _, model in models.items():
            model = model.cuda()
    model_acc = train_model(models, out_str, train_loader, test_loader, args.lr, args.epochs, 
                            sisr_thres=args.sisr_thres)