import os
import numpy as np
import torch
from torch._C import default_generator
import torch.nn.functional as F
from torch import nn
import _dwp.utils as utils
from _dwp.logger import Logger
import _dwp.myexman
from _dwp.my_utils import save_args_params

def train(trainloader, testloader, lvae, optimizer, args):
    logger = Logger(name='logs', base=args.root)
    best_loss = 11e8
    for epoch in range(1, args.num_epochs + 1):

        #set learning rate
        adjust_learning_rate(optimizer, lr_linear(epoch-1))

        train_KLD = utils.MovingMetric()
        train_recon_loss = utils.MovingMetric()
        train_loss = utils.MovingMetric()

        for i, x in enumerate(trainloader):
            optimizer.zero_grad()
            x = x.to(lvae.device)
            _ , recon_loss, kl_loss = lvae(x) 
            
            avg_KLD = kl_loss
            loss = recon_loss + avg_KLD

            loss.backward()
            optimizer.step()

            train_KLD.add(avg_KLD.item(), 1)
            train_recon_loss.add(recon_loss.item(), 1)
            train_loss.add(loss.item(),1)

        test_KLD = utils.MovingMetric()
        test_recon_loss = utils.MovingMetric()
        test_loss = utils.MovingMetric()

        for i, x in enumerate(testloader):
            x = x.to(lvae.device)
            _ , recon_loss, kl_loss = lvae(x)
           
            avg_KLD = kl_loss
            loss = recon_loss + avg_KLD

            test_KLD.add(avg_KLD.item(), 1)
            test_recon_loss.add(recon_loss.item(), 1)
            test_loss.add(loss.item(),1)
        
        train_KLD = train_KLD.get_val()
        train_recon_loss = train_recon_loss.get_val()
        train_loss = train_loss.get_val()
        test_KLD = test_KLD.get_val()
        test_recon_loss = test_recon_loss.get_val()
        test_loss = test_loss.get_val()

        logger.add_scalar(epoch, 'train_KLD', train_KLD)
        logger.add_scalar(epoch, 'train_recon_loss', train_recon_loss)
        logger.add_scalar(epoch, 'train_loss', train_loss)

        logger.add_scalar(epoch, 'test_KLD', test_KLD)
        logger.add_scalar(epoch, 'test_recon_loss', test_recon_loss)
        logger.add_scalar(epoch, 'test_loss', test_loss)

        logger.iter_info()
        logger.save()

        if (epoch-1) % 10 == 0:
            torch.save(lvae.state_dict() , os.path.join(args.root, 'lvae_params_epoch_{}.torch'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(args.root, 'opt_params_epoch_{}.torch'.format(epoch)))

        is_best = (test_loss < best_loss)
        if is_best:
            best_loss = test_loss
            torch.save(lvae.state_dict(), os.path.join(args.root, 'lvae_params.torch'))   
            if args.add_save_path : 
                torch.save(lvae.state_dict(), os.path.join(args.add_save_path, 'lvae_params.torch'))  

    torch.save(lvae.state_dict(), os.path.join(args.root, 'lvae_params_lastepoch.torch'))
    torch.save(optimizer.state_dict(), os.path.join(args.root, 'opt_params_lastepoch.torch'))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch):
    '''sets the learning rate in a way that it is lr at epoch 0 and linearily 
    decreases to 0 at args.num_epochs. After num_epochs the lr is constantly zero'''
    lr = args.lr * np.minimum((-epoch) * 1. / (args.num_epochs) + 1, 1.)
    return max(0, lr)


if __name__ == '__main__':
    parser = myexman.ExParser(file=__file__)
    #train settings 
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_id', default='0')

    #optimisation
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', default=1e-4, type=float)
    
    #evaluation
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--test_bs', default=512, type=int)
    parser.add_argument('--verbose', default=1, type=int)

    #model specifics
    parser.add_argument('--dims', default=[100,50,10], nargs='*', type=float)
    parser.add_argument('--z_dim', default=2, type=int)
    
    #misc
    parser.add_argument('--add_save_path', default='')

    args = parser.parse_args()

    #save the args to the dict, from where the vaes are initialised
    save_args_params(args,args.add_save_path)

    #et GPU, device and seeds
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #get dataloaders
    if args.data_dir:
        trainloader, D = utils.get_dataloader(os.path.join(args.data_dir, 'train.npy'), args.batch_size, shuffle=True)
        testloader, D = utils.get_dataloader(os.path.join(args.data_dir, 'test.npy'), args.test_bs, shuffle=False)
    else:
        trainloader, D = utils.get_dataloader(args.train, args.batch_size, shuffle=True)
        testloader, D = utils.get_dataloader(args.test, args.test_bs, shuffle=False)

    dims = [int(el) for el in args.dims]
    lvae = LVAE(dims,args.z_dim,device)
    
    #configure optimisation
    optimizer = torch.optim.Adam(lvae.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    train(trainloader, testloader, lvae, optimizer, args)
