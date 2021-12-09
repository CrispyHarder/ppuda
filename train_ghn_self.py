import os
import numpy as np
import torch
import _dwp.utils as utils
from _dwp.logger import Logger
import _dwp.myexman as myexman
from ppuda.ghn.nn import GHN
from ppuda.utils.utils import default_device

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



if __name__ == '__main__':
    parser = myexman.ExParser(file=__file__)
    #train settings 
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu_id', default='0')

    #optimisation
    parser.add_argument('--lr', default=1e-3, type=float)
    #maybe add lr scheduler later 
    
    #evaluation
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--test_bs', default=512, type=int)
    parser.add_argument('--verbose', default=1, type=int)

    #model specifics
    parser.add_argument('--decoder', type=str, default='conv', choices=['mlp', 'conv', 'dwp'],
                                help='decoder to predict final parameters')

    #misc
    #nothing yet 

    args = parser.parse_args()

    #get GPU, device and seeds
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #get dataloaders for image data 
    if args.data_dir:
        trainloader, D = utils.get_dataloader(os.path.join(args.data_dir, 'train.npy'), args.batch_size, shuffle=True)
        testloader, D = utils.get_dataloader(os.path.join(args.data_dir, 'test.npy'), args.test_bs, shuffle=False)
    else:
        trainloader, D = utils.get_dataloader(args.train, args.batch_size, shuffle=True)
        testloader, D = utils.get_dataloader(args.test, args.test_bs, shuffle=False)

    #get the model 
    ghn = GHN(max_shape=(64,64,3,3),
              num_classes=10,
              hypernet='gatedgnn',
              decoder=args.decoder,
              weight_norm=True,
              ve=True,
              layernorm=True,
              hid=32,
              debug_level=1).to(device)

    #get the resnets with their graphs 
    graphs = GraphBatch([Graph(nets_torch, ve_cutoff=50 if self.ve else 1)])
                graphs.to_device(self.embed.weight.device)

    
    #configure optimisation
    optimizer = torch.optim.Adam(ghn.parameters(), lr=args.lr) 

    train(trainloader, testloader, ghn, optimizer, args)
