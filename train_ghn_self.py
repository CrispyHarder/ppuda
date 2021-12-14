import os
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import _dwp.utils as utils
from _dwp.logger import Logger
import _dwp.myexman as myexman
from ppuda.utils.darts_utils import accuracy
from models.resnet import resnet20,resnet32,resnet44, resnet56
import time
from torch.optim.lr_scheduler import MultiStepLR

if __name__ == '__main__':
    parser = myexman.ExParser(file=__file__)
    #train settings 
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_split_seed', default=42, type=int)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--ghn_from_start', action='store_true', default=False)

    #optimisation
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr_steps', type=str, default='200,250', help='epochs when to decrease lr')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    
    #evaluation
    parser.add_argument('--test_bs', default=512, type=int)
    parser.add_argument('--verbose', default=1, type=int)

    #model specifics
    parser.add_argument('--decoder', type=str, default='conv', choices=['mlp', 'conv', 'dwp', 'conv3'],
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

    #now import packages who might need to have device set before hand 
    from ppuda.ghn.nn import GHN
    from ppuda.deepnets1m.graph import Graph, GraphBatch
    from ppuda.ghn.decoder import Conv3Decoder, ConvDecoder

    #get dataloaders for image data 
    trainloader, testloader = utils.load_dataset(data='cifar', train_bs=args.batch_size, test_bs=args.test_bs,
                                             num_examples=None, seed=args.data_split_seed)

    #load/init the model 
    if not args.ghn_from_start:
        from ppuda.ghn.nn import GHN2
        ghn = GHN2('cifar10')
    else:
        from ppuda.ghn.nn import GHN 
        ghn = GHN(max_shape=(64,64,3,3),
                    num_classes=19,
                    hypernet='gatedgnn',
                    decoder=args.decoder,
                    weight_norm=True,
                    ve=True,
                    layernorm=True,
                    hid=32,
                    debug_level=1).to(device)
    
    if args.decoder == 'conv':
        hid = ghn.hid
        ghn.decoder = ConvDecoder(in_features=hid,
                            hid=(hid * 4, hid * 8),
                            out_shape=(64,64,3,3),
                            num_classes=10)
    elif args.decoder == 'conv3':
        ghn.decoder = Conv3Decoder(out_shape=(64,64,3,3),num_classes=10)
    else:
        raise NotImplementedError(f'no {args.decoder} decoder')
    ghn = ghn.to(device)
    ghn.train()
    

    #get the resnets with their graphs
    res20, res32, res44, res56 = resnet20(), resnet32(), resnet44(), resnet56()
    models = [res32, res44, res56]
    
    graphs = GraphBatch([Graph(model, ve_cutoff=50) for model in models])
    res20_graph = GraphBatch([Graph(res20, ve_cutoff=50)])

    graphs.to_device(device)
    res20_graph.to_device(device)

    for model in models:
        model = model.to(device)
    res20 = res20.to(device)

    
    #configure optimisation
    optimizer = torch.optim.Adam(ghn.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma)

    #Training part 
    logger = Logger(name='logs', base=args.root)
    best_loss = 11e8
    for epoch in range(1, args.num_epochs + 1):
        
        train_loss = utils.AvgrageMeter()
        val_loss = utils.AvgrageMeter()
        train_top1 = utils.AvgrageMeter()
        val_top1 = utils.AvgrageMeter()
        train_top5 = utils.AvgrageMeter()
        val_top5 = utils.AvgrageMeter()
        val_res_top1 = utils.AvgrageMeter()

        start_epoch = time.time()

        for i,(images,labels) in enumerate(trainloader):
            logits = 0
            loss = 0
            count = 0

            optimizer.zero_grad()

            models_pred = ghn(models,graphs)

            images = images.to(device)
            labels = labels.to(device)

            for i,model in enumerate(models_pred):
                y = model(images)
                loss += F.cross_entropy(y, labels)
                logits += y 
                count += 1 
            
            loss = loss/count 
            logits = logits/count

            loss.backward()

            parameters = []
            for group in optimizer.param_groups:
                parameters.extend(group['params'])

            nn.utils.clip_grad_norm_(parameters, 5)
            optimizer.step()

            prec1, prec5 = accuracy(logits, labels, topk=(1, 5))
            n = len(labels)
            train_loss.update(loss.item(),n)
            train_top1.update(prec1.item(),n)
            train_top5.update(prec5.item(),n)

        #during eval time, ghn does not change
        models_pred = ghn(models,graphs)
        res20 = ghn([res20],res20_graph)[0]
        for _,(images,labels) in enumerate(testloader):
            
            res20_logits = 0
            logits = 0
            loss = 0
            count = 0

            images = images.to(device)
            labels = labels.to(device)

            for i,model in enumerate(models_pred): 
                y = model(images)
                loss += F.cross_entropy(y, labels)
                logits += y 
                count += 1 
            
            res20_logits += res20(images)

            
            loss = loss/count 
            logits = logits/count

            prec1, prec5 = accuracy(logits, labels, topk=(1, 5))
            res20_p1, _ = accuracy(res20_logits, labels, topk=(1, 5))

            n = len(labels)
            val_loss.update(loss.item(),n)
            val_top1.update(prec1.item(),n)
            val_top5.update(prec5.item(),n)
            val_res_top1.update(res20_p1.item(),n)

        logger.add_scalar(epoch, 'time', time.time()-start_epoch)
        logger.add_scalar(epoch, 'train_loss', train_loss.avg)
        logger.add_scalar(epoch, 'train_top1', train_top1.avg)
        logger.add_scalar(epoch, 'val_top1', val_top1.avg)
        logger.add_scalar(epoch, 'val_res_top1', val_res_top1.avg)
        logger.add_scalar(epoch, 'val_loss', val_loss.avg)
        logger.add_scalar(epoch, 'train_top5', train_top5.avg)
        logger.add_scalar(epoch, 'val_top5', val_top5.avg)
        
        logger.iter_info()
        logger.save()

        if (epoch-1) % 10 == 0:
            torch.save(ghn.state_dict() , os.path.join(args.root, 'ghn_params_epoch_{}.torch'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(args.root, 'opt_params_epoch_{}.torch'.format(epoch)))

        is_best = (val_loss.avg < best_loss)
        if is_best:
            best_loss = val_loss.avg
            torch.save(ghn.state_dict(), os.path.join(args.root, 'ghn_params.torch')) 

        scheduler.step()

    torch.save(ghn.state_dict(), os.path.join(args.root, 'ghn_params_lastepoch.torch'))
    torch.save(optimizer.state_dict(), os.path.join(args.root, 'opt_params_lastepoch.torch'))

