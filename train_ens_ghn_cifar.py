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
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

if __name__ == '__main__':
    parser = myexman.ExParser(file=__file__)
    #train settings 
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--data_split_seed', default=42, type=int)
    parser.add_argument('--gpu_id', default='0')

    #optimisation
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--lr_steps', type=str, default='200,250', help='epochs when to decrease lr')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')

    #model
    parser.add_argument('-gn','--gen_noise', default=True, action='store_false')
    parser.add_argument('-tn','--train_noise', default=False, action='store_true')
    parser.add_argument('--var_init', default=1, type=float)
    parser.add_argument('--mu_scale', default=1, type=float)
    parser.add_argument('--var_scale', default=0.1, type=float)
    parser.add_argument('--csim_weight', default=1, type=float)
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
    from ppuda.deepnets1m.graph import Graph, GraphBatch
    from ppuda.ghn.decoder import ConvDecoder

    #get dataloaders for image data 
    trainloader, valloader, _ = utils.load_cifar10_loaders(bs=args.batch_size, test_bs=500)

    #load/init the model 
    from ppuda.ghn.nn import GHN2
    ghn = GHN2('cifar10')
    hid = ghn.hid
    ghn.decoder = ConvDecoder(in_features=hid,
                        hid=(hid * 4, hid * 8),
                        out_shape=(64,64,3,3),
                        num_classes=10,
                        gen_noise = True,
                        var_init = args.var_init,
                        mu_scale  = args.mu_scale,
                        var_scale = args.var_scale,
                        train_noise= args.train_noise)

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
    CosineSimilarity = torch.nn.CosineSimilarity(dim=0)
    csim_weight =  args.csim_weight

    #Training and evaluation below
    logger = Logger(name='logs', base=args.root)
    writer = SummaryWriter(args.root)
    best_loss = 11e8

    # a first eval before training 
    ghn.eval()
    val_loss = utils.AvgrageMeter()
    val_ce = utils.AvgrageMeter()
    val_cossim = utils.AvgrageMeter()
    val_top1 = utils.AvgrageMeter()
    val_res_cossim = utils.AvgrageMeter()
    val_res_top1 = utils.AvgrageMeter()

    for i,(images,labels) in enumerate(valloader):
        ce_loss = 0
        logits = 0
        loss = 0
        count = 0

        images = images.to(device)
        labels = labels.to(device)

        #for validation models 
        for j in range(2):
            models_pred = ghn(models,graphs)
            logit_vec = []
            for i,model in enumerate(models_pred):
                y = model(images)
                logit_vec.append(y.flatten())
                ce_loss += F.cross_entropy(y, labels)
                logits += y 
                count += 1 
            if j == 0:
                logits_0 = torch.cat(logit_vec).to(device)
            if j == 1:
                logits_1 = torch.cat(logit_vec).to(device)
                cos_sim = CosineSimilarity(logits_0,logits_1)
        ce_loss = ce_loss/count 
        logits = logits/count
        loss = ce_loss + cos_sim * csim_weight

        prec1, _ = accuracy(logits, labels, topk=(1, 5))
        n = len(labels)
        val_loss.update(loss.item(),n)
        val_top1.update(prec1.item(),n)
        val_ce.update(ce_loss.item(),n)
        val_cossim.update(cos_sim.item(),n)

        # for res20 
        for j in range(2):
            res20 = ghn([res20],res20_graph)[0]  
            y = res20(images)
            if j == 0:
                logits = y 
            if j == 0:
                logits_0 = y.flatten().to(device)
            if j == 1:
                logits_1 = y.flatten().to(device)
                cos_sim = CosineSimilarity(logits_0,logits_1)
        
        prec1, _ = accuracy(logits, labels, topk=(1, 5))
        n = len(labels)
        val_res_top1.update(prec1.item(),n)
        val_res_cossim.update(cos_sim.item(),n)

    logger.add_scalar(0, 'val_loss', val_loss.avg)
    logger.add_scalar(0, 'val_ce', val_ce.avg)
    logger.add_scalar(0, 'val_cossim', val_cossim.avg)
    logger.add_scalar(0, 'val_top1', val_top1.avg)
    logger.add_scalar(0, 'res_top1', val_res_top1.avg)
    logger.add_scalar(0, 'two_res_cossim', val_res_cossim.avg)
    logger.iter_info()
    logger.save()
    writer.add_scalar('val/loss',val_loss.avg,0)
    writer.add_scalar('val/ce', val_ce.avg,0)
    writer.add_scalar('val/cossim', val_cossim.avg,0)
    writer.add_scalar('val/top1', val_top1.avg,0)
    writer.add_scalar('val/res_top1', val_res_top1.avg,0)
    writer.add_scalar('val/two_res_cossim', val_res_cossim.avg,0)

    torch.save(ghn.state_dict(), os.path.join(args.root, 'ghn_params_init.torch'))

    for epoch in range(1, args.num_epochs + 1):
        
        train_loss = utils.AvgrageMeter()
        train_ce = utils.AvgrageMeter()
        train_cossim = utils.AvgrageMeter()
        train_top1 = utils.AvgrageMeter()

        val_loss = utils.AvgrageMeter()
        val_ce = utils.AvgrageMeter()
        val_cossim = utils.AvgrageMeter()
        val_top1 = utils.AvgrageMeter()
        val_res_cossim = utils.AvgrageMeter()
        val_res_top1 = utils.AvgrageMeter()

        start_epoch = time.time()

        for i,(images,labels) in enumerate(trainloader):
            ghn.train()
            ce_loss = 0
            logits = 0
            loss = 0
            count = 0

            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)

            for j in range(2):
                models_pred = ghn(models,graphs)
                logit_vec = []
                for i,model in enumerate(models_pred):
                    y = model(images)
                    logit_vec.append(y.flatten())
                    ce_loss += F.cross_entropy(y, labels)
                    logits += y 
                    count += 1 
                if j == 0:
                    logits_0 = torch.cat(logit_vec).to(device)
                if j == 1:
                    logits_1 = torch.cat(logit_vec).to(device)
                    cos_sim = CosineSimilarity(logits_0,logits_1)
            ce_loss = ce_loss/count 
            logits = logits/count

            loss = ce_loss + cos_sim * csim_weight
            loss.backward()

            parameters = []
            for group in optimizer.param_groups:
                parameters.extend(group['params'])

            nn.utils.clip_grad_norm_(parameters, 5)
            optimizer.step()

            prec1, _ = accuracy(logits, labels, topk=(1, 5))
            n = len(labels)
            train_loss.update(loss.item(),n)
            train_top1.update(prec1.item(),n)
            train_ce.update(ce_loss.item(),n)
            train_cossim.update(cos_sim.item(),n)

        for i,(images,labels) in enumerate(valloader):
            ghn.eval()
            ce_loss = 0
            logits = 0
            loss = 0
            count = 0

            images = images.to(device)
            labels = labels.to(device)

            #for validation models 
            for j in range(2):
                models_pred = ghn(models,graphs)
                logit_vec = []
                for i,model in enumerate(models_pred):
                    y = model(images)
                    logit_vec.append(y.flatten())
                    ce_loss += F.cross_entropy(y, labels)
                    logits += y 
                    count += 1 
                if j == 0:
                    logits_0 = torch.cat(logit_vec).to(device)
                if j == 1:
                    logits_1 = torch.cat(logit_vec).to(device)
                    cos_sim = CosineSimilarity(logits_0,logits_1)
            ce_loss = ce_loss/count 
            logits = logits/count
            loss = ce_loss + cos_sim * csim_weight

            prec1, _ = accuracy(logits, labels, topk=(1, 5))
            n = len(labels)
            val_loss.update(loss.item(),n)
            val_top1.update(prec1.item(),n)
            val_ce.update(ce_loss.item(),n)
            val_cossim.update(cos_sim.item(),n)


            # for res20 
            for j in range(2):
                res20 = ghn([res20],res20_graph)[0]  
                y = res20(images)
                if j == 0:
                    logits = y 
                if j == 0:
                    logits_0 = y.flatten().to(device)
                if j == 1:
                    logits_1 = y.flatten().to(device)
                    cos_sim = CosineSimilarity(logits_0,logits_1)
            
            prec1, _ = accuracy(logits, labels, topk=(1, 5))
            n = len(labels)
            val_res_top1.update(prec1.item(),n)
            val_res_cossim.update(cos_sim.item(),n)

        duration = time.time()-start_epoch   
        logger.add_scalar(epoch, 'time', duration)
        logger.add_scalar(epoch, 'train_loss', train_loss.avg)
        logger.add_scalar(epoch, 'train_ce', train_ce.avg)
        logger.add_scalar(epoch, 'train_cossim', train_cossim.avg)
        logger.add_scalar(epoch, 'train_top1', train_top1.avg)
        logger.add_scalar(epoch, 'val_loss', val_loss.avg)
        logger.add_scalar(epoch, 'val_ce', val_ce.avg)
        logger.add_scalar(epoch, 'val_cossim', val_cossim.avg)
        logger.add_scalar(epoch, 'val_top1', val_top1.avg)
        logger.add_scalar(epoch, 'res_top1', val_res_top1.avg)
        logger.add_scalar(epoch, 'two_res_cossim', val_res_cossim.avg)
        logger.iter_info()
        logger.save()

        writer.add_scalar('time', duration,epoch)
        writer.add_scalar('train/loss', train_loss.avg,epoch)
        writer.add_scalar('train/ce', train_ce.avg,epoch)
        writer.add_scalar('train/cossim', train_cossim.avg,epoch)
        writer.add_scalar('train/top1', train_top1.avg,epoch)
        writer.add_scalar('val/loss', val_loss.avg,epoch)
        writer.add_scalar('val/ce', val_ce.avg,epoch)
        writer.add_scalar('val/cossim', val_cossim.avg,epoch)
        writer.add_scalar('val/top1', val_top1.avg,epoch)
        writer.add_scalar('val/res_top1', val_res_top1.avg,epoch)
        writer.add_scalar('val/two_res_cossim', val_res_cossim.avg,epoch)

        if (epoch-1) % 10 == 0:
            torch.save(ghn.state_dict() , os.path.join(args.root, 'ghn_params_epoch_{}.torch'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(args.root, 'opt_params_epoch_{}.torch'.format(epoch)))

        is_best = (val_loss.avg < best_loss)
        if is_best:
            best_loss = val_loss.avg
            torch.save(ghn.state_dict(), os.path.join(args.root, 'ghn_params_val.torch')) 

        scheduler.step()

    torch.save(ghn.state_dict(), os.path.join(args.root, 'ghn_params_lastepoch.torch'))
    torch.save(optimizer.state_dict(), os.path.join(args.root, 'opt_params_lastepoch.torch'))
    writer.flush()