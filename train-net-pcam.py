import torch
import _dwp.utils as utils
import numpy as np
import os
import time
from models.resnet import ResNet
from _dwp.logger import Logger
import _dwp.myexman as myexman
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from _dwp.my_utils import MultistepMultiGammaLR

def predict(data, net):
    pred = []
    l = []
    for x, y in data:
        l.append(y.numpy())
        x = x.to(device)
        p = F.log_softmax(net(x), dim=1)
        pred.append(p.data.cpu().numpy())
    return np.concatenate(pred), np.concatenate(l)

def eval_step(it,net,data,logger,writer,args,opt,train_acc,train_nll,lrscheduler,step=True):
    global t0
    global best_acc

    net.eval()
    with torch.no_grad():
        logp_val, labels = predict(data, net)
        val_acc = np.mean(logp_val.argmax(1) == labels)
        val_nll = -logp_val[np.arange(len(labels)), labels].mean()

    logger.add_scalar(it, 'train_nll', train_nll.get_val())
    logger.add_scalar(it, 'train_acc', train_acc.get_val())
    logger.add_scalar(it, 'val_nll', val_nll)
    logger.add_scalar(it, 'val_acc', val_acc)
    logger.add_scalar(it, 'lr', opt.param_groups[0]['lr'])
    logger.add_scalar(it, 'sec', time.time() - t0)
    logger.iter_info()
    logger.save()

    writer.add_scalar( 'train/nll', train_nll.get_val(),it)
    writer.add_scalar( 'val/nll', val_nll,it)
    writer.add_scalar( 'train/acc', train_acc.get_val(),it)
    writer.add_scalar( 'val/acc', val_acc,it)
    writer.add_scalar( 'lr', opt.param_groups[0]['lr'],it)
    writer.add_scalar( 'sec', time.time() - t0,it)
    
    is_best = best_acc < val_acc
    if is_best:
        best_acc = val_acc
        torch.save(net.state_dict(), os.path.join(args.root, 'net_params.torch'))     

    t0 = time.time()
    if step:
        lrscheduler.step()

parser = myexman.ExParser(file=__file__)
#general settings
parser.add_argument('--name', default='')
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--seed', default=5743, type=int)
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
parser.add_argument('--bs', default=128, type=int, help='Batch size')
parser.add_argument('--test_bs', default=500, type=int, help='Batch size for test dataloader')

#model settings
parser.add_argument('--n_classes', default=2, type=int)

#model init settings MULTI init (if used, single init is ignored)
parser.add_argument('--mult_init_mode', default= 'ghn_base', type = str,
                    help = '''such as vqvae1.3''')
parser.add_argument('--mult_init_root', type=str, default=os.path.join('data','resnet20_pcam','3x3'))
parser.add_argument('--mult_init_prior', type=str, default='',
                    help='''such as pixelcnn0''')

#optimizer settings
parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--milestones', type=int, nargs='*', default=[24,28])
parser.add_argument('--gammas', default=[0.5,0.2], nargs='*', type=float)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

fmt = {
    'lr': '.5f',
    'sec': '.0f',
}
logger = Logger('logs', base=args.root, fmt=fmt)

# Load Datasets
t0 = time.time()
trainloader, valloader, testloader = utils.load_pcam_loaders(args.bs, args.test_bs)
load_time = time.time()
print(f"loading data took {load_time-t0} seconds")
net = ResNet([3,3,3],num_classes=args.n_classes).to(device)

# Initialization
net.mult_weights_init(args.mult_init_mode, args.mult_init_root, device=device,dataset='pcam', prior=args.mult_init_prior)


opt = torch.optim.SGD(net.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

lrscheduler = MultistepMultiGammaLR(opt, milestones=args.milestones, 
                                    gamma=args.gammas)


t0 = time.time()

it = 0

#add a tensorboard writer
writer = SummaryWriter(args.root)
best_acc = 0.

#for init eval 
train_acc = utils.MovingMetric()
train_nll = utils.MovingMetric()
eval_step(it,net,valloader,logger,writer,args,opt,train_acc,train_nll,lrscheduler,step=False)

for e in range(1, args.epochs + 1):
    net.train()
    train_acc = utils.MovingMetric()
    train_nll = utils.MovingMetric()

    for x, y in trainloader:
        opt.zero_grad()
        it += 1
        x = x.to(device)
        y = y.to(device)

        p = net(x)

        nll = F.cross_entropy(p, y)

        nll.backward()

        opt.step()

        acc = torch.sum(p.max(1)[1] == y)
        train_acc.add(acc.item(), p.size(0))
        train_nll.add(nll.item() * x.size(0), x.size(0))

        if it % 300 == 0:
            eval_step(it,net,valloader,logger,writer,args,opt,train_acc,train_nll,lrscheduler)
            net.train() #after eval set net to train an reset the moving metrices
            train_acc = utils.MovingMetric()
            train_nll = utils.MovingMetric()


torch.save(net.state_dict(), os.path.join(args.root, 'net_params_lastepoch.torch'))
torch.save(opt.state_dict(), os.path.join(args.root, 'opt_params_lastepoch.torch'))

writer.flush()
