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

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch):
    lr = args.lr * np.minimum((args.decrease_from - epoch) * 1. / (args.epochs - args.decrease_from) + 1, 1.)
    return max(0, lr)


def predict(data, net):
    pred = []
    l = []
    for x, y in data:
        l.append(y.numpy())
        x = x.to(device)
        p = F.log_softmax(net(x), dim=1)
        pred.append(p.data.cpu().numpy())
    return np.concatenate(pred), np.concatenate(l)


parser = myexman.ExParser(file=__file__)
#general settings
parser.add_argument('--name', default='')
parser.add_argument('--data', default='cifar')
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--data_split_seed', default=456, type=int)
parser.add_argument('--seed', default=5743, type=int)
parser.add_argument('--resume', default='')
parser.add_argument('--epochs', default=120, type=int, help='Number of epochs')
parser.add_argument('--bs', default=128, type=int, help='Batch size')
parser.add_argument('--test_bs', default=500, type=int, help='Batch size for test dataloader')

#model settings
parser.add_argument('--n_classes', default=10, type=int)

#model init settings MULTI init (if used, single init is ignored)
parser.add_argument('--mult_init_mode', default= 'xavier', type = str,
                    help = '''such as vqvae1.3''')
parser.add_argument('--mult_init_root', type=str, default=os.path.join('data','resnet20','3x3'))
parser.add_argument('--mult_init_prior', type=str, default='',
                    help='''such as pixelcnn0''')

#optimizer settings
parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--milestones', type=int, nargs='*', default=[80,100])
parser.add_argument('--gammas', default=[0.5,0.2], nargs='*', type=float)
parser.add_argument('--decrease_from', default=0, type=int) #unused 


#evaluation and leftovers
parser.add_argument('--eval_freq', default=1, type=int)



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
trainloader, testloader = utils.load_dataset(data=args.data, train_bs=args.bs, test_bs=args.test_bs,
                                             num_examples=None, seed=args.data_split_seed,
                                             augmentation=False)
                                            
# Network
net = ResNet([3,3,3],num_classes=args.n_classes)
net.mult_weights_init(args.mult_init_mode, args.mult_init_root, device=device, prior=args.mult_init_prior)
net = net.to(device)

# Optimizer
opt = torch.optim.SGD(net.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
lrscheduler = MultistepMultiGammaLR(opt, milestones=args.milestones, 
                                    gamma=args.gammas)

N = len(trainloader.dataset)
t0 = time.time()

it = 0

#add a tensorboard writer
writer = SummaryWriter(args.root)
best_acc = 0.
for e in range(1, args.epochs + 1):
    if args.milestones:
        lrscheduler.step()
    else:
        adjust_learning_rate(opt, lr_linear(e - 1))
    net.train()
    train_acc = utils.MovingMetric()
    train_nll = utils.MovingMetric()
    train_loss = utils.MovingMetric()
    opt.zero_grad()

    for x, y in trainloader:
        opt.zero_grad()
        it += 1
        x = x.to(device)
        y = y.to(device)

        p = net(x)

        data_term = F.cross_entropy(p, y)

        loss = data_term 

        loss.backward()

        opt.step()

        acc = torch.sum(p.max(1)[1] == y)
        train_acc.add(acc.item(), p.size(0))
        train_nll.add(data_term.item() * x.size(0), x.size(0))
        train_loss.add(loss.item() * x.size(0), x.size(0))

    if ((e % args.eval_freq) == 0 or e == 1):
        net.eval()

        logp_test, labels = predict(testloader, net)
        test_acc = np.mean(logp_test.argmax(1) == labels)
        test_nll = -logp_test[np.arange(len(labels)), labels].mean()

        logger.add_scalar(e, 'loss', train_loss.get_val())
        logger.add_scalar(e, 'train_nll', train_nll.get_val())
        logger.add_scalar(e, 'test_nll', test_nll)
        logger.add_scalar(e, 'train_acc', train_acc.get_val())
        logger.add_scalar(e, 'test_acc', test_acc)
        logger.add_scalar(e, 'lr', opt.param_groups[0]['lr'])
        logger.add_scalar(e, 'sec', time.time() - t0)

        logger.iter_info()
        logger.save()

        writer.add_scalar( 'train/loss', train_loss.get_val(),e)
        writer.add_scalar( 'train/nll', train_nll.get_val(),e)
        writer.add_scalar( 'test/nll', test_nll,e)
        writer.add_scalar( 'train/acc', train_acc.get_val(),e)
        writer.add_scalar( 'test/acc', test_acc,e)
        writer.add_scalar( 'lr', opt.param_groups[0]['lr'],e)
        writer.add_scalar( 'sec', time.time() - t0,e)
        
        epoch = e 
        if (epoch-1) % 10 == 0:
            torch.save(net.state_dict() , os.path.join(args.root, 'net_params_epoch_{}.torch'.format(epoch)))
            torch.save(opt.state_dict(), os.path.join(args.root, 'opt_params_epoch_{}.torch'.format(epoch)))

        
        is_best = best_acc < test_acc
        if is_best:
            best_acc = test_acc
            torch.save(net.state_dict(), os.path.join(args.root, 'net_params.torch'))     
   
        t0 = time.time()

torch.save(net.state_dict(), os.path.join(args.root, 'net_params_lastepoch.torch'))
torch.save(opt.state_dict(), os.path.join(args.root, 'opt_params_lastepoch.torch'))

writer.flush()
