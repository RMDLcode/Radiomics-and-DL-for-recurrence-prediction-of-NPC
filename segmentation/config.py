import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')
crop_size_z=32

parser.add_argument('--n_threads', type=int, default=8,help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',help='use cpu only')
parser.add_argument('--gpu_id', type=list,default=[2], help='use gpu')
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--n_labels', type=int, default=2,help='number of classes')
 
# data in/out and dataset
parser.add_argument('--dataset_path',default = './reccurence_data/train',help='trainset path')
parser.add_argument('--test_data_path',default = './reccurence_data/test',help='Testset path')
parser.add_argument('--save',default='test_1',help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=2,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=50, metavar='N',help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',help='learning rate (default: 1e-3)')
parser.add_argument('--early-stop', default=None, type=int, help='early stopping')
parser.add_argument('--crop_size', type=int, default=crop_size_z)
parser.add_argument('--inputshape', type=int, default=256)

# test
parser.add_argument('--test_cut_size', type=int, default=crop_size_z, help='size of sliding window')
parser.add_argument('--test_cut_stride', type=int, default=crop_size_z//2, help='stride of sliding window')

args = parser.parse_args()


