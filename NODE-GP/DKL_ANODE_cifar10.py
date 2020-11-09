from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch import nn
import torch
import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
import gpytorch
import math
import tqdm
import numpy as np

#normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
common_trans = [transforms.ToTensor(), normalize]
train_compose = transforms.Compose(aug_trans + common_trans)
test_compose = transforms.Compose(common_trans)

dataset = "cifar10"



if ('CI' in os.environ):  # this is for running the notebook in our testing framework
    train_set = torch.utils.data.TensorDataset(torch.randn(8, 3, 32, 32), torch.rand(8).round().long())
    test_set = torch.utils.data.TensorDataset(torch.randn(4, 3, 32, 32), torch.rand(4).round().long())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False)
    num_classes = 2
elif dataset == 'cifar10':
    train_set = dset.CIFAR10('data', train=True, transform=train_compose, download=True)
    test_set = dset.CIFAR10('data', train=False, transform=test_compose)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=250, shuffle=False)
    num_classes = 10
elif dataset == 'cifar100':
    train_set = dset.CIFAR100('data', train=True, transform=train_compose, download=True)
    test_set = dset.CIFAR100('data', train=False, transform=test_compose)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    num_classes = 100
else:
    raise RuntimeError('dataset must be one of "cifar100" or "cifar10"')



from densenet import DenseNet
#from NODE import node
from models.resnet_node import ResNet18, lr_schedule
#net = ResNet18(ODEBlock)
#print(net)
class DenseNetFeatureExtractor(ResNet18):
	def forward(self, x):
		#print(x.shape)
		features = self.features(x)
		#print(features.shape)
		out = F.relu(features, inplace=True)
		out = F.avg_pool2d(out, kernel_size=4).view(features.size(0), -1)
		#print(out.shape)
		return out

feature_extractor = DenseNetFeatureExtractor([2,2,2,2])
#num_features = feature_extractor.classifier.in_features
num_features = 512

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64*4):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )
        
        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a MultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
'''
Creating the full SVDKL Model

With both the DenseNet feature extractor and GP layer defined, we can put them together in a single module that simply calls one and then the other, much like building any Sequential neural network in PyTorch. This completes defining our DKL model.
'''

class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        #print(self.grid_bounds)
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res
feature_extractor =  torch.nn.DataParallel(feature_extractor)
model = DKLModel(feature_extractor, num_dim=num_features)


'''
checkpoint = torch.load('/home/cs16resch11004/anode/checkpoints/cifar10_ANODE_ANODE_NT2.pth')
net = model.feature_extractor.parameters()
#net.to(device)
#net = nn.DataParallel(net)
keys = list(checkpoint['state_dict'].keys())

#print(model.feature_extractor.parameters().keys[0])
i = 0

for param in model.feature_extractor.parameters():
    print(param.shape)
print(i)
print(len(keys))

for i in range(len(keys)):
    print(checkpoint['state_dict'][keys[i]].cpu().shape)
net.load_state_dict(checkpoint['state_dict'])
'''
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)

def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 250:
        optim_factor = 2
    elif epoch > 150:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))

#print(model)
# If you run this example without CUDA, I hope you like waiting!
if torch.cuda.is_available():
    model = model.cuda()
    #model = torch.nn.DataParallel(model)
    
    likelihood = likelihood.cuda()
'''
Defining Training and Testing Code

Next, we define the basic optimization loop and testing code. This code is entirely analogous to the standard PyTorch training loop. We create a torch.optim.SGD optimizer with the parameters of the neural network on which we apply the standard amount of weight decay suggested from the paper, the parameters of the Gaussian process (from which we omit weight decay, as L2 regualrization on top of variational inference is not necessary), and the mixing parameters of the Softmax likelihood.

We use the standard learning rate schedule from the paper, where we decrease the learning rate by a factor of ten 50% of the way through training, and again at 75% of the way through training.

'''
n_epochs = 350
lr = 0.1
name = 'ANODE_NT2_WD5_IND256_NODE'
st_epochs = 0
Train_Acc = []
Test_Acc = []
Train_Loss = []
Test_Loss = []
Val_Loss = []
Val_Acc = []

#instead of 1e-4 need to try trying 5e-4 with learning schedule only for weightd

optimizer = SGD([
    {'params': model.feature_extractor.parameters(),'weight_decay': 5e-4},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
#scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))
#print(model)

####Resume
'''
checkpoint = torch.load('./checkpoints1/dkl_'+name+'_cifar_checkpoint.dat')
model.load_state_dict(checkpoint['model'])
likelihood.load_state_dict(checkpoint['likelihood'])

Train_Acc = list(np.load('./checkpoints1/Train_Acc_cifar10'+name+'.npy'))
Test_Acc = list(np.load('./checkpoints1/Test_Acc_cifar10'+name+'.npy'))
Train_Loss = list(np.load('./checkpoints1/Train_Loss_cifar10'+name+'.npy'))
Test_Loss = list(np.load('./checkpoints1/Test_Loss_cifar10'+name+'.npy'))
st_epochs = len(Train_Acc)
'''
####

def train(epoch):
    model.train()
    likelihood.train()
    total_loss = 0
    correct = 0
    minibatch_iter = tqdm.tqdm_notebook(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(8):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            pred = likelihood(output).probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()#Added

            loss = -mll(output, target)
            loss.backward()
            total_loss += loss.item()  #added
           
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())
        Train_Loss.append(total_loss)
        Train_Acc.append(correct)
    print('Train set: Accuracy: {}/{} ({}%)'.format(
    correct, len(train_loader.dataset), 100. * correct / float(len(train_loader.dataset))
))
val_acc = 0        
def test():
    global val_acc
    model.eval()
    likelihood.eval()
    val_loss = 0
    test_loss = 0
    count = 0
    correct = 0
    val = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for data, target in test_loader:
            #print(target)
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)  # This gives us 16 samples from the predictive distribution
            #total_loss += -mll(output, target).item() #added
            pred = likelihood(output).probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            
            if count < 32:
                val += pred.eq(target.view_as(pred)).cpu().sum()
                val_loss += -mll(output, target).item()
            else:
                correct += pred.eq(target.view_as(pred)).cpu().sum()
                test_loss += -mll(output, target).item()
                #print(val)
            count += 1
        Test_Loss.append(test_loss)
        Test_Acc.append(100. * correct / float(len(test_loader.dataset)*0.2))
        Val_Loss.append(val_loss)
        Val_Acc.append(100. * val / float(len(test_loader.dataset)*0.8))
        
        #print(val)
        print(name+'_DKL Test set: Accuracy: {}/{} ({}%)'.format(
            correct, len(test_loader.dataset)*0.2, 100. * correct / float(len(test_loader.dataset)*0.2)
        ))
        print(name+'_DKL Validation set: Accuracy: {}/{} ({}%)'.format(
            val, len(test_loader.dataset)*0.8, 100. * val / float(len(test_loader.dataset)*0.8)
        ))
    if val > val_acc:
        val_acc = val

        state_dict = model.state_dict()
        likelihood_state_dict = likelihood.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, './checkpoints2/dkl_'+name+'_cifar_checkpoint.dat')


import time
start = time.time()
#print(model)
#global val_acc

for epoch in range(st_epochs + 1, n_epochs + 1):
    start1 = time.time()
    with gpytorch.settings.use_toeplitz(False):
        train(epoch)
        test()
    total_time = (time.time() - start1)/60.0
    print("Epoch TIME: {:.3f} ".format(total_time))
    
    np.save('./checkpoints2/Train_Acc_cifar10'+name+'.npy', Train_Acc)
    np.save('./checkpoints2/Test_Acc_cifar10'+name+'.npy', Test_Acc)
    np.save('./checkpoints2/Val_Acc_cifar10'+name+'.npy', Val_Acc)
    np.save('./checkpoints2/Train_Loss_cifar10'+name+'.npy', Train_Loss)
    np.save('./checkpoints2/Test_Loss_cifar10'+name+'.npy', Test_Loss)
    np.save('./checkpoints2/Val_Loss_cifar10'+name+'.npy', Val_Loss)
    
    s = scheduler.step()
    print(s)
    
total_time = (time.time() - start)/60.0
print("TOTAL TIME: {:.3f} ".format(total_time))

'''


checkpoint = torch.load('dkl_'+name+'_cifar_checkpoint.dat')
model.load_state_dict(checkpoint['model'])
likelihood.load_state_dict(checkpoint['likelihood'])
test()
'''