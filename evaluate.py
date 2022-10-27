import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from autoattack import AutoAttack
# from utils import normalize
# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)



def normalize(args, X):
    if args.dataset=="cifar":
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
        std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    elif args.dataset=="imagenette" or args.dataset=="imagenet" :
        mu = torch.tensor(imagenet_mean).view(3, 1, 1).cuda()
        std = torch.tensor(imagenet_std).view(3, 1, 1).cuda()
    return (X - mu) / std



def evaluate_aa(args, model,log_path,aa_batch=128):
    if args.dataset=="cifar":
        test_transform_nonorm = transforms.Compose([
            transforms.ToTensor()
        ])
        test_dataset_nonorm = datasets.CIFAR10(
        args.data_dir, train=False, transform=test_transform_nonorm, download=True)
    if args.dataset=="imagenette" or args.dataset=="imagenet" :
        test_transform_nonorm = transforms.Compose([
            transforms.Resize([args.resize, args.resize]),
            transforms.ToTensor()
        ])
        test_dataset_nonorm = datasets.ImageFolder(args.data_dir+"val/",test_transform_nonorm)
    test_loader_nonorm = torch.utils.data.DataLoader(
        dataset=test_dataset_nonorm,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    model.eval()
    l = [x for (x, y) in test_loader_nonorm]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader_nonorm]
    y_test = torch.cat(l, 0)
    class normalize_model():
        def __init__(self, model):
            self.model_test = model
        def __call__(self, x):
            x_norm = normalize(args, x)
            return self.model_test(x_norm)
    new_model = normalize_model(model)
    epsilon = args.epsilon / 255.
    adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard',log_path=log_path)
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=aa_batch)
