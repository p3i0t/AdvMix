import os
import logging
import hydra
from omegaconf import DictConfig

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, datasets
import augmentations

from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet

from utils import AverageMeter, cal_parameters

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = logging.getLogger(__name__)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def aug(image, preprocess, args):
    """Perform AugMix augmentations and compute mixture.

    Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

    Returns:
    mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    if args.all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):  # size of composed augmentations set
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
        1, 4)
        for _ in range(depth):   # compose one augmentation with depth number of single aug operation.
          op = np.random.choice(aug_list)
          image_aug = op(image_aug, args.aug_severity)

        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(Dataset):
    """Dataset wrapper to perform AugMix augmentation."""
    def __init__(self, dataset, preprocess, args, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.args = args

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), aug(x, self.preprocess, self.args), aug(x, self.preprocess, self.args))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def jason_shanon_loss(prob_list):
    from functools import reduce
    # Clamp mixture distribution to avoid exploding KL divergence
    p_mix = reduce(lambda a, b: a + b, prob_list) / len(prob_list)
    p_mix = p_mix.clamp(1e-7, 1.).log()

    return reduce(lambda a, b: a + b, [F.kl_div(p_mix, p, reduction='batchmean') for p in prob_list]) / len(prob_list)


def train_epoch(classifier, train_loader, args, optimizer, scheduler):
    """Train for one epoch."""
    classifier.train()
    loss_meter = AverageMeter('loss')
    ce_meter = AverageMeter('ce_loss')
    js_meter = AverageMeter('js_loss')
    acc_meter = AverageMeter('acc_loss')

    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.no_jsd:
            x = x.to(args.device)
            y = y.to(args.device)
            logits = classifier(x)
            loss = F.cross_entropy(logits, y)

            loss_meter.update(loss.item(), x[0].size(0))
            acc = (logits_clean.argmax(dim=1) == y).float().mean().item()
            acc_meter.update(acc, x[0].size(0))
        else:
            x_all = torch.cat(x, 0).to(args.device)
            y = y.to(args.device)
            logits_all = classifier(x_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
              logits_all, x[0].size(0))

            # Cross-entropy is only computed on clean images
            ce_loss = F.cross_entropy(logits_clean, y)

            p_clean = logits_clean.softmax(dim=1)
            p_aug1 = logits_aug1.softmax(dim=1)
            p_aug2 = logits_aug2.softmax(dim=1)

            js_loss = jason_shanon_loss([p_clean, p_aug1, p_aug2])

            loss = ce_loss + 12 * js_loss

            loss_meter.update(loss.item(), x[0].size(0))
            ce_meter.update(ce_loss.item(), x[0].size(0))
            js_meter.update(js_loss.item(), x[0].size(0))
            acc = (logits_clean.argmax(dim=1) == y).float().mean().item()
            acc_meter.update(acc, x[0].size(0))

        loss.backward()
        optimizer.step()
        scheduler.step()

        # loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    return loss_meter.avg, ce_meter.avg, js_meter.avg, acc_meter.avg


def train_epoch_advmix(classifier, train_loader, args, optimizer, scheduler):
    """Train for one epoch."""
    classifier.train()
    loss_meter = AverageMeter('loss')
    ce_meter = AverageMeter('ce_loss')
    js_meter = AverageMeter('js_loss')
    acc_meter = AverageMeter('acc_loss')

    # ajust according to std.
    eps = eval(args.epsilon) / std_
    eps_iter = eval(args.epsilon_iter) / std_

    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = torch.cat(x, 0).to(args.device)
        y = y.to(args.device)

        # start with uniform noise
        delta = torch.zeros_like(x).uniform_(-eps, eps)
        delta.requires_grad_()
        delta = clamp(delta, clip_min - x, clip_max - x)

        loss = F.cross_entropy(classifier(x + delta), torch.cat([y]*3, dim=0))
        grad_delta = torch.autograd.grad(loss, delta)[0].detach()  # get grad of noise

        # update delta with grad
        delta = (delta + torch.sign(grad_delta) * eps_iter).clamp_(-eps, eps)
        delta = clamp(delta, clip_min - x, clip_max - x)

        x_ = torch.cat([x, x + delta], dim=0)

        logits_list = torch.split(classifier(x_), y.size(0))
        # logits_clean, logits_aug1, logits_aug2, logits_clean_adv， logits_aug1_adv, logits_aug2_adv

        # Cross-entropy is only computed on clean images
        ce_loss = F.cross_entropy(logits_list[0], y)

        prob_list = [logits.softmax(dim=1) for logits in logits_list]
        js_loss = jason_shanon_loss(prob_list[:3]) \
                  + jason_shanon_loss([prob_list[0], prob_list[3]])\
                  + jason_shanon_loss([prob_list[1], prob_list[4]])\
                  + jason_shanon_loss([prob_list[2], prob_list[5]])

        loss = ce_loss + 12 * js_loss

        loss_meter.update(loss.item(), x[0].size(0))
        ce_meter.update(ce_loss.item(), x[0].size(0))
        js_meter.update(js_loss.item(), x[0].size(0))

        acc = (logits_list[0].argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x[0].size(0))

        loss.backward()
        optimizer.step()
        scheduler.step()

        # loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    return loss_meter.avg, ce_meter.avg, js_meter.avg, acc_meter.avg


# Note that we don't use cifar10 specific normalization, so generally use 0.5 as mean and std.
mean_ = 0.5
std_ = 0.5
clip_min = -1.
clip_max = 1.


def attack_pgd(model, x, y, eps, eps_iter, attack_iters, restarts):
    """
    Perform PGD attack on one mini-batch.
    :param model: pytorch model.
    :param x: x of minibatch.
    :param y: y of minibatch.
    :param eps: L-infinite norm budget.
    :param eps_iter: step size for each iteration.
    :param attack_iters: number of iterations.
    :param restarts:  number of restart times
    :return: best adversarial perturbations delta in all restarts
    """
    assert x.device == y.device
    max_loss = torch.zeros_like(y).float()
    max_delta = torch.zeros_like(x)

    for i in range(restarts):
        delta = torch.zeros_like(x).uniform_(-eps, eps)
        delta.data = clamp(delta, clip_min - x, clip_max - x)
        delta.requires_grad = True

        for _ in range(attack_iters):
            logits = model(x + delta)
            # index = torch.where(output.max(1)[1] == y)
            index = torch.where(logits.argmax(dim=1) == y)  # get the correct predictions, pgd performed only on them
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(logits, y)
            loss.backward()

            # select & update
            grad = delta.grad.detach()
            delta_update = (delta[index] + eps_iter * torch.sign(grad[index])).clamp_(-eps, eps)
            delta_update = clamp(delta_update, clip_min - x[index], clip_max - x[index])

            # write back
            delta.data[index] = delta_update
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(x + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def eval_epoch(model, data_loader, args, adversarial=False):
    """Self-implemented PGD evaluation"""
    eps = eval(args.epsilon) / std_
    eps_iter = eval(args.pgd_epsilon_iter) / std_
    attack_iters = 40
    restarts = 1

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')
    model.eval()
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        if adversarial is True:
            delta = attack_pgd(model, x, y, eps, eps_iter, attack_iters, restarts)
        else:
            delta = 0.

        with torch.no_grad():
            logits = model(x + delta)
            loss = F.cross_entropy(logits, y)

            loss_meter.update(loss.item(), x.size(0))
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            acc_meter.update(acc, x.size(0))

    return loss_meter.avg, acc_meter.avg


class CorruptionDataset(Dataset):
    # for cifar10 and cifar100
    def __init__(self, x, y, transform=None):
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x[item]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[item]

    def __len__(self):
        return self.x.shape[0]


def eval_c(classifier, base_path, args):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([mean_] * 3, [std_] * 3)])

        x = np.load(base_path + corruption + '.npy')
        y = np.load(base_path + 'labels.npy').astype(np.int64)
        dataset = CorruptionDataset(x, y, transform=preprocess)

        test_loader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        test_loss, test_acc = eval_epoch(classifier, test_loader, args)
        corruption_accs.append(test_acc)

    return np.mean(corruption_accs)


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(config_path='config.yaml')
def run(args: DictConfig) -> None:
    # Load datasets
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4)])
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([mean_] * 3, [std_] * 3)])
    test_transform = preprocess

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            data_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            data_dir, train=False, transform=test_transform, download=True)
        base_c_path = os.path.join(data_dir, 'CIFAR-10-C/')
        args.n_classes = 10
    else:
        train_data = datasets.CIFAR100(
            data_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            data_dir, train=False, transform=test_transform, download=True)

        base_c_path = os.path.join(data_dir, 'CIFAR-100-C/')
        args.n_classes = 100

    train_data = AugMixDataset(train_data, preprocess, args, args.no_jsd)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    # Create model
    if args.model == 'densenet':
        classifier = densenet(num_classes=args.n_classes)
    elif args.model == 'wide_resnet':
        n_layers = 40
        widen_factor = 2
        droprate = 0.
        classifier = WideResNet(n_layers, args.n_classes, widen_factor, droprate)
    elif args.model == 'resnext':
        classifier = resnext29(num_classes=args.n_classes)

    classifier = classifier.to(args.device)
    logger.info('Model: {}, # parameters: {}'.format(args.model, cal_parameters(classifier)))

    cudnn.benchmark = True
    classifier = torch.nn.DataParallel(classifier).to(args.device)

    if args.inference:
        classifier.load_state_dict(torch.load('{}_{}.pth'.format(args.model, args.augmentation_type)))
        test_loss, test_acc = eval_epoch(classifier, test_loader, args, adversarial=False)
        logger.info('Clean Test CE:{:.4f}, acc:{:.4f}'.format(test_loss, test_acc))
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True)

        best_acc = 0
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                args.n_epochs * len(train_loader),
                1,  # lr_lambda computes multiplicative factor
                1e-6 / args.learning_rate))

        for epoch in range(args.n_epochs):
            if args.augmentation_type:
                loss, ce_loss, js_loss, acc = train_epoch_advmix(classifier, train_loader,  args, optimizer, scheduler)
            else:
                loss, ce_loss, js_loss, acc = train_epoch(classifier, train_loader,  args, optimizer, scheduler)

            lr = scheduler.get_lr()[0]
            logger.info('Epoch {}, lr:{:.4f}, loss:{:.4f}, CE:{:.4f}, JS:{:.4f}, Acc:{:.4f}'
                        .format(epoch + 1, lr, loss, ce_loss, js_loss, acc))

            test_loss, test_acc = eval_epoch(classifier, test_loader, args, adversarial=False)
            logger.info('Test CE:{:.4f}, acc:{:.4f}'.format(test_loss, test_acc))
            if epoch > 10 and epoch % 2 == 0:
                adv_loss, adv_acc = eval_epoch(classifier, test_loader, args, adversarial=True)
                logger.info('Adversarial evaluation, CE:{:.4f}, acc:{:.4f}'.format(adv_loss, adv_acc))

            if test_acc > best_acc:
                best_acc = test_acc
                logging.info('===> New optimal, save checkpoint ...')

                torch.save(classifier.state_dict(), '{}_{}.pth'.format(args.model, args.augmentation_type))

    test_c_acc = eval_c(classifier, base_c_path, args)
    logger.info('Mean Corruption Error:{:.4f}'.format(test_c_acc))


if __name__ == '__main__':
    run()

