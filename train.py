import os
import logging
import hydra
from omegaconf import DictConfig

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import augmentations

from utils import AverageMeter, get_model

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
        print(self.args)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), aug(x, self.preprocess, self.args), aug(x, self.preprocess, self.args))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def train_epoch(classifier, train_loader, args, optimizer, scheduler):
    """Train for one epoch."""
    classifier.train()
    loss_meter = AverageMeter('loss')
    ce_meter = AverageMeter('ce_loss')
    js_meter = AverageMeter('js_loss')
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.no_jsd:
            images = images.to(args.device)
            targets = targets.to(args.device)
            logits = classifier(images)
            loss = F.cross_entropy(logits, targets)
        else:
            images_all = torch.cat(images, 0).to(args.device)
            targets = targets.to(args.device)
            logits_all = classifier(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
              logits_all, images[0].size(0))

            # Cross-entropy is only computed on clean images
            ce_loss = F.cross_entropy(logits_clean, targets)

            p_clean, p_aug1, p_aug2 = F.softmax(
              logits_clean, dim=1), F.softmax(
                  logits_aug1, dim=1), F.softmax(
                      logits_aug2, dim=1)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            js_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

            loss = ce_loss + 12 * js_loss
            loss_meter.update(loss.item(), images[0].size(0))
            ce_meter.update(ce_loss.item(), images[0].size(0))
            js_meter.update(js_loss.item(), images[0].size(0))
        loss.backward()
        optimizer.step()
        scheduler.step()
        break
        # loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    return loss_meter.avg, ce_meter.avg, js_meter.avg


# Note that we don't use cifar10 specific normalization, so generally use 0.5 as mean and std.

mean_ = 0.5
std_ = 0.5

clip_min = -1.
clip_max = 1.


def eval_epoch(model, data_loader, args, adversarial=False):
    """Self-implemented PGD evaluation"""
    eps = eval(args.epsilon) / std_
    eps_iter = eval(args.pgd_epsilon_iter) / std_
    attack_iters = 50
    restarts = 2

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


def eval_c(classifier, test_data, base_path, args):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        test_loss, test_acc = eval_epoch(classifier, test_loader)
        corruption_accs.append(test_acc)
        # print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        #     corruption, test_loss, 100 - 100. * test_acc))

    return np.mean(corruption_accs)


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


@hydra.main(config_path='config.yaml')
def run(args: DictConfig) -> None:

    # print(args)
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

    classifier = get_model(args.classifier_name, args.n_classes).to(args.device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)

    cudnn.benchmark = True

    best_acc = 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.n_epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    for epoch in range(args.n_epochs):
        loss, ce_loss, js_loss = train_epoch(classifier, train_loader,  args, optimizer, scheduler)
        logger.info('Epoch {}, loss:{:.4f}, CE:{:.4f}, JS:{:.4f}'.format(epoch + 1, loss, ce_loss, js_loss))

        test_loss, test_acc = eval_epoch(classifier, test_loader, args, adversarial=False)
        logger.info('Test loss:{:.4f}, acc:{:.4f}'.format(test_loss, test_acc))
        if test_acc > best_acc:
            best_acc = test_acc
            logging.info('===> New optimal, save checkpoint ...')

            torch.save(classifier.state_dict(), '{}.pth'.format(args.classifier_name))

    test_c_acc = eval_c(classifier, test_data, base_c_path, args)
    logger.info('Mean Corruption Error:{:.4f}'.format(test_c_acc))


if __name__ == '__main__':
    run()

