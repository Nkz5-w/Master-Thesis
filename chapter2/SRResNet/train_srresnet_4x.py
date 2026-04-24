import warnings
import time
import torch.optim
warnings.filterwarnings('ignore', category=UserWarning)
from models import SRResNet
from datasets import SRDataset
from torch import nn
import torch.backends.cudnn as cudnn
from utils import *

mean = 0.0
std = 0.1

# Data parameters
data_folder = './'
crop_size = 96
scaling_factor = 4

# Model parameters
large_kernel_size = 9
small_kernel_size = 3
n_channels = 64
n_blocks = 16

# Learning parameters
checkpoint = 'checkpoints/checkpoint_srresnet_4_single_ckm_pathloss.pth.tar'
batch_size = 32
iterations = 100000
workers = 4
print_freq = 100
lr = 1e-4
grad_clip = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def main():
    global checkpoint
    print('\nCurrent scaling factor: %s' % scaling_factor)

    # Initialize model or load checkpoint
    if checkpoint is None:
        print('Initializing random model...')
        model = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                         n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        print('Done!')
    else:
        print("Loading pretrained SRResNet model...")
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        print('Done!')

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    train_dataset = SRDataset(data_folder,
                              split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='[0, 1]',
                              hr_img_type='[0, 1]')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)

    print('Training start:')
    for epoch in range(0, epochs):
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                   'checkpoints/checkpoint_srresnet_4_single_ckm_pathloss.pth.tar')
        print('Model saved successfully!')
    print('Training completed!')


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        lr_imgs = lr_imgs[:, 0:1, :, :]  # (N , 1, w, h), in [0, 1]
        hr_imgs = hr_imgs[:, 0:1, :, :]  # (N , 1, w, h), in [0, 1]

        sr_imgs = model(lr_imgs)  # (N, 1, w, h), in [0, 1]
        loss = criterion(sr_imgs, hr_imgs)
        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()
        losses.update(loss.item(), lr_imgs.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0 and i != 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                batch_time=batch_time,
                                                                data_time=data_time, loss=losses))
    del lr_imgs, hr_imgs, sr_imgs


if __name__ == '__main__':
    main()
