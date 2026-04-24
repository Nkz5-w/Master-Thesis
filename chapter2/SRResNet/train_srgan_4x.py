import warnings

warnings.filterwarnings('ignore', category=UserWarning)
import time
import torch.backends.cudnn as cudnn
from torch import nn
from models import Generator, Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import *

mean = 0.0
std = 0.1

# Data parameters
data_folder = './'
crop_size = 96
scaling_factor = 4

# Generator parameters
large_kernel_size_g = 9
small_kernel_size_g = 3
n_channels_g = 64
n_blocks_g = 16
srresnet_checkpoint = 'checkpoints/checkpoint_srresnet_4_noise.pth.tar'

# Discriminator parameters
kernel_size_d = 3
n_channels_d = 64
n_blocks_d = 8
fc_size_d = 1024

# Learning parameters
checkpoint = 'checkpoints/checkpoint_srgan_4_noise.pth.tar'
batch_size = 32
iterations = 10000
workers = 4
vgg19_i = 5
vgg19_j = 4
beta = 1e-3
print_freq = 100
lr = 1e-3
grad_clip = 1.0

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def main():
    global epoch, checkpoint, srresnet_checkpoint
    if checkpoint is None:
        generator = Generator(large_kernel_size=large_kernel_size_g,
                              small_kernel_size=small_kernel_size_g,
                              n_channels=n_channels_g,
                              n_blocks=n_blocks_g,
                              scaling_factor=scaling_factor)
        print('Loading pre-trained SRResNet...')
        generator.initialize_with_srresnet(srresnet_checkpoint=srresnet_checkpoint)
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                       lr=lr)

        discriminator = Discriminator(kernel_size=kernel_size_d,
                                      n_channels=n_channels_d,
                                      n_blocks=n_blocks_d,
                                      fc_size=fc_size_d)
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                       lr=lr)
    else:
        print("Loading pretrained SRGAN...")
        checkpoint = torch.load(checkpoint)
        generator = checkpoint['generator']
        discriminator = checkpoint['discriminator']
        optimizer_g = checkpoint['optimizer_g']
        optimizer_d = checkpoint['optimizer_d']
        print("Done!")

    # Truncated VGG19 network to be used in the loss calculation
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.eval()

    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    train_dataset = SRDataset(data_folder,
                              split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='[0, 1]',
                              hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)

    print('Training start:')
    for epoch in range(0, epochs):
        train(train_loader=train_loader,
              generator=generator,
              discriminator=discriminator,
              truncated_vgg19=truncated_vgg19,
              content_loss_criterion=content_loss_criterion,
              adversarial_loss_criterion=adversarial_loss_criterion,
              optimizer_g=optimizer_g,
              optimizer_d=optimizer_d,
              epoch=epoch)

        torch.save({'epoch': epoch,
                    'generator': generator,
                    'discriminator': discriminator,
                    'optimizer_g': optimizer_g,
                    'optimizer_d': optimizer_d},
                   'checkpoints/checkpoint_srgan_4_noise.pth.tar')
        print('Model saved successfully!')
    print('Training completed!')


def train(train_loader, generator, discriminator, truncated_vgg19, content_loss_criterion, adversarial_loss_criterion,
          optimizer_g, optimizer_d, epoch):
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_c = AverageMeter()  # content loss
    losses_a = AverageMeter()  # adversarial loss in the generator
    losses_d = AverageMeter()  # adversarial loss in the discriminator

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        lr_imgs = lr_imgs.to(device)  # (N, 1, w/k, w/k), in [0, 1]
        hr_imgs = hr_imgs.to(device)  # (N, 1, w, h), imagenet-normed

        lr_imgs = lr_imgs[:, 0:1, :, :]  # (N , 1, w, h), in [0, 1]

        # 创建一个掩码，表示非零点的位置
        mask = lr_imgs != 0  # mask 为 True 的位置表示非零点

        # 生成与 lr_imgs 相同形状的高斯噪声
        noise = torch.randn_like(lr_imgs) * std + mean

        # 只在非零点的地方添加噪声
        lr_imgs = lr_imgs + mask * noise

        # GENERATOR UPDATE
        # Generate
        sr_imgs = generator(lr_imgs)  # (N, 1, w, h), in [0, 1]
        sr_imgs = sr_imgs.repeat(1, 3, 1, 1)  # (N, 3, w, h)
        sr_imgs = convert_image(sr_imgs, source='[0, 1]', target='imagenet-norm')  # (N, 3, w, h), imagenet-normed

        # Calculate VGG feature maps for the super-resolved (SR) and high resolution (HR) images
        sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
        hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()  # detached because they're constant, targets

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)

        # Calculate the Perceptual loss
        content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss

        optimizer_g.zero_grad()
        perceptual_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer_g, grad_clip)

        # Update generator
        optimizer_g.step()

        # Keep track of loss
        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

        # DISCRIMINATOR UPDATE

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        # But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just
        # use that here? Because, if we used that, we'd be back-propagating (finding gradients) over the G too when
        # backward() is called It's actually faster to detach the SR images from the G and forward-prop again,
        # than to back-prop. over the G unnecessarily See FAQ section in the tutorial

        # Binary Cross-Entropy loss
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                           adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

        optimizer_d.zero_grad()
        adversarial_loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer_d, grad_clip)

        optimizer_d.step()

        losses_d.update(adversarial_loss.item(), hr_imgs.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----'
                  'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----'
                  'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch,
                                                                          i,
                                                                          len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss_c=losses_c,
                                                                          loss_a=losses_a,
                                                                          loss_d=losses_d))

    del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated


if __name__ == '__main__':
    main()
