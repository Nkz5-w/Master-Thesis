import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
from torchvision.transforms import InterpolationMode
import pandas as pd
import lpips
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from datasets import SRDataset
from torchvision import transforms
import os
from contextlib import redirect_stdout

# def add_gaussian_noise_imagenet_normed(image, snr):
#     """
#     给 ImageNet 归一化的图像添加高斯噪声，并根据 SNR 控制噪声强度。
#     参数：
#     image (torch.Tensor): 形状为 (1, 3, w, h) 的 ImageNet 归一化图像。
#     snr (float): 信噪比，越大表示噪声越小。
#     返回：
#     noisy_image (torch.Tensor): 添加了高斯噪声的图像。
#     """
#     # ImageNet 归一化的均值和标准差
#     mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
#     # 反归一化，将图像恢复到原始像素值范围
#     un_normalized_image = image * std + mean
#     # 计算信号能量
#     signal_power = torch.mean(un_normalized_image ** 2)
#     # 根据 SNR 计算噪声标准差
#     sigma = torch.sqrt(signal_power / (10 ** (snr / 10)))
#     # 生成与图像形状相同的高斯噪声
#     noise = sigma * torch.randn_like(un_normalized_image)
#     # 将噪声加到未归一化的图像上
#     noisy_image = un_normalized_image + noise
#     # 再次归一化，确保输出仍符合 ImageNet 归一化标准
#     noisy_image = (noisy_image - mean) / std
#     return noisy_image

with redirect_stdout(open(os.devnull, 'w')):
    loss_lp = lpips.LPIPS(net='alex')  # 这里的输出将被抑制
data_folder = "./"
test_data_names = ["test"]
scaling_factor = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_lp.to(device)
# snr = 0

# Model checkpoints
# srgan_checkpoint = "./checkpoints/checkpoint_srgan_2_new.pth.tar"
# srresnet_checkpoint = "./checkpoints/checkpoint_srresnet_2_new.pth.tar"
# srgan_checkpoint = "./checkpoints/checkpoint_srgan_4_new.pth.tar"
# srresnet_checkpoint = "./checkpoints/checkpoint_srresnet_4_new.pth.tar"
# srgan_checkpoint = "./checkpoints/checkpoint_srgan_8_new.pth.tar"
# srresnet_checkpoint = "./checkpoints/checkpoint_srresnet_8_aoa.pth.tar"
# srgan_checkpoint = "./checkpoints/checkpoint_srgan_16_new.pth.tar"
# srresnet_checkpoint = "./checkpoints/checkpoint_srresnet_16_aoa.pth.tar"
srgan_checkpoint = "./checkpoints/checkpoint_srgan_4_single_ckm_pathloss.pth.tar"
srresnet_checkpoint = "./checkpoints/checkpoint_srresnet_4_single_ckm_pathloss.pth.tar"


def main():
    srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
    srresnet.eval()
    model_srresnet = srresnet

    srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
    srgan_generator.eval()
    model = srgan_generator

    print('\nCurrent scaling factor: %s' % scaling_factor)
    for test_data_name in test_data_names:
        test_dataset = SRDataset(data_folder,
                                 split='test',
                                 crop_size=128,
                                 scaling_factor=scaling_factor,
                                 lr_img_type='[0, 1]',
                                 hr_img_type='[0, 1]',
                                 test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        # Keep track of the PSNRs and the SSIMs across batches
        PSNRs = AverageMeter()
        PSNRs_RES = AverageMeter()
        PSNRs_NN = AverageMeter()
        PSNRs_BIC = AverageMeter()
        SSIMs = AverageMeter()
        SSIMs_RES = AverageMeter()
        SSIMs_NN = AverageMeter()
        SSIMs_BIC = AverageMeter()
        LPIPSs = AverageMeter()
        LPIPSs_RES = AverageMeter()
        LPIPSs_NN = AverageMeter()
        LPIPSs_BIC = AverageMeter()
        MSEs = AverageMeter()
        MSEs_RES = AverageMeter()
        MSEs_NN = AverageMeter()
        MSEs_BIC = AverageMeter()
        RMSEs = AverageMeter()
        RMSEs_RES = AverageMeter()
        RMSEs_NN = AverageMeter()
        RMSEs_BIC = AverageMeter()

        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                lr_imgs = lr_imgs.to(device)  # (1, 3, w/k, h/k), [0, 1]
                hr_imgs = hr_imgs.to(device)  # (1, 3, w, h), [0, 1]

                lr_imgs = lr_imgs[:, 0:1, :, :]  # (1, 1, w/k, h/k), [0, 1]
                hr_imgs = hr_imgs[:, 0:1, :, :]  # (1, 1, w, h), [0, 1]

                # 加噪声
                # lr_imgs = add_gaussian_noise_imagenet_normed(lr_imgs, snr)

                # SRResNet
                sr_imgs = model(lr_imgs).squeeze()  # (w, h), [0, 1]

                # SRGAN
                sr_res_imgs = model_srresnet(lr_imgs).squeeze()  # (w, h), [0, 1]

                # Benchline
                downsample = transforms.Resize(
                    (int(hr_imgs.shape[2] / scaling_factor), int(hr_imgs.shape[3] / scaling_factor)),
                    interpolation=InterpolationMode.BICUBIC)
                upsample_bic = transforms.Resize(
                    (int(hr_imgs.shape[2]), int(hr_imgs.shape[3])),
                    interpolation=InterpolationMode.BICUBIC)
                upsample_nn = transforms.Resize(
                    (int(hr_imgs.shape[2]), int(hr_imgs.shape[3])),
                    interpolation=InterpolationMode.NEAREST)

                lr_imgs_t = downsample(hr_imgs)  # (1, 1, w/k, h/k), [0, 1]


                # Bicubic Upsampling
                bicubic_imgs = upsample_bic(lr_imgs_t).squeeze()  # (w, h), [0, 1]

                # Nearest Neighbour Upsampling
                nn_imgs = upsample_nn(lr_imgs_t).squeeze()  # (w, h), [0, 1]

                hr_imgs = hr_imgs.squeeze()  # (w, h), [0, 1]

                # # radio_map_seer dB, [-147, -47]
                # hr_imgs_1 = 100 * hr_imgs - 147
                # sr_imgs_1 = 100 * sr_imgs - 147
                # bicubic_imgs_1 = 100 * bicubic_imgs - 147
                # nearest_imgs_1 = 100 * nn_imgs - 147
                # sr_res_imgs_1 = 100 * sr_res_imgs - 147

                # # ckm_path_loss dB, [-250, -50]
                hr_imgs_1 = 200 * hr_imgs - 250
                nearest_imgs_1 = 200 * nn_imgs - 250
                bicubic_imgs_1 = 200 * bicubic_imgs - 250
                sr_imgs_1 = 200 * sr_imgs - 250
                sr_res_imgs_1 = 200 * sr_res_imgs - 250

                # # ckm_aoa °, [-200°, 180°]
                # hr_imgs_1 = 380 * hr_imgs - 200
                # sr_imgs_1 = 380 * sr_imgs - 200
                # bicubic_imgs_1 = 380 * bicubic_imgs - 200
                # nearest_imgs_1 = 380 * nn_imgs - 200
                # sr_res_imgs_1 = 380 * sr_res_imgs - 200

                rmse = np.sqrt(mean_squared_error(hr_imgs_1.cpu().numpy(), sr_imgs_1.cpu().numpy()))
                rmse_res = np.sqrt(mean_squared_error(hr_imgs_1.cpu().numpy(), sr_res_imgs_1.cpu().numpy()))
                rmse_bic = np.sqrt(mean_squared_error(hr_imgs_1.cpu().numpy(), bicubic_imgs_1.cpu().numpy()))
                rmse_nn = np.sqrt(mean_squared_error(hr_imgs_1.cpu().numpy(), nearest_imgs_1.cpu().numpy()))

                # Calculate PSNRs and SSIMs(RGB)
                # hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)
                # sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                #     0)  # (w, h), in y-channel
                # bicubic_imgs_y = convert_image(bicubic_imgs, source='[-1, 1]', target='y-channel').squeeze(
                #     0)  # (w, h), in y-channel
                # nearest_imgs_y = convert_image(nn_imgs, source='[-1, 1]', target='y-channel').squeeze(
                #     0)  # (w, h), in y-channel
                # sr_res_imgs_y = convert_image(sr_res_imgs, source='[-1, 1]', target='y-channel').squeeze(
                #     0)  # (w, h), in y-channel

                # Calculate PSNRs and SSIMs(单通道)
                hr_imgs_y = convert_image(hr_imgs, source='[0, 1]', target='[0, 255]').squeeze()  # (w,h),[0,255]
                sr_imgs_y = convert_image(sr_imgs, source='[0, 1]', target='[0, 255]').squeeze()
                bicubic_imgs_y = convert_image(bicubic_imgs, source='[0, 1]', target='[0, 255]').squeeze()
                nearest_imgs_y = convert_image(nn_imgs, source='[0, 1]', target='[0, 255]').squeeze()
                sr_res_imgs_y = convert_image(sr_res_imgs, source='[0, 1]', target='[0, 255]').squeeze()

                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                               data_range=255.)
                psnr_res = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_res_imgs_y.cpu().numpy(),
                                                   data_range=255.)
                psnr_bic = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), bicubic_imgs_y.cpu().numpy(),
                                                   data_range=255.)
                psnr_nn = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), nearest_imgs_y.cpu().numpy(),
                                                  data_range=255.)

                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                             data_range=255.)
                ssim_res = structural_similarity(hr_imgs_y.cpu().numpy(), sr_res_imgs_y.cpu().numpy(),
                                                 data_range=255.)
                ssim_bic = structural_similarity(hr_imgs_y.cpu().numpy(), bicubic_imgs_y.cpu().numpy(),
                                                 data_range=255.)
                ssim_nn = structural_similarity(hr_imgs_y.cpu().numpy(), nearest_imgs_y.cpu().numpy(),
                                                data_range=255.)

                mse = mean_squared_error(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy())
                mse_res = mean_squared_error(hr_imgs_y.cpu().numpy(), sr_res_imgs_y.cpu().numpy())
                mse_bic = mean_squared_error(hr_imgs_y.cpu().numpy(), bicubic_imgs_y.cpu().numpy())
                mse_nn = mean_squared_error(hr_imgs_y.cpu().numpy(), nearest_imgs_y.cpu().numpy())

                # Calculate LPIPS
                lpips = loss_lp(hr_imgs_y, sr_imgs_y)
                lpips_res = loss_lp(hr_imgs_y, sr_res_imgs_y)
                lpips_bic = loss_lp(hr_imgs_y, bicubic_imgs_y)
                lpips_nn = loss_lp(hr_imgs_y, nearest_imgs_y)
                lpips = np.round(lpips.cpu().numpy().flatten(), 6)
                lpips_res = np.round(lpips_res.cpu().numpy().flatten(), 6)
                lpips_bic = np.round(lpips_bic.cpu().numpy().flatten(), 6)
                lpips_nn = np.round(lpips_nn.cpu().numpy().flatten(), 6)

                # Updating outcomes
                PSNRs.update(psnr, lr_imgs.size(0))
                PSNRs_RES.update(psnr_res, lr_imgs.size(0))
                PSNRs_BIC.update(psnr_bic, lr_imgs.size(0))
                PSNRs_NN.update(psnr_nn, lr_imgs.size(0))

                SSIMs.update(ssim, lr_imgs.size(0))
                SSIMs_RES.update(ssim_res, lr_imgs.size(0))
                SSIMs_BIC.update(ssim_bic, lr_imgs.size(0))
                SSIMs_NN.update(ssim_nn, lr_imgs.size(0))

                LPIPSs.update(lpips, lr_imgs.size(0))
                LPIPSs_RES.update(lpips_res, lr_imgs.size(0))
                LPIPSs_BIC.update(lpips_bic, lr_imgs.size(0))
                LPIPSs_NN.update(lpips_nn, lr_imgs.size(0))

                MSEs.update(mse, lr_imgs.size(0))
                MSEs_RES.update(mse_res, lr_imgs.size(0))
                MSEs_BIC.update(mse_bic, lr_imgs.size(0))
                MSEs_NN.update(mse_nn, lr_imgs.size(0))

                RMSEs.update(rmse, lr_imgs.size(0))
                RMSEs_RES.update(rmse_res, lr_imgs.size(0))
                RMSEs_BIC.update(rmse_bic, lr_imgs.size(0))
                RMSEs_NN.update(rmse_nn, lr_imgs.size(0))

        # Print average PSNR, SSIM and LPIPS
        models = ['NN', 'Bicubic', 'SRGAN', 'SRResNet']
        psnr_values = [PSNRs_NN.avg, PSNRs_BIC.avg, PSNRs.avg, PSNRs_RES.avg]
        ssim_values = [SSIMs_NN.avg, SSIMs_BIC.avg, SSIMs.avg, SSIMs_RES.avg]
        lpips_values = [LPIPSs_NN.avg.item(), LPIPSs_BIC.avg.item(), LPIPSs.avg.item(), LPIPSs_RES.avg.item()]
        mse_values = [MSEs_NN.avg, MSEs_BIC.avg, MSEs.avg, MSEs_RES.avg]
        rmse_values = [RMSEs_NN.avg, RMSEs_BIC.avg, RMSEs.avg, RMSEs_RES.avg]

        # # Print average PSNR, SSIM and LPIPS
        # models = ['NN', 'Bicubic', 'SRResNet']
        # psnr_values = [PSNRs_NN.avg, PSNRs_BIC.avg, PSNRs_RES.avg]
        # ssim_values = [SSIMs_NN.avg, SSIMs_BIC.avg, SSIMs_RES.avg]
        # lpips_values = [LPIPSs_NN.avg.item(), LPIPSs_BIC.avg.item(), LPIPSs_RES.avg.item()]
        # mse_values = [MSEs_NN.avg, MSEs_BIC.avg, MSEs_RES.avg]
        # rmse_values = [RMSEs_NN.avg, RMSEs_BIC.avg, RMSEs_RES.avg]

        # 创建一个DataFrame
        data = {
            'Model': models,
            'PSNR': psnr_values,
            'SSIM': ssim_values,
            'LPIPS': lpips_values,
            'MSE': mse_values,
            'RMSE': rmse_values
        }
        df = pd.DataFrame(data)

        # 打印输出表格

        print(df.to_string(index=False,
                           formatters={'PSNR': '{:.4f}'.format, 'SSIM': '{:.4f}'.format, 'LPIPS': '{:.4f}'.format,
                                       'MSE': '{:.4f}'.format, 'RMSE': '{:.4f}'.format})
              )

    print("\n")


if __name__ == "__main__":
    main()
