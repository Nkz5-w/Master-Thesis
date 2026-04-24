clc; clear; close all;

% =========================
% 放大倍率（必须有！）
% =========================
scale = [2, 4, 8, 16];

% =========================
% 数据
% =========================
PSNR_NN      = [30.3046, 25.4620, 22.0985, 19.2215];
PSNR_Bicubic = [31.7628, 27.1981, 23.9130, 20.4104];
PSNR_SRGAN   = [41.5899, 35.8248, 27.2767, 20.6039];
PSNR_SRRes   = [45.8390, 41.5307, 33.0113, 22.0010];

SSIM_NN      = [0.9379, 0.8557, 0.7480, 0.6575];
SSIM_Bicubic = [0.9406, 0.8627, 0.7557, 0.6440];
SSIM_SRGAN   = [0.9893, 0.9856, 0.9036, 0.6696];
SSIM_SRRes   = [0.9945, 0.9900, 0.9536, 0.7149];

LPIPS_NN      = [0.0623, 0.1307, 0.1966, 0.2718];
LPIPS_Bicubic = [0.1384, 0.2717, 0.4047, 0.4521];
LPIPS_SRGAN   = [0.0043, 0.0071, 0.0825, 0.2409];
LPIPS_SRRes   = [0.0023, 0.0071, 0.0552, 0.2127];

RMSE_NN      = [3.6023, 6.2828, 9.1818, 12.7943];
RMSE_Bicubic = [3.0542, 5.1818, 7.5097, 11.2743];
RMSE_SRGAN   = [1.1331, 1.6089, 4.0555, 10.1512];
RMSE_SRRes   = [0.7217, 1.0628, 2.7926, 9.5922];

% =========================
% 画图
% =========================
figure('Units','pixels','Position',[100 100 400 350]);

h1 = plot(scale, PSNR_NN, '-o','LineWidth',1.2); hold on;
h2 = plot(scale, PSNR_Bicubic, '-s','LineWidth',1.2);
h3 = plot(scale, PSNR_SRGAN, '-^','LineWidth',1.2);
h4 = plot(scale, PSNR_SRRes, '-d','LineWidth',1.2);

h = [h1 h2 h3 h4];
for i = 1:4
    set(h(i),'MarkerFaceColor',h(i).Color);
end

grid on;
xlabel('k', 'FontName', 'Times New Roman', 'FontAngle', 'italic');
ylabel('PSNR (dB)', 'FontName', 'Times New Roman');
legend('NN','Bicubic','SRGAN','SRResNet','FontName', 'Times New Roman');

figure('Units','pixels','Position',[100 100 400 350]);
h1 = plot(scale, SSIM_NN, '-o','LineWidth',1.2); hold on;
h2 = plot(scale, SSIM_Bicubic, '-s','LineWidth',1.2);
h3 = plot(scale, SSIM_SRGAN, '-^','LineWidth',1.2);
h4 = plot(scale, SSIM_SRRes, '-d','LineWidth',1.2);
% 统一设置为“实心且与线同色”
h = [h1 h2 h3 h4];
for i = 1:length(h)
    set(h(i), 'MarkerFaceColor', h(i).Color);
end
grid on;
xlabel('k', 'FontName', 'Times New Roman', 'FontAngle', 'italic');
ylabel('SSIM', 'FontName', 'Times New Roman');
legend('NN','Bicubic','SRGAN','SRResNet','FontName', 'Times New Roman');

figure('Units','pixels','Position',[100 100 400 350]);
h1 = plot(scale, LPIPS_NN, '-o','LineWidth',1.2); hold on;
h2=plot(scale, LPIPS_Bicubic, '-s','LineWidth',1.2);
h3=plot(scale, LPIPS_SRGAN, '-^','LineWidth',1.2);
h4=plot(scale, LPIPS_SRRes, '-d','LineWidth',1.2);
% 统一设置为“实心且与线同色”
h = [h1 h2 h3 h4];
for i = 1:length(h)
    set(h(i), 'MarkerFaceColor', h(i).Color);
end
grid on; 
xlabel('k', 'FontName', 'Times New Roman', 'FontAngle', 'italic');
ylabel('LPIPS', 'FontName', 'Times New Roman');
legend('NN','Bicubic','SRGAN','SRResNet','FontName', 'Times New Roman');

figure('Units','pixels','Position',[100 100 400 350]);
h1=plot(scale, RMSE_NN, '-o','LineWidth',1.2); hold on;
h2=plot(scale, RMSE_Bicubic, '-s','LineWidth',1.2);
h3=plot(scale, RMSE_SRGAN, '-^','LineWidth',1.2);
h4=plot(scale, RMSE_SRRes, '-d','LineWidth',1.2);
% 统一设置为“实心且与线同色”
h = [h1 h2 h3 h4];
for i = 1:length(h)
    set(h(i), 'MarkerFaceColor', h(i).Color);
end
grid on; 
xlabel('k', 'FontName', 'Times New Roman', 'FontAngle', 'italic');
ylabel('RMSE(dB)', 'FontName', 'Times New Roman');
legend('NN','Bicubic','SRGAN','SRResNet','FontName', 'Times New Roman');

set(gca,'Position',[0.13 0.15 0.75 0.75]); % 关键
set(gcf,'Color','w');

set(gcf,'PaperPositionMode','auto');
print(gcf,'PSNR','-dpng','-r300');