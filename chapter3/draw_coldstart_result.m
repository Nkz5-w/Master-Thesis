clear; close all;
load('BER_ideal_trimmed.mat')
load('BER_lmmse_trimmed.mat')
load('BER_lmmse_avg_trimmed.mat')
load('BER_lmmse_theo_trimmed.mat')
load('BER_ls_trimmed.mat')
load('BER_lmmse_dft_trimmed.mat')

load('MSE_lmmse_all.mat')
load('MSE_lmmse_avg_all.mat')
load('MSE_lmmse_theo_all.mat')
load('MSE_ls_all.mat')
load('MSE_lmmse_dft_all.mat')

num_frames = 799;

% =========================
% 三层平滑参数（关键！）
% =========================
window1 = 250;   % 第一层：主平滑（log域）
window2 = 100;    % 第二层：中等平滑
window3 = 50;    % 第三层：轻微抛光（去最后抖动）

% =========================
% 第一层：log域（核心）
% =========================
smooth_log = @(x) 10.^movmean(log10(x), window1);

BER_ls_s1         = smooth_log(BER_ls_trimmed);
BER_lmmse_s1      = smooth_log(BER_lmmse_trimmed);
BER_lmmse_avg_s1  = smooth_log(BER_lmmse_avg_trimmed);
BER_lmmse_theo_s1 = smooth_log(BER_lmmse_theo_trimmed);
BER_ideal_s1      = smooth_log(BER_ideal_trimmed);
BER_lmmse_dft_s1  = smooth_log(BER_lmmse_dft_trimmed);  % 添加 DFT

% =========================
% 第二层：线性域（中等）
% =========================
BER_ls_s2         = movmean(BER_ls_s1, window2);
BER_lmmse_s2      = movmean(BER_lmmse_s1, window2);
BER_lmmse_avg_s2  = movmean(BER_lmmse_avg_s1, window2);
BER_lmmse_theo_s2 = movmean(BER_lmmse_theo_s1, window2);
BER_ideal_s2      = movmean(BER_ideal_s1, window2);
BER_lmmse_dft_s2  = movmean(BER_lmmse_dft_s1, window2);  % 添加 DFT

% =========================
% 第三层：线性域（轻微抛光）
% =========================
BER_ls_smooth         = movmean(BER_ls_s2, window3);
BER_lmmse_smooth      = movmean(BER_lmmse_s2, window3);
BER_lmmse_avg_smooth  = movmean(BER_lmmse_avg_s2, window3);
BER_lmmse_theo_smooth = movmean(BER_lmmse_theo_s2, window3);
BER_ideal_smooth      = movmean(BER_ideal_s2, window3);
BER_lmmse_dft_smooth  = movmean(BER_lmmse_dft_s2, window3);  % 添加 DFT

% =========================
% 3. 范围
% =========================
all_data = [BER_ls_smooth, BER_lmmse_smooth, ...
            BER_lmmse_avg_smooth, BER_lmmse_theo_smooth, ...
            BER_ideal_smooth, BER_lmmse_dft_smooth];  % 添加 DFT

ymin = min(all_data(all_data > 0));
ymax = max(all_data);

% =========================
% 4. 绘图
% =========================

% index = 0:1:num_frames;
% figure('Position', [100, 100, 600, 450]);
% semilogy(index, BER_ls_smooth, '-', 'LineWidth', 1.2, 'Color', [0,0.447,0.741]); hold on;
% semilogy(index, BER_lmmse_smooth, '-', 'LineWidth', 1.2, 'Color', [0.850,0.325,0.098]);
% semilogy(index, BER_lmmse_dft_smooth, '-', 'LineWidth', 1.2, 'Color', [0.929,0.694,0.125]);
% semilogy(index, BER_lmmse_avg_smooth, '-', 'LineWidth', 1.2, 'Color', [0.494,0.184,0.556]);
% semilogy(index, BER_lmmse_theo_smooth, '-', 'LineWidth', 1.2, 'Color', [0.466,0.674,0.188]);
% semilogy(index, BER_ideal_smooth, 'r-', 'LineWidth', 1.2);
% hold off;
index = 0:1:num_frames;
figure('Position', [100, 100, 600, 450]);

% 定义下采样间隔（例如每10个点显示一个标记）
decimation_factor = 100;  % 可根据数据密度调整

% 选择要显示标记的下采样索引
selected_idx = 1:decimation_factor:length(index);
if selected_idx(end) ~= length(index)
    selected_idx = [selected_idx, length(index)];
end

% 绘制各曲线，只在指定索引位置显示标记
semilogy(index, BER_ls_smooth, '-o', 'LineWidth', 1.2, 'MarkerFaceColor', [0,0.447,0.741], ...
    'MarkerIndices', selected_idx, 'MarkerSize', 6); hold on;

semilogy(index, BER_lmmse_smooth, '-v', 'LineWidth', 1.2, 'MarkerFaceColor', [0.850,0.325,0.098], ...
    'MarkerIndices', selected_idx, 'MarkerSize', 6);

semilogy(index, BER_lmmse_dft_smooth, '-s', 'LineWidth', 1.2, 'MarkerFaceColor', [0.929,0.694,0.125], ...
    'MarkerIndices', selected_idx, 'MarkerSize', 6);

semilogy(index, BER_lmmse_avg_smooth, '-^', 'LineWidth', 1.2, 'MarkerFaceColor', [0.494,0.184,0.556], ...
    'MarkerIndices', selected_idx, 'MarkerSize', 6);

semilogy(index, BER_lmmse_theo_smooth, '-d', 'LineWidth', 1.2, 'MarkerFaceColor', [0.466,0.674,0.188], ...
    'MarkerIndices', selected_idx, 'MarkerSize', 6);

semilogy(index, BER_ideal_smooth, 'r-', 'LineWidth', 1.2, ...
    'MarkerIndices', selected_idx, 'Marker', 'o', 'MarkerSize', 6,'MarkerFaceColor', [1,0,0]);

hold off;

xlabel('Frame Index','FontName', 'Times New Roman');
ylabel('BER','FontName', 'Times New Roman');
legend('LS','LMMSE-Pilot','LMMSE-DFT','LMMSE-Avg','LMMSE-CKM','Ideal CFR','FontName', 'Times New Roman');  % 更新图例

grid on;
box on;
ylim([ymin*0.95, ymax*1.3]);
xlim([0 num_frames+1]);
% 设置横坐标刻度间隔，例如每隔100显示一个刻度
set(gca, 'XTick', 0:100:800);


MSE_ls_all = 10*log10(MSE_ls_all);
MSE_lmmse_all = 10*log10(MSE_lmmse_all);
MSE_lmmse_avg_all = 10*log10(MSE_lmmse_avg_all);
MSE_lmmse_theo_all = 10*log10(MSE_lmmse_theo_all);
MSE_lmmse_dft_all = 10*log10(MSE_lmmse_dft_all);

% MSE 部分
% =========================
% 三层平滑参数
% =========================
window1 = 150;   % 主平滑
window2 = 60;    % 中等
window3 = 30;    % 抛光

% =========================
% 第一层
% =========================
MSE_ls_s1         = movmean(MSE_ls_all, window1, 2);
MSE_lmmse_s1      = movmean(MSE_lmmse_all, window1, 2);
MSE_lmmse_avg_s1  = movmean(MSE_lmmse_avg_all, window1, 2);
MSE_lmmse_theo_s1 = movmean(MSE_lmmse_theo_all, window1, 2);
MSE_lmmse_dft_s1  = movmean(MSE_lmmse_dft_all, window1, 2);  % 添加 DFT

% =========================
% 第二层
% =========================
MSE_ls_s2         = movmean(MSE_ls_s1, window2, 2);
MSE_lmmse_s2      = movmean(MSE_lmmse_s1, window2, 2);
MSE_lmmse_avg_s2  = movmean(MSE_lmmse_avg_s1, window2, 2);
MSE_lmmse_theo_s2 = movmean(MSE_lmmse_theo_s1, window2, 2);
MSE_lmmse_dft_s2  = movmean(MSE_lmmse_dft_s1, window2, 2);  % 添加 DFT

% =========================
% 第三层
% =========================
MSE_ls_smooth         = movmean(MSE_ls_s2, window3, 2);
MSE_lmmse_smooth      = movmean(MSE_lmmse_s2, window3, 2);
MSE_lmmse_avg_smooth  = movmean(MSE_lmmse_avg_s2, window3, 2);
MSE_lmmse_theo_smooth = movmean(MSE_lmmse_theo_s2, window3, 2);
MSE_lmmse_dft_smooth  = movmean(MSE_lmmse_dft_s2, window3, 2);  % 添加 DFT

MSE_ls_plot         = mean(MSE_ls_smooth, 1);
MSE_lmmse_plot      = mean(MSE_lmmse_smooth, 1);
MSE_lmmse_avg_plot  = mean(MSE_lmmse_avg_smooth, 1);
MSE_lmmse_theo_plot = mean(MSE_lmmse_theo_smooth, 1);
MSE_lmmse_dft_plot  = mean(MSE_lmmse_dft_smooth, 1);  % 添加 DFT

figure;

% 定义下采样间隔（例如每5个点显示一个标记）
decimation_factor = 100;  % 可根据需要调整

% 选择要显示标记的下采样索引
selected_idx = 1:decimation_factor:length(index);
if selected_idx(end) ~= length(index)
    selected_idx = [selected_idx, length(index)];
end

% 绘制第一条曲线，标记只出现在下采样点上
plot(index, MSE_ls_plot, '-o', 'LineWidth', 1.2, 'MarkerFaceColor', [0,0.447,0.741], ...
    'MarkerIndices', selected_idx); hold on;

% 第二条曲线
plot(index, MSE_lmmse_plot, '-v', 'LineWidth', 1.2, 'MarkerFaceColor', [0.850,0.325,0.098], ...
    'MarkerIndices', selected_idx);

% 第三条曲线
plot(index, MSE_lmmse_dft_plot, '-s', 'LineWidth', 1.2, 'MarkerFaceColor', [0.929,0.694,0.125], ...
    'MarkerIndices', selected_idx);

% 第四条曲线
plot(index, MSE_lmmse_avg_plot, '-^', 'LineWidth', 1.2, 'MarkerFaceColor', [0.494,0.184,0.556], ...
    'MarkerIndices', selected_idx);

% 第五条曲线
plot(index, MSE_lmmse_theo_plot, '-d', 'LineWidth', 1.2, 'MarkerFaceColor', [0.466,0.674,0.188], ...
    'MarkerIndices', selected_idx);

hold off;

xlabel('Frame Index','FontName', 'Times New Roman');
ylabel('MSE (dB)','FontName', 'Times New Roman');
legend('LS','LMMSE-Pilot','LMMSE-DFT','LMMSE-Avg','LMMSE-CKM','FontName','Times New Roman');  % 更新图例

grid on;
box on;
all_data = [MSE_lmmse_theo_plot, MSE_lmmse_avg_plot,MSE_lmmse_plot, MSE_ls_plot, MSE_lmmse_dft_plot];  % 添加 DFT
ymin = min(all_data);
ymax = max(all_data);
ylim([ymin*1.1, -6]);
xlim([0 num_frames+1]);
% 设置横坐标刻度间隔，例如每隔100显示一个刻度
set(gca, 'XTick', 0:100:800);
set(gca, 'YTick', -22:2:-8);