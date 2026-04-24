clear; close all; rng(42);

SNR = 4:2:16;                % 仿真信噪比范围（dB）
num_realizations = 200;       % 仿真次数
num_symbol = 5;              % OFDM符号数
num_pilot_inter = 5;         % 导频间隔
modulation_mode = 2;         % 调制方式
num_subcarrier = 512;        % 载波数
delta_f = 40e3;              % 子载波间隔
maxDopp = 0;                 % 最大多普勒频率
pathDelays=[5e-8 3e-7 6e-7]; % 多径时延（s）
pathGains=[-5 -6 -10];       % 多径平均路径增益（dB）
fs = num_subcarrier * delta_f;
assert(fs > 1/min(pathDelays),'采样频率不够,不足以描述当前信道。');

chan = comm.RayleighChannel(PathGainsOutputPort=true, ...
               SampleRate=fs,...
               MaximumDopplerShift=maxDopp, ...
               PathDelays=pathDelays, ...
               AveragePathGains=pathGains);
channelInfo = info(chan);
pathFilters = channelInfo.ChannelFilterCoefficients;
toffset = channelInfo.ChannelFilterDelay; %这个非常重要

num_cp=32;% 循环前缀
CP_length = num_cp / fs;  % CP时间长度
max_delay = max(pathDelays); % 最大多径延迟
assert(num_cp > ceil(max_delay*fs) + toffset,'CP必须大于(最大物理时延 + 通道滤波器群时延)');

num_pilot=ceil(num_symbol/num_pilot_inter)+1;% 导频数
pilot_energy = log2(modulation_mode);% 导频符号能量
num_data=num_symbol+num_pilot;% 总符号数
num_bits=log2(modulation_mode)*num_subcarrier*num_symbol;% 发送的总比特数
B = delta_f*(num_subcarrier+num_cp);% 带宽
T_symbol = 1 / delta_f + CP_length;% OFDM符号时间

fprintf('带宽：%.2fMHz \n', B/1e6);
fprintf('导频开销：%.2f%% \n', (num_pilot-1)/num_symbol*100);
fprintf('采样频率: %.2f MHz\n', fs/1e6);
fprintf('子载波间隔: %.2f kHz\n', delta_f/1e3);

% 判断频选类型
classify_fading(pathDelays, pathGains, delta_f, B);

% 判断时变类型
fprintf('最大多普勒频率: %.2f Hz\n', maxDopp);
if maxDopp > 0
    Tc = 0.242 / maxDopp;
    fprintf('相干时间：%.2f ms\n', Tc*1e3);
    St =  num_pilot_inter * T_symbol;
    fprintf('导频间隔：%.2f ms\n', St*1e3);
    if Tc > St
        fprintf('慢衰落信道\n');
    else
        fprintf('快衰落信道\n');
    end
else 
    fprintf('时不变信道\n');
end

%导频以及数据位置计算
pilot_index=zeros(1,num_pilot);
data_index=zeros(1,num_symbol);
for i=1:num_pilot-1
    pilot_index(1,i)=(i-1)*(num_pilot_inter+1)+1;
end
pilot_index(1,num_pilot)=num_data;
for j=1:num_symbol
    data_index(1,j)=ceil(j/num_pilot_inter)+j;
end

%各种初始化，提前分配空间，提升效率
num_bit_err_ls=zeros(length(SNR),num_realizations);
num_bit_err_ideal=zeros(length(SNR),num_realizations);
num_bit_err_esprit=zeros(length(SNR),num_realizations);
num_bit_err_music=zeros(length(SNR),num_realizations);
num_bit_err_ckm=zeros(length(SNR),num_realizations);
mse_ls=zeros(length(SNR),num_realizations);
mse_esprit=zeros(length(SNR),num_realizations);
mse_music=zeros(length(SNR),num_realizations);
mse_ckm=zeros(length(SNR),num_realizations); 
H_LS=zeros(num_subcarrier,num_data);

Piloted_modulated_symbols=zeros(num_subcarrier,num_data);
pilot_data = randi([0 1], log2(modulation_mode)*num_subcarrier, 1); % 随机导频序列
pilot_symbols=qammod(pilot_data,modulation_mode,'InputType','bit');
Piloted_modulated_symbols(:,pilot_index)=repmat(pilot_symbols,1,num_pilot); % 插入导频
pilot_patt=repmat(pilot_symbols,1,num_pilot); % 发送导频矩阵

for c1=1:length(SNR)
    fprintf("信噪比: %.2fdB\n",SNR(c1));
    for num1=1:num_realizations
        release(chan);
        reset(chan);

        dataTx=randi([0 1],1,num_bits);%产生发送的随机序列
        BitsTx=reshape(dataTx,log2(modulation_mode)*num_subcarrier,num_symbol);%将产生的随机序列转换成01矩阵便于调制

        Modulated_symbols=qammod(BitsTx, modulation_mode,'InputType','bit');%载波调制

        Piloted_modulated_symbols(:,data_index)=Modulated_symbols;%加入导频
        
        Tx_piloted_symbols=ofdmmod(Piloted_modulated_symbols,num_subcarrier,num_cp);%OFDM调制   

        [Rx_piloted_symbols_undemod,path_gains] = chan(Tx_piloted_symbols);%Rayleigh信道
        Rx_piloted_symbols_undemod=awgn(Rx_piloted_symbols_undemod, SNR(c1), 'measured');%高斯信道 
        
        Rx_piloted_symbols=ofdmdemod(Rx_piloted_symbols_undemod,num_subcarrier,num_cp);%OFDM解调
        
        Rx_pilot=Rx_piloted_symbols(:,pilot_index);%导频符号
        Rx_symbols=Rx_piloted_symbols(:,data_index);%数据符号
        
        H_ls=Rx_pilot./pilot_patt;%LS估计

        % 理想信道估计(利用ofdmchannelresponse得到多径信道，并进行toffset补偿)
        H = ofdmChannelResponse(path_gains,pathFilters,num_subcarrier,num_cp,1:num_subcarrier,toffset);
        subcarrier_idx = (1:num_subcarrier).';
        H = H .* exp(-1j*2*pi*subcarrier_idx*toffset/num_subcarrier); 
        H_pilot_ideal=H(:,pilot_index);
        H_data_ideal=H(:,data_index);

        % ESPRIT信道估计
        L = 3; % 已知路径数
        H_esprit = esprit_channel_est(H_ls,delta_f,num_subcarrier,L);
        H_esprit_full = repmat(H_esprit,1,num_data); % 扩展为所有OFDM符号
        H_data_esprit = H_esprit_full(:,data_index);

        % MUSIC信道估计
        H_music = music_channel_est(H_ls, delta_f, num_subcarrier, L);
        H_music_full = repmat(H_music,1,num_data); % 扩展为所有OFDM符号
        H_data_music = H_music_full(:,data_index);

        % CKM辅助PM信道估计
        H_ckm = ckm_channel_est(H_ls, delta_f, num_subcarrier, L, pathDelays, toffset, fs);
        H_ckm_full = repmat(H_ckm,1,num_data); % 扩展为所有OFDM符号
        H_data_ckm = H_ckm_full(:,data_index);
       
        % 插值
        for ii=1:num_subcarrier
            H_LS(ii,:)=interp1(pilot_index,H_ls(ii,1:num_pilot),1:num_data,'pchip', 'extrap');
        end
        H_data_ls=H_LS(:,data_index);        
        
        % 迫零均衡
        Tx_data_esti_ls=Rx_symbols./H_data_ls;
        Tx_data_esti_ideal=Rx_symbols./H_data_ideal;
        Tx_data_esti_esprit = Rx_symbols./H_data_esprit;
        Tx_data_esti_music = Rx_symbols./H_data_music;
        Tx_data_esti_ckm = Rx_symbols./H_data_ckm;

        % LS符号解调
        demod_in_ls=Tx_data_esti_ls(:).';
        demod_out_ls=qamdemod(demod_in_ls,modulation_mode,'OutputType','bit');
        demod_out_ls=reshape(demod_out_ls,1,num_bits);
        
        % ESPRIT符号解调
        demod_in_esprit = Tx_data_esti_esprit(:).';
        demod_out_esprit = qamdemod(demod_in_esprit,modulation_mode,'OutputType','bit');
        demod_out_esprit = reshape(demod_out_esprit,1,num_bits);

        % MUSIC符号解调
        demod_in_music = Tx_data_esti_music(:).';
        demod_out_music = qamdemod(demod_in_music, modulation_mode,'OutputType','bit');
        demod_out_music = reshape(demod_out_music,1,num_bits);

        % CKM符号解调
        demod_in_ckm = Tx_data_esti_ckm(:).';
        demod_out_ckm = qamdemod(demod_in_ckm, modulation_mode,'OutputType','bit');
        demod_out_ckm = reshape(demod_out_ckm,1,num_bits);

        % 理想信道估计符号解调
        demod_in_ideal=Tx_data_esti_ideal(:).';
        demod_out_ideal=qamdemod(demod_in_ideal,modulation_mode,'OutputType','bit');
        demod_out_ideal=reshape(demod_out_ideal,1,num_bits);

        % BER
        num_bit_err_ls(c1,num1) = sum(demod_out_ls ~= dataTx);
        num_bit_err_ideal(c1,num1)=sum(demod_out_ideal ~= dataTx);
        num_bit_err_esprit(c1,num1)=sum(demod_out_esprit ~= dataTx);
        num_bit_err_music(c1,num1) = sum(demod_out_music ~= dataTx);
        num_bit_err_ckm(c1,num1) = sum(demod_out_ckm ~= dataTx);

        % MSE
        H_esprit_pilot = repmat(H_esprit,1,num_pilot);
        H_music_pilot = repmat(H_music,1,num_pilot);
        H_ckm_pilot = repmat(H_ckm,1,num_pilot);
        mse_ls(c1,num1)=mean(abs(H_pilot_ideal-H_ls).^2,'all');
        mse_esprit(c1,num1)=mean(abs(H_pilot_ideal-H_esprit_pilot).^2,'all');
        mse_music(c1,num1) = mean(abs(H_pilot_ideal - H_music_pilot).^2,'all');
        mse_ckm(c1,num1) = mean(abs(H_pilot_ideal - H_ckm_pilot).^2,'all');
    end
end

bit_err_ls_sorted = sort(num_bit_err_ls, 2, 'descend');
bit_err_ls_trimmed = bit_err_ls_sorted(:, trim_num+1:end);
BER_ls_raw = mean(bit_err_ls_trimmed, 2) / num_bits;
% 滑动平均
window_size = 1;  % 滑动窗口大小，可根据需要调整
BER_ls = movmean(BER_ls_raw, window_size);

% MUSIC
bit_err_music_sorted = sort(num_bit_err_music, 2, 'descend');
bit_err_music_trimmed = bit_err_music_sorted(:, trim_num+1:end);
BER_music_raw = mean(bit_err_music_trimmed, 2) / num_bits;
BER_music = movmean(BER_music_raw, window_size);

% ESPRIT
bit_err_esprit_sorted = sort(num_bit_err_esprit, 2, 'descend');
bit_err_esprit_trimmed = bit_err_esprit_sorted(:, trim_num+1:end);
BER_esprit_raw = mean(bit_err_esprit_trimmed, 2) / num_bits;
BER_esprit = movmean(BER_esprit_raw, window_size);

% CKM
bit_err_ckm_sorted = sort(num_bit_err_ckm, 2, 'descend');
bit_err_ckm_trimmed = bit_err_ckm_sorted(:, trim_num+1:end);
BER_ckm_raw = mean(bit_err_ckm_trimmed, 2) / num_bits;
BER_ckm = movmean(BER_ckm_raw, window_size);

% Ideal（也去掉前10%最大值，保持一致性）
bit_err_ideal_sorted = sort(num_bit_err_ideal, 2, 'descend');
bit_err_ideal_trimmed = bit_err_ideal_sorted(:, trim_num+1:end);
BER_ideal_raw = mean(bit_err_ideal_trimmed, 2) / num_bits;
BER_ideal = movmean(BER_ideal_raw, window_size);

% MSE（不去掉最大值，直接计算均值）
MSE_ls = mean(mse_ls, 2);
MSE_music = mean(mse_music, 2);
MSE_esprit = mean(mse_esprit, 2);
MSE_ckm = mean(mse_ckm, 2);

% 转换 MSE 为 dB
MSE_ls_dB = 10*log10(MSE_ls);
MSE_music_dB = 10*log10(MSE_music);
MSE_esprit_dB = 10*log10(MSE_esprit);
MSE_ckm_dB = 10*log10(MSE_ckm);

% 定义颜色（使用 RGB 数值）
colors = struct();
colors.ls = [0, 0.447, 0.741];        % 蓝色
colors.music = [0.850, 0.325, 0.098]; % 橙红色
colors.esprit = [0.929, 0.694, 0.125];% 金黄色
colors.ckm = [0.494, 0.184, 0.556];   % 紫色
colors.ideal = [1, 0, 0];

% =========================
% BER 图
% =========================
figure('Position', [100, 100, 600, 450], 'Color', 'w');
semilogy(SNR, BER_ls, '-o', 'LineWidth', 1.1, 'Color', colors.ls, ...
         'MarkerFaceColor', colors.ls, 'MarkerSize', 6); hold on;
semilogy(SNR, BER_music, '-v', 'LineWidth', 1.1, 'Color', colors.music, ...
         'MarkerFaceColor', colors.music, 'MarkerSize', 6);
semilogy(SNR, BER_esprit, '-s', 'LineWidth', 1.1, 'Color', colors.esprit, ...
         'MarkerFaceColor', colors.esprit, 'MarkerSize', 8);
semilogy(SNR, BER_ckm, '-d', 'LineWidth', 1.1, 'Color', colors.ckm, ...
         'MarkerFaceColor', colors.ckm, 'MarkerSize', 6);
semilogy(SNR, BER_ideal, '-p', 'LineWidth', 1.1, 'Color', colors.ideal, ...
         'MarkerFaceColor', colors.ideal, 'MarkerSize', 6);
hold off;

legend('LS', 'PM-MUSIC', 'PM-ESPRIT', 'PM-CKM', 'Ideal CFR', ...
       'FontName', 'Times New Roman', 'Location', 'best');
xlabel('SNR (dB)', 'FontName', 'Times New Roman', 'FontSize', 12);
ylabel('BER', 'FontName', 'Times New Roman', 'FontSize', 12);
grid on;
box on;

% 设置坐标轴
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
xlim([min(SNR), max(SNR)]);

% 设置 BER 图的 y 轴刻度
all_ber = [BER_ls; BER_music; BER_esprit; BER_ckm; BER_ideal];
min_ber = min(all_ber(all_ber > 0));
max_ber = max(all_ber);
ylim([min_ber * 0.8, max_ber * 1.5]);

% 设置对数刻度
min_power = floor(log10(min_ber));
max_power = ceil(log10(max_ber));
ytick_values = 10.^(min_power:max_power);
set(gca, 'YTickMode', 'auto');

% =========================
% MSE 图（dB）
% =========================
figure('Position', [100, 100, 600, 450], 'Color', 'w');
plot(SNR, MSE_ls_dB, '-o', 'LineWidth', 1.1, 'Color', colors.ls, ...
     'MarkerFaceColor', colors.ls, 'MarkerSize', 6); hold on;
plot(SNR, MSE_music_dB, '-v', 'LineWidth', 1.1, 'Color', colors.music, ...
     'MarkerFaceColor', colors.music, 'MarkerSize', 6);
plot(SNR, MSE_esprit_dB, '-s', 'LineWidth', 1.1, 'Color', colors.esprit, ...
     'MarkerFaceColor', colors.esprit, 'MarkerSize', 8);
plot(SNR, MSE_ckm_dB, '-d', 'LineWidth', 1.1, 'Color', colors.ckm, ...
     'MarkerFaceColor', colors.ckm, 'MarkerSize', 6);
hold off;

legend('LS', 'PM-MUSIC', 'PM-ESPRIT', 'PM-CKM', ...
       'FontName', 'Times New Roman', 'Location', 'best');
xlabel('SNR (dB)', 'FontName', 'Times New Roman');
ylabel('MSE (dB)', 'FontName', 'Times New Roman');
grid on;
box on;

% 设置坐标轴
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
xlim([min(SNR), max(SNR)]);

% 设置 MSE 图的 y 轴范围
all_mse_dB = [MSE_ls_dB; MSE_music_dB; MSE_esprit_dB; MSE_ckm_dB];
min_mse = min(all_mse_dB);
max_mse = max(all_mse_dB);
ylim([-22, -2]);

% 设置线性刻度（dB 用线性坐标）
set(gca, 'YTickMode', 'auto');



function fading_type = classify_fading(pathDelays, pathGains, delta_f, B)
    % CLASSIFY_FADING_RMS 基于 RMS 时延扩展判断 OFDM 信道衰落类型
    % (支持 dB 单位的功率输入)
    %
    % 输入:
    %   pathDelays     : 路径时延向量 (秒), e.g., [0, 5e-8, 2e-7]
    %   pathPowers_dB  : 路径功率向量 (dB), e.g., [0, -3, -10, -20]
    %   T_symbol       : OFDM 有效符号时间 (秒), 不含 CP
    %   B              : OFDM 系统总带宽 (Hz)
    %
    % 输出:
    %   fading_type    : 字符串 ('Flat Fading', 'Partial Frequency-Selective', 'Strong Frequency-Selective')

    % 1. 数据预处理与单位转换
    if isempty(pathDelays) || length(pathDelays) ~= length(pathGains)
        error('pathDelays 和 pathPowers_dB 长度必须一致且非空');
    end
    
    pathGains_lin = 10 .^ (pathGains / 10);
    
    % 过滤掉功率过低的路径 (避免噪声干扰 RMS 计算)
    % 策略：保留比最大功率低 60dB 以内的路径
    max_power_dB = max(pathGains);
    threshold_dB = max_power_dB - 60; 
    threshold_lin = 10 ^ (threshold_dB / 10);
    
    valid_idx = pathGains_lin > threshold_lin;
    
    if sum(valid_idx) == 0
        error('所有路径功率均过低 (低于最大值 60dB)，无法计算有效时延扩展');
    end
    
    delays = pathDelays(valid_idx);
    powers = pathGains_lin(valid_idx);
    
    % 2. 计算 RMS 时延扩展 (σ_τ)
    % 归一化功率 (作为概率质量函数)
    P_norm = powers / sum(powers);
    
    % 一阶矩：平均时延 E[τ]
    mean_tau = sum(P_norm .* delays);
    
    % 二阶矩：均方时延 E[τ^2]
    mean_tau_sq = sum(P_norm .* (delays.^2));
    
    % 方差 Var(τ) = E[τ^2] - (E[τ])^2
    variance_tau = mean_tau_sq - mean_tau^2;
    
    % 防止浮点数误差导致方差为微负数
    if variance_tau < 0
        variance_tau = 0;
    end
    
    sigma_tau = sqrt(variance_tau);
    
    % 处理单径或极短时延情况 (σ_τ ≈ 0)
    if sigma_tau < eps
        sigma_tau = eps; 
    end

    % 3. 关键参数计算
    % 相干带宽 (经验公式：Bc ≈ 1 / (5 * σ_τ)，对应频率相关系数 ~0.5)
    Bc = 1 / (5 * sigma_tau); 
    
    % 4. 判断逻辑
    if Bc < delta_f
        fading_type = 'Strong Frequency-Selective';
        reason_str = '相干带宽 < 子载波间隔 (单个子载波内信道变化剧烈)';
    elseif Bc >= delta_f && Bc < B
        fading_type = 'Partial Frequency-Selective';
        reason_str = '子载波间隔 < 相干带宽 < 总带宽 (典型 OFDM 频选场景)';
    else % Bc >= B
        fading_type = 'Flat Fading';
        reason_str = '相干带宽 > 总带宽 (全带宽内信道响应近似恒定)';
    end

    % 5. 输出详细报告
    fprintf('----------------------------------------------\n');
    fprintf('有效路径数: %d (已过滤 <-60dB 弱径)\n', sum(valid_idx));
    fprintf('RMS 时延扩展 (σ_τ):   %.2f ns\n', sigma_tau * 1e9);
    fprintf('相干带宽 (Bc ≈ 1/5σ): %.2f kHz\n', Bc / 1e3);
    fprintf('子载波间隔 (Δf):      %.2f kHz\n', delta_f / 1e3);
    fprintf('系统总带宽 (B):       %.2f MHz\n', B / 1e6);
    fprintf('%s\n', reason_str);
    fprintf('%s\n', fading_type);
    fprintf('----------------------------------------------\n');
end

function H_music = music_channel_est(H_ls, delta_f, num_subcarrier, L)
    % 输入
    % H_ls: LS估计 (num_subcarrier × num_pilot)
    % L: 已知路径数
    %
    % 输出
    % H_music: 重建的CFR

    % 1. 平均降低噪声
    H_obs = mean(H_ls,2);

    % 2. 构造Hankel矩阵
    M = floor(num_subcarrier/3);
    K = num_subcarrier - M;
    X = zeros(M,K);
    for i = 1:K
        X(:,i) = H_obs(i:i+M-1);
    end

    % 3. 协方差矩阵
    R = (X*X')/K;

    % 4. 特征值分解
    [U,D] = eig(R);
    [~,idx] = sort(diag(D),'descend');
    U = U(:,idx);

    % 5. 信号子空间与噪声子空间
    %Us = U(:,1:L);
    Uw = U(:,L+1:end); % 噪声子空间

    % 6. 构建搜索网格
    tau_grid = linspace(0, max(1/delta_f, 1e-7), 2000); % 可以微调
    P_music = zeros(size(tau_grid));

    % 7. 计算 MUSIC 谱
    k = (0:M-1).';
    for ii = 1:length(tau_grid)
        v = exp(-1j*2*pi*delta_f*k*tau_grid(ii));
        P_music(ii) = 1/(v'*(Uw*Uw')*v);
    end

    % 8. 找到 L 个谱峰对应的时延
    [~, locs] = findpeaks(abs(P_music), 'NPeaks', L, 'SortStr', 'descend');
    tau_est = tau_grid(locs).';

    % 9. 估计路径增益
    k_full = (0:num_subcarrier-1).';
    A = zeros(num_subcarrier,L);
    for l = 1:L
        A(:,l) = exp(-1j*2*pi*delta_f*k_full*tau_est(l));
    end
    alpha = pinv(A) * H_obs;

    % 10. 重建 CFR
    H_music = A * alpha;
end

function H_esprit = esprit_channel_est(H_ls, delta_f, num_subcarrier, L)
    % 输入
    % H_ls: LS估计 (num_subcarrier × num_pilot)
    % L: 路径数
    
    % 输出
    % H_esprit: 重建的CFR
    
    % 使用所有pilot平均降低噪声
    H_obs = mean(H_ls,2);
    
    % 构造Hankel矩阵
    M = floor(num_subcarrier/3);
    K = num_subcarrier - M;
    
    X = zeros(M,K);
    
    for i = 1:K
        X(:,i) = H_obs(i:i+M-1);
    end
    
    % 协方差矩阵
    R = (X*X')/K;
    
    % 特征值分解
    [U,D] = eig(R);
    [~,idx] = sort(diag(D),'descend');
    U = U(:,idx);
    
    % 信号子空间
    Us = U(:,1:L);
    
    % 构造移位子空间
    U1 = Us(1:end-1,:);
    U2 = Us(2:end,:);
    
    % ESPRIT旋转矩阵
    Phi = pinv(U1)*U2;
    
    % 特征值
    lambda = eig(Phi);
    
    % 路径时延估计
    tau_est = -angle(lambda)/(2*pi*delta_f);
    
    % ===== 路径增益估计 =====
    
    k = (0:num_subcarrier-1)';
    
    A = zeros(num_subcarrier,L);
    
    for l = 1:L
        A(:,l) = exp(-1j*2*pi*delta_f*k*tau_est(l));
    end
    
    alpha = pinv(A)*H_obs;
    
    % ===== 重建CFR =====
    H_esprit = zeros(num_subcarrier,1); 
    for l = 1:L
        H_esprit = H_esprit + alpha(l)*exp(-1j*2*pi*delta_f*k*tau_est(l));
    end
end

function H_ckm = ckm_channel_est(H_ls, delta_f, num_subcarrier, L, pathDelays, toffset, fs)
    % ===== 路径增益估计 =====
    H_obs = mean(H_ls,2);
    k = (0:num_subcarrier-1)';
    pathDelays_eff = pathDelays + toffset/fs;
    % ===== 重建CFR =====
    A = zeros(num_subcarrier,L);
    for l = 1:L
        A(:,l) = exp(-1j*2*pi*delta_f*k*pathDelays_eff(l));
    end
    alpha = pinv(A)*H_obs;
    H_ckm = A*alpha;
end