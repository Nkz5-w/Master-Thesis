clear; close all;

SNR = 4:2:16;                % 仿真信噪比范围（dB）
num_realizations = 20;       % 仿真次数
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

% 时间统计初始化
time_ls = zeros(length(SNR), num_realizations);
time_esprit = zeros(length(SNR), num_realizations);
time_music = zeros(length(SNR), num_realizations);
time_ckm = zeros(length(SNR), num_realizations);

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
        
        % LS估计（作为基准）
        tic;
        H_ls=Rx_pilot./pilot_patt;
        time_ls(c1, num1) = toc;

        % 理想信道估计(利用ofdmchannelresponse得到多径信道，并进行toffset补偿)
        H = ofdmChannelResponse(path_gains,pathFilters,num_subcarrier,num_cp,1:num_subcarrier,toffset);
        subcarrier_idx = (1:num_subcarrier).';
        H = H .* exp(-1j*2*pi*subcarrier_idx*toffset/num_subcarrier); 
        H_pilot_ideal=H(:,pilot_index);
        H_data_ideal=H(:,data_index);

        % ESPRIT信道估计（需要时延估计）
        L = 3; % 已知路径数
        tic;
        H_esprit = esprit_channel_est(H_ls,delta_f,num_subcarrier,L);
        time_esprit(c1, num1) = toc;
        H_esprit_full = repmat(H_esprit,1,num_data); % 扩展为所有OFDM符号
        H_data_esprit = H_esprit_full(:,data_index);

        % MUSIC信道估计（需要时延估计）
        tic;
        H_music = music_channel_est(H_ls, delta_f, num_subcarrier, L);
        time_music(c1, num1) = toc;
        H_music_full = repmat(H_music,1,num_data); % 扩展为所有OFDM符号
        H_data_music = H_music_full(:,data_index);

        % CKM辅助PM信道估计（免时延估计，直接使用已知路径时延）
        tic;
        H_ckm = ckm_channel_est(H_ls, delta_f, num_subcarrier, L, pathDelays, toffset, fs);
        time_ckm(c1, num1) = toc;
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

% ========== 时间统计结果 ==========
fprintf('\n========== 计算时间统计结果 ==========\n');
fprintf('方法\t\t平均时间(ms)\t标准差(ms)\t相对加速比\n');
fprintf('------------------------------------------------\n');

% 计算平均时间和标准差
time_ls_mean = mean(time_ls(:)) * 1000;  % 转换为毫秒
time_esprit_mean = mean(time_esprit(:)) * 1000;
time_music_mean = mean(time_music(:)) * 1000;
time_ckm_mean = mean(time_ckm(:)) * 1000;

time_ls_std = std(time_ls(:)) * 1000;
time_esprit_std = std(time_esprit(:)) * 1000;
time_music_std = std(time_music(:)) * 1000;
time_ckm_std = std(time_ckm(:)) * 1000;

% 计算相对加速比（以MUSIC为基准）
speedup_esprit = time_music_mean / time_esprit_mean;
speedup_ckm = time_music_mean / time_ckm_mean;

fprintf('LS\t\t%.4f ± %.4f\t1.00x\n', time_ls_mean, time_ls_std);
fprintf('PM-MUSIC\t%.4f ± %.4f\t1.00x\n', time_music_mean, time_music_std);
fprintf('PM-ESPRIT\t%.4f ± %.4f\t%.2fx\n', time_esprit_mean, time_esprit_std, speedup_esprit);
fprintf('PM-CKM\t\t%.4f ± %.4f\t%.2fx\n', time_ckm_mean, time_ckm_std, speedup_ckm);
fprintf('================================================\n');
fprintf('优势：CKM方法相比MUSIC加速 %.2f 倍，相比ESPRIT加速 %.2f 倍\n', speedup_ckm, speedup_ckm/speedup_esprit);
fprintf('原因：CKM直接使用已知的多径时延信息，免去了复杂的时延估计步骤\n');

% BER 处理（去除前10%最大值）
trim_percent = 0.1;
trim_num = round(num_realizations * trim_percent);

bit_err_ls_sorted = sort(num_bit_err_ls, 2, 'descend');
bit_err_ls_trimmed = bit_err_ls_sorted(:, trim_num+1:end);
BER_ls_raw = mean(bit_err_ls_trimmed, 2) / num_bits;
window_size = 1;
BER_ls = movmean(BER_ls_raw, window_size);

bit_err_music_sorted = sort(num_bit_err_music, 2, 'descend');
bit_err_music_trimmed = bit_err_music_sorted(:, trim_num+1:end);
BER_music_raw = mean(bit_err_music_trimmed, 2) / num_bits;
BER_music = movmean(BER_music_raw, window_size);

bit_err_esprit_sorted = sort(num_bit_err_esprit, 2, 'descend');
bit_err_esprit_trimmed = bit_err_esprit_sorted(:, trim_num+1:end);
BER_esprit_raw = mean(bit_err_esprit_trimmed, 2) / num_bits;
BER_esprit = movmean(BER_esprit_raw, window_size);

bit_err_ckm_sorted = sort(num_bit_err_ckm, 2, 'descend');
bit_err_ckm_trimmed = bit_err_ckm_sorted(:, trim_num+1:end);
BER_ckm_raw = mean(bit_err_ckm_trimmed, 2) / num_bits;
BER_ckm = movmean(BER_ckm_raw, window_size);

bit_err_ideal_sorted = sort(num_bit_err_ideal, 2, 'descend');
bit_err_ideal_trimmed = bit_err_ideal_sorted(:, trim_num+1:end);
BER_ideal_raw = mean(bit_err_ideal_trimmed, 2) / num_bits;
BER_ideal = movmean(BER_ideal_raw, window_size);

% MSE 处理
MSE_ls = mean(mse_ls, 2);
MSE_music = mean(mse_music, 2);
MSE_esprit = mean(mse_esprit, 2);
MSE_ckm = mean(mse_ckm, 2);

MSE_ls_dB = 10*log10(MSE_ls);
MSE_music_dB = 10*log10(MSE_music);
MSE_esprit_dB = 10*log10(MSE_esprit);
MSE_ckm_dB = 10*log10(MSE_ckm);

% 定义颜色
colors = struct();
colors.ls = [0, 0.447, 0.741];
colors.music = [0.850, 0.325, 0.098];
colors.esprit = [0.929, 0.694, 0.125];
colors.ckm = [0.494, 0.184, 0.556];
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
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
xlim([min(SNR), max(SNR)]);

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
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
xlim([min(SNR), max(SNR)]);
ylim([-22, -2]);

% ========== 图1：平均计算时间对比 ==========
figure('Position', [100, 100, 400, 600], 'Color', 'w');
methods = {'LS', 'PM-MUSIC', 'PM-ESPRIT', 'PM-CKM'};
times = [time_ls_mean, time_music_mean, time_esprit_mean, time_ckm_mean];
std_vals = [time_ls_std, time_music_std, time_esprit_std, time_ckm_std];

bar(times, 'FaceColor', [0.5, 0.5, 0.8]);
hold on;
errorbar(1:4, times, std_vals, 'k', 'LineStyle', 'none', 'LineWidth', 1.5);
hold off;
set(gca, 'XTickLabel', methods, 'FontSize', 12);
ylabel('平均计算时间 (ms)', 'FontSize', 12);
grid on;
box on;

% 调整 y 轴范围，让上方留出 20% 空间
ymax = max(times + std_vals);
ylim([0, ymax * 1.2]);

% ========== 图2：相对加速比对比 ==========
figure('Position', [100, 100, 400, 600], 'Color', 'w');
methods_reduced = {'PM-MUSIC', 'PM-ESPRIT', 'PM-CKM'};
speedup_reduced = [1, speedup_esprit, speedup_ckm];

bar(speedup_reduced, 'FaceColor', [0.8, 0.5, 0.5]);
set(gca, 'XTickLabel', methods_reduced, 'FontSize', 12);
xtickangle(30);  % 旋转45度，避免重叠
ylabel('相对加速比 (以MUSIC为基准)', 'FontSize', 12);
grid on;
box on;

% 调整加速比图的 y 轴范围
ymax_speedup = max(speedup_reduced);
ylim([0, ymax_speedup * 1.2]);

% 添加文本标注
text(1, speedup_reduced(1) + ymax_speedup * 0.05, sprintf('%.2f', speedup_reduced(1)), ...
     'HorizontalAlignment', 'center', 'FontSize', 12);
text(2, speedup_reduced(2) + ymax_speedup * 0.05, sprintf('%.2f', speedup_reduced(2)), ...
     'HorizontalAlignment', 'center', 'FontSize', 12);
text(3, speedup_reduced(3) + ymax_speedup * 0.05, sprintf('%.2f', speedup_reduced(3)), ...
     'HorizontalAlignment', 'center', 'FontSize', 12);

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