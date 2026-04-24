%修复采样频率和子载波间隔（与信道最小时延有关，如果采样频率低于最小时延的倒数，则ofdmchannelresponse失效）
%进一步修复采样频率和子载波间隔，子载波间隔不能过大也不可以过小
%提炼Rhh_avg，并且修改maxDopp设定
clear; close all; rng(42);

SNR = 4:2:16;           % 仿真信噪比范围（dB）
num_realizations = 200; % 仿真次数
num_symbol = 5;         % OFDM符号数
num_pilot_inter = 5;    % 导频间隔
modulation_mode = 2;    % 调制方式
num_subcarrier = 512;   % 载波数
delta_f = 40e3;         % 子载波间隔
maxDopp = 0;            % 最大多普勒频率
pathDelays=[5e-8 3e-7 6e-7];
pathGains=[-5 -6 -10];

fs = num_subcarrier * delta_f;
assert(fs > 1/min(pathDelays),'采样频率不够,不足以描述当前信道。');
chan = comm.RayleighChannel(PathGainsOutputPort=true, ...
               SampleRate=fs,...
               MaximumDopplerShift=maxDopp, ...
               PathDelays=pathDelays, ...
               AveragePathGains=pathGains);
channelInfo = info(chan);
pathFilters = channelInfo.ChannelFilterCoefficients;
toffset = channelInfo.ChannelFilterDelay;

num_cp=32;% 循环前缀
CP_length = num_cp / fs;  % CP时间长度
max_delay = max(pathDelays); % 最大多径延迟
assert(num_cp > ceil(max_delay*fs) + toffset,'CP必须大于(最大物理时延 + 通道滤波器群时延)');

num_pilot=ceil(num_symbol/num_pilot_inter)+1;% 导频数
pilot_energy = log2(modulation_mode); % 导频符号能量
num_data=num_symbol+num_pilot;% 总符号数
num_bits=log2(modulation_mode)*num_subcarrier*num_symbol;% 发送的总比特数
B = delta_f*(num_subcarrier+num_cp);% 带宽
T_symbol = 1 / delta_f + CP_length;% OFDM符号时间

fprintf('带宽：%.2fMHz \n', B/1e6);
fprintf('导频开销：%.2f%% \n', (num_pilot-1)/num_symbol*100);
fprintf('采样频率: %.2f MHz\n', fs/1e6);
fprintf('子载波间隔: %.2f kHz\n', delta_f/1e3);

classify_fading(pathDelays, pathGains, delta_f, B);

% 判断信道类型
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
num_bit_err_lmmse=zeros(length(SNR),num_realizations);
num_bit_err_lmmse_ckm=zeros(length(SNR),num_realizations);
num_bit_err_lmmse_avg=zeros(length(SNR),num_realizations);
num_bit_err_ideal=zeros(length(SNR),num_realizations);
num_bit_err_lmmse_pilot=zeros(length(SNR),num_realizations);

mse_ls=zeros(length(SNR),num_realizations);
mse_lmmse=zeros(length(SNR),num_realizations);
mse_lmmse_ckm=zeros(length(SNR),num_realizations);
mse_lmmse_avg=zeros(length(SNR),num_realizations);
mse_lmmse_pilot=zeros(length(SNR),num_realizations);
H_LS=zeros(num_subcarrier,num_data);
H_LMMSE=zeros(num_subcarrier,num_data);
H_LMMSE_CKM=zeros(num_subcarrier,num_data);
H_LMMSE_AVG=zeros(num_subcarrier,num_data);
H_LMMSE_PILOT=zeros(num_subcarrier,num_data);

Piloted_modulated_symbols=zeros(num_subcarrier,num_data);
pilot_data = randi([0 1], log2(modulation_mode)*num_subcarrier, 1); % 随机导频序列
pilot_symbols=qammod(pilot_data,modulation_mode,'InputType','bit');
Piloted_modulated_symbols(:,pilot_index)=repmat(pilot_symbols,1,num_pilot); % 插入导频
pilot_patt=repmat(pilot_symbols,1,num_pilot); % 发送导频矩阵

%CKM先验
Rhh_theo = calcRhh(pathGains, pathDelays, fs, num_subcarrier, toffset);
Rhh_avg = load_Rhh_avg();

for c1=1:length(SNR)
    fprintf("信噪比: %.2fdB\n",SNR(c1));
    for num1=1:num_realizations
        release(chan);
        reset(chan);

        dataTx=randi([0 1],1,num_bits);%产生发送的随机序列
        BitsTx=reshape(dataTx,log2(modulation_mode)*num_subcarrier,num_symbol);%将产生的随机序列转换成01矩阵便于调制

        Modulated_symbols=qammod(BitsTx, 2,'InputType','bit');%载波调制

        Piloted_modulated_symbols(:,data_index)=Modulated_symbols;%加入导频
        
        Tx_piloted_symbols=ofdmmod(Piloted_modulated_symbols,num_subcarrier,num_cp);%OFDM调制   

        [Rx_piloted_symbols_undemod,path_gains] = chan(Tx_piloted_symbols);%Rayleigh信道
        Rx_piloted_symbols_undemod=awgn(Rx_piloted_symbols_undemod, SNR(c1), 'measured');%高斯信道 
        
        Rx_piloted_symbols=ofdmdemod(Rx_piloted_symbols_undemod,num_subcarrier,num_cp);%OFDM解调
        
        Rx_pilot=Rx_piloted_symbols(:,pilot_index);%导频符号
        Rx_symbols=Rx_piloted_symbols(:,data_index);%数据符号
        
        % 理想信道估计(利用ofdmchannelresponse得到多径信道，并进行toffset补偿)
        H = ofdmChannelResponse(path_gains,pathFilters,num_subcarrier,num_cp,1:num_subcarrier,toffset);
        subcarrier_idx = (1:num_subcarrier).';
        H = H .* exp(-1j*2*pi*subcarrier_idx*toffset/num_subcarrier); 
        H_pilot_ideal=H(:,pilot_index);
        H_data_ideal=H(:,data_index);

        H_ls=Rx_pilot./pilot_patt;%LS估计

        H_ls_smoothed=movmean(H_ls,4,2);%对LS估计滑动平均降噪
        noise_power=mean(abs(Rx_pilot-H_ls_smoothed.*pilot_patt).^2,'all');%噪声功率估计,不能直接用SNR因为SNR是未知的

        %LMMSE信道估计（利用DFT计算Rhh)
        cir_all = zeros(num_subcarrier, num_pilot);

        for sym_idx = 1:num_pilot
            % IFFT变换到时域
            cir_tmp = ifft(H_ls(:, sym_idx));
            % 计算功率
            cir_all(:, sym_idx) = abs(cir_tmp).^2;
        end
        
        % 平均多个导频符号的PDP，提高估计精度
        pdp_avg = mean(cir_all, 2);
        pdp_avg_db = 10*log10(pdp_avg + eps);
        
        % 2. 径检测：设置噪声门限
        % 估计噪声基底（取尾部平均值）
        noise_floor_idx = round(0.8*num_subcarrier):num_subcarrier;
        noise_floor = mean(pdp_avg(noise_floor_idx));
        
        % 设置门限（通常为噪声基底以上10-15dB）
        threshold_db = 10*log10(noise_floor) + 18;
        threshold_linear = 10^(threshold_db/10);
        
        % 检测有效径
        valid_paths = find(pdp_avg > threshold_linear);
        if isempty(valid_paths)
            % 如果没有检测到有效径，取最大峰值位置
            [~, max_idx] = max(pdp_avg);
            valid_paths = max_idx;
        end
        
        % 3. 提取每条径的时延和功率
        % 时延向量（秒）
        t_delay = (0:num_subcarrier-1) / fs;
        
        % 提取有效径的参数
        path_delays_est = t_delay(valid_paths);
        path_gains_est = sqrt(pdp_avg(valid_paths));  % 幅度
        
        % 4. 利用提取的参数重构Rhh矩阵
        Rhh_structured = zeros(num_subcarrier, num_subcarrier);
        for m = 1:num_subcarrier
            for n = 1:num_subcarrier
                % 子载波频率差
                freq_diff = (m - n) * delta_f;
                
                % 修正：正确计算多径叠加，得到标量
                correlation = 0;
                for path_idx = 1:length(valid_paths)
                    gain_power = path_gains_est(path_idx)^2;  % 功率
                    delay = path_delays_est(path_idx);
                    correlation = correlation + gain_power * exp(-1j*2*pi*freq_diff*delay);
                end
                
                Rhh_structured(m, n) = correlation;
            end
        end
     
        Rhh_structured = (Rhh_structured + Rhh_structured') / 2;%确保矩阵是Hermitian的
        Rhh=H_ls_smoothed*H_ls_smoothed'/num_pilot;%仅利用导频计算Rhh
        
        H_lmmse_pilot = Rhh / (Rhh+noise_power/pilot_energy*eye(num_subcarrier))*H_ls;
        H_lmmse = Rhh_structured / (Rhh_structured + noise_power / pilot_energy * eye(num_subcarrier)) * H_ls;
        H_lmmse_avg= Rhh_avg / (Rhh_avg + noise_power/pilot_energy*eye(num_subcarrier))*H_ls; % LMMSE信道估计（使用历史平均Rhh）
        H_lmmse_ckm = Rhh_theo / (Rhh_theo + noise_power/pilot_energy * eye(num_subcarrier)) * H_ls;
       
        % 插值
        for ii=1:num_subcarrier
            H_LS(ii,:)=interp1(pilot_index,H_ls(ii,1:num_pilot),1:num_data,'pchip', 'extrap');
            H_LMMSE(ii,:)=interp1(pilot_index,H_lmmse(ii,1:(num_pilot)),1:num_data,'pchip', 'extrap');
            H_LMMSE_AVG(ii,:)=interp1(pilot_index,H_lmmse_avg(ii,1:(num_pilot)),1:num_data,'pchip', 'extrap');
            H_LMMSE_CKM(ii,:)=interp1(pilot_index,H_lmmse_ckm(ii,1:(num_pilot)),1:num_data,'pchip', 'extrap');
            H_LMMSE_PILOT(ii,:)=interp1(pilot_index,H_lmmse_pilot(ii,1:(num_pilot)),1:num_data,'pchip', 'extrap');
        end
        H_data_ls=H_LS(:,data_index);        
        H_data_lmmse=H_LMMSE(:,data_index);
        H_data_lmmse_avg=H_LMMSE_AVG(:,data_index);
        H_data_lmmse_ckm=H_LMMSE_CKM(:,data_index);
        H_data_lmmse_pilot=H_LMMSE_PILOT(:,data_index);
        
        % 迫零均衡
        Tx_data_esti_ls=Rx_symbols./H_data_ls;
        Tx_data_esti_lmmse=Rx_symbols./H_data_lmmse;
        Tx_data_esti_lmmse_avg=Rx_symbols./H_data_lmmse_avg;
        Tx_data_esti_ideal=Rx_symbols./H_data_ideal;
        Tx_data_esti_lmmse_ckm=Rx_symbols./H_data_lmmse_ckm;
        Tx_data_esti_lmmse_pilot=Rx_symbols./H_data_lmmse_pilot;

        % LS符号解调
        demod_in_ls=Tx_data_esti_ls(:).';
        demod_out_ls=qamdemod(demod_in_ls,modulation_mode,'OutputType','bit');
        demod_out_ls=reshape(demod_out_ls,1,num_bits);

        % LMMSE符号解调1
        demod_in_lmmse=Tx_data_esti_lmmse(:).';
        demod_out_lmmse=qamdemod(demod_in_lmmse,modulation_mode,'OutputType','bit');
        demod_out_lmmse=reshape(demod_out_lmmse,1,num_bits);

        % LMMSE符号解调2
        demod_in_lmmse_avg=Tx_data_esti_lmmse_avg(:).';
        demod_out_lmmse_avg=qamdemod(demod_in_lmmse_avg,modulation_mode,'OutputType','bit');
        demod_out_lmmse_avg=reshape(demod_out_lmmse_avg,1,num_bits);

        % LMMSE符号解调3
        demod_in_lmmse_ckm=Tx_data_esti_lmmse_ckm(:).';
        demod_out_lmmse_ckm=qamdemod(demod_in_lmmse_ckm,modulation_mode,'OutputType','bit');
        demod_out_lmmse_ckm=reshape(demod_out_lmmse_ckm,1,num_bits);

        % LMMSE符号解调4
        demod_in_lmmse_pilot=Tx_data_esti_lmmse_pilot(:).';
        demod_out_lmmse_pilot=qamdemod(demod_in_lmmse_pilot,modulation_mode,'OutputType','bit');
        demod_out_lmmse_pilot=reshape(demod_out_lmmse_pilot,1,num_bits);

        % 理想信道符号解调
        demod_in_ideal=Tx_data_esti_ideal(:).';
        demod_out_ideal=qamdemod(demod_in_ideal,modulation_mode,'OutputType','bit');
        demod_out_ideal=reshape(demod_out_ideal,1,num_bits);

        % BER
        num_bit_err_ls(c1,num1) = sum(demod_out_ls ~= dataTx);
        num_bit_err_lmmse(c1,num1) = sum(demod_out_lmmse ~= dataTx);
        num_bit_err_lmmse_avg(c1,num1) = sum(demod_out_lmmse_avg ~= dataTx);
        num_bit_err_ideal(c1,num1)=sum(demod_out_ideal ~= dataTx);
        num_bit_err_lmmse_ckm(c1,num1)=sum(demod_out_lmmse_ckm ~= dataTx);
        num_bit_err_lmmse_pilot(c1,num1)=sum(demod_out_lmmse_pilot ~= dataTx);

        % MSE
        mse_ls(c1,num1)=mean(abs(H_pilot_ideal-H_ls).^2,'all');
        mse_lmmse(c1,num1)=mean(abs(H_pilot_ideal-H_lmmse).^2,'all');
        mse_lmmse_avg(c1,num1)=mean(abs(H_pilot_ideal-H_lmmse_avg).^2,'all');
        mse_lmmse_ckm(c1,num1)=mean(abs(H_pilot_ideal-H_lmmse_ckm).^2,'all');
        mse_lmmse_pilot(c1,num1)=mean(abs(H_pilot_ideal-H_lmmse_pilot).^2,'all');
    end
end

% 可视化
MSE_ls=mean(mse_ls,2);
MSE_lmmse=mean(mse_lmmse,2);
MSE_lmmse_avg=mean(mse_lmmse_avg,2);
MSE_lmmse_ckm=mean(mse_lmmse_ckm,2);
MSE_lmmse_pilot=mean(mse_lmmse_pilot,2);

% 将 MSE 转换为 dB
MSE_ls_dB = 10*log10(MSE_ls);
MSE_lmmse_dB = 10*log10(MSE_lmmse);
MSE_lmmse_avg_dB = 10*log10(MSE_lmmse_avg);
MSE_lmmse_ckm_dB = 10*log10(MSE_lmmse_ckm);
MSE_lmmse_pilot_dB = 10*log10(MSE_lmmse_pilot);

% 设置统一的图形大小和字体
figure('Position', [100, 100, 600, 450], 'Color', 'w');
plot(SNR, MSE_ls_dB, '-o', 'LineWidth', 1.2,'MarkerFaceColor', [0,0.447,0.741]); hold on;
plot(SNR, MSE_lmmse_pilot_dB, '-v', 'LineWidth', 1.2,'MarkerFaceColor', [0.850,0.325,0.098]);
plot(SNR, MSE_lmmse_dB, '-s', 'LineWidth', 1.2,'MarkerFaceColor', [0.929,0.694,0.125]);
plot(SNR, MSE_lmmse_avg_dB, '-^', 'LineWidth', 1.2,'MarkerFaceColor', [0.494,0.184,0.556]);
plot(SNR, MSE_lmmse_ckm_dB, '-d', 'LineWidth', 1.2,'MarkerFaceColor', [0.466,0.674,0.188]);
legend('LS','LMMSE-Pilot','LMMSE-DFT', 'LMMSE-Avg', 'LMMSE-CKM');
xlabel('SNR(dB)');
ylabel('MSE (dB)');  % 修改 ylabel，明确表示单位为 dB
grid on;
set(gca, 'YScale', 'linear', 'FontSize', 12, 'FontName', 'Times New Roman');  % YScale 改为 linear

% ----------------- 去除最大 BER 百分比计算平均值 -----------------
percent_trim = 0.2; 
fprintf('剪裁比例 %.2f\n', percent_trim);
num_trim = round(num_realizations * percent_trim);

% 预分配
BER_ls_trimmed = zeros(length(SNR),1);
BER_lmmse_pilot_trimmed = zeros(length(SNR),1);
BER_lmmse_trimmed = zeros(length(SNR),1);
BER_lmmse_avg_trimmed = zeros(length(SNR),1);
BER_lmmse_ckm_trimmed = zeros(length(SNR),1);
BER_ideal_trimmed = zeros(length(SNR),1);

for i = 1:length(SNR)
    % LS
    sorted_ls = sort(num_bit_err_ls(i,:));
    trimmed_ls = sorted_ls(1:end-num_trim);
    BER_ls_trimmed(i) = mean(trimmed_ls)/num_bits;

    % LMMSE
    sorted_lmmse_pilot = sort(num_bit_err_lmmse_pilot(i,:));
    trimmed_lmmse_pilot = sorted_lmmse_pilot(1:end-num_trim);
    BER_lmmse_pilot_trimmed(i) = mean(trimmed_lmmse_pilot)/num_bits;
    
    % LMMSE
    sorted_lmmse = sort(num_bit_err_lmmse(i,:));
    trimmed_lmmse = sorted_lmmse(1:end-num_trim);
    BER_lmmse_trimmed(i) = mean(trimmed_lmmse)/num_bits;

    % LMMSE avg
    sorted_lmmse_avg = sort(num_bit_err_lmmse_avg(i,:));
    trimmed_lmmse_avg = sorted_lmmse_avg(1:end-num_trim);
    BER_lmmse_avg_trimmed(i) = mean(trimmed_lmmse_avg)/num_bits;

    % LMMSE theo
    sorted_lmmse_ckm = sort(num_bit_err_lmmse_ckm(i,:));
    trimmed_lmmse_ckm = sorted_lmmse_ckm(1:end-num_trim);
    BER_lmmse_ckm_trimmed(i) = mean(trimmed_lmmse_ckm)/num_bits;

    % Ideal
    sorted_ideal = sort(num_bit_err_ideal(i,:));
    trimmed_ideal = sorted_ideal(1:end-num_trim);
    BER_ideal_trimmed(i) = mean(trimmed_ideal)/num_bits;
end

% ==================== 第二部分：绘制BER性能曲线 ====================
% 平滑参数
smooth_ratio = 0.3;  % 平滑窗口比例 (10% 数据点)

% 对每条曲线平滑
BER_ls_smooth           = smooth(BER_ls_trimmed, smooth_ratio, 'loess');
BER_lmmse_pilot_smooth  = smooth(BER_lmmse_pilot_trimmed, smooth_ratio, 'loess');
BER_lmmse_smooth        = smooth(BER_lmmse_trimmed, smooth_ratio, 'loess');
BER_lmmse_avg_smooth    = smooth(BER_lmmse_avg_trimmed, smooth_ratio, 'loess');
BER_lmmse_ckm_smooth    = smooth(BER_lmmse_ckm_trimmed, smooth_ratio, 'loess');
BER_ideal_smooth        = smooth(BER_ideal_trimmed, smooth_ratio, 'loess');

% 绘图
figure('Position', [100, 100, 600, 450], 'Color', 'w');
plot(SNR, BER_ls_smooth, '-o', 'LineWidth', 1.2,  ...
     'MarkerFaceColor', [0,0.447,0.741]); hold on;
plot(SNR, BER_lmmse_pilot_smooth, '-v', 'LineWidth', 1.2, ...
     'MarkerFaceColor', [0.85,0.325,0.098]);
plot(SNR, BER_lmmse_smooth, '-s', 'LineWidth', 1.2, ...
     'MarkerFaceColor', [0.929,0.694,0.125]);
plot(SNR, BER_lmmse_avg_smooth, '-^', 'LineWidth', 1.2, ...
     'MarkerFaceColor', [0.494,0.184,0.556]);
plot(SNR, BER_lmmse_ckm_smooth, '-d', 'LineWidth', 1.2, ...
     'MarkerFaceColor', [0.466,0.674,0.188]);
plot(SNR, BER_ideal_smooth, '-rp', 'LineWidth', 1.2, ...
     'MarkerFaceColor', [1,0,0]);

% 图例和标签
legend('LS', 'LMMSE-Pilot', 'LMMSE-DFT', 'LMMSE-Avg', 'LMMSE-CKM', 'Ideal CFR');
xlabel('SNR(dB)');
ylabel('BER');

% 自动调整y轴范围，让数据充满窗口
allBER = [BER_ls_smooth; BER_lmmse_pilot_smooth; BER_lmmse_smooth; ...
          BER_lmmse_avg_smooth; BER_lmmse_ckm_smooth; BER_ideal_smooth];
ymin = min(allBER);  % 下边界取10的最小次幂
ymax = max(allBER);   % 上边界取10的最大次幂
ylim([ymin*0.8 0.1]);

% 对数坐标
set(gca, 'YScale', 'log', 'FontSize', 12, 'FontName', 'Times New Roman');
set(gca, 'YTickMode', 'auto');
grid on;

function fading_type = classify_fading(pathDelays, pathGains, delta_f, B)
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

function Rhh = calcRhh(pathGains, pathDelays, fs, num_subcarrier, toffset)
    % 综合考虑时延扩展和多普勒扩展
    sigma2 = 10.^(pathGains/10);% 路径功率 (线性)
    Rhh_freq = zeros(num_subcarrier, num_subcarrier);  % 频率相关
    pathDelays_samples = pathDelays * fs;% 转换为采样点
    k = (0:num_subcarrier-1).';% 子载波索引
    for p = 1:length(sigma2)
        delay = pathDelays_samples(p) + toffset; % 加上群延迟
        Rhh_freq = Rhh_freq + sigma2(p) * exp(-1j*2*pi*(k - k.') * delay / num_subcarrier);
    end
    Rhh = Rhh_freq;
end
