clear; close all; 
rng(42);

index = 0:1:799;                    % OFDM帧数
SNR = 8;                            % 仿真信噪比（dB）
num_realization = 20;               % 仿真次数
num_symbol = 5;                     % OFDM符号数
num_pilot_inter = 5;                % 导频间隔
modulation_mode = 2;                % 调制方式
num_subcarrier = 512;               % 载波数
num_cp=32;                          % 循环前缀数
delta_f = 40e3;                     % 子载波间隔
v = 300 / 3.6;                      % 移动速度（m/s）
fc = 2 * 1e9;                       % 载波频率（Hz）
c = 3 * 1e8;                        % 光速
maxDopp = 0 * fc / c;               % 最大多普勒频率
pathDelays=[5e-8 3e-7 6e-7];
pathGains=[-5 -6 -10];
fs = num_subcarrier * delta_f;      % 采样频率
assert(fs > 1/min(pathDelays),'采样频率不够,不足以描述当前信道。');

% 信道
chan = comm.RayleighChannel(PathGainsOutputPort=true, ...
               SampleRate=fs,...
               MaximumDopplerShift=maxDopp, ...
               PathDelays=pathDelays, ...
               AveragePathGains=pathGains);
channelInfo = info(chan);
pathFilters = channelInfo.ChannelFilterCoefficients;
toffset = channelInfo.ChannelFilterDelay;

CP_length = num_cp / fs;  % CP时间长度
max_delay = max(pathDelays); % 最大多径延迟
assert(num_cp > ceil(max_delay*fs) + toffset,'CP必须大于(最大物理时延 + 通道滤波器群时延)');

num_pilot=ceil(num_symbol/num_pilot_inter)+1; % 导频数
pilot_energy = log2(modulation_mode); % 导频符号能量
num_data=num_symbol+num_pilot; % 总符号数
num_bits=log2(modulation_mode)*num_subcarrier*num_symbol; % 发送的总比特数
B = delta_f*(num_subcarrier+num_cp);% 带宽
T_symbol = 1 / delta_f + CP_length;% OFDM符号时间

fprintf('导频开销：%.2f%% \n', (num_pilot-1)/(num_symbol+num_pilot-1)*100);
fprintf('采样频率: %.2f MHz\n', fs/1e6);
fprintf('子载波间隔: %.2f kHz\n', delta_f/1e3);
fprintf('符号时间: %.2f us\n', T_symbol*1e6);

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
num_frames = length(index);
num_bit_err_ls=zeros(num_frames,num_realization);
num_bit_err_lmmse=zeros(num_frames,num_realization);
num_bit_err_lmmse_theo=zeros(num_frames,num_realization);
num_bit_err_lmmse_avg=zeros(num_frames,num_realization);
num_bit_err_ideal=zeros(num_frames,num_realization);
H_LS=zeros(num_subcarrier,num_data);
H_LMMSE=zeros(num_subcarrier,num_data);
H_LMMSE_THEO=zeros(num_subcarrier,num_data);
H_LMMSE_AVG=zeros(num_subcarrier,num_data);
Piloted_modulated_symbols=zeros(num_subcarrier,num_data);

pilot_data = randi([0 1], log2(modulation_mode)*num_subcarrier, 1); % 随机导频序列
pilot_symbols=qammod(pilot_data,modulation_mode,'InputType','bit');
Piloted_modulated_symbols(:,pilot_index)=repmat(pilot_symbols,1,num_pilot); % 插入导频
pilot_patt=repmat(pilot_symbols,1,num_pilot); % 发送导频矩阵

Rhh_theo = calcRhh(pathGains, pathDelays, fs, num_subcarrier, toffset);

BER_ls_all          = zeros(num_realization, num_frames);
BER_lmmse_all       = zeros(num_realization, num_frames);
BER_lmmse_dft_all   = zeros(num_realization, num_frames);
BER_lmmse_avg_all   = zeros(num_realization, num_frames);
BER_lmmse_theo_all  = zeros(num_realization, num_frames);
BER_ideal_all       = zeros(num_realization, num_frames);

MSE_ls_all          = zeros(num_realization, num_frames);
MSE_lmmse_all       = zeros(num_realization, num_frames);
MSE_lmmmse_dft_all  = zeros(num_realization, num_frames);
MSE_lmmse_avg_all   = zeros(num_realization, num_frames);
MSE_lmmse_theo_all  = zeros(num_realization, num_frames);
MSE_lmmse_dft_all   = zeros(num_realization, num_frames);

for u = 1:num_realization
    fprintf('仿真 %d / %d\n', u, num_realization);
    Rhh_sum = zeros(num_subcarrier);
    count = 0;
    for f = 1:num_frames
        release(chan);
        reset(chan);
        % --- 生成数据 ---
        dataTx = randi([0 1], 1, num_bits);
        BitsTx = reshape(dataTx, log2(modulation_mode)*num_subcarrier, num_symbol);
        Modulated_symbols = qammod(BitsTx, modulation_mode, 'InputType', 'bit', 'UnitAveragePower', true);
        
        % 构造带导频的帧
        Piloted_modulated_symbols = zeros(num_subcarrier, num_data);
        Piloted_modulated_symbols(:, pilot_index) = pilot_patt;
        Piloted_modulated_symbols(:, data_index) = Modulated_symbols;
        
        % OFDM 调制与传输
        Tx_piloted_symbols = ofdmmod(Piloted_modulated_symbols, num_subcarrier, num_cp);
        [Rx_piloted_symbols_undemod, path_gains] = chan(Tx_piloted_symbols);
        Rx_piloted_symbols_undemod = awgn(Rx_piloted_symbols_undemod, SNR, 'measured');
        Rx_piloted_symbols = ofdmdemod(Rx_piloted_symbols_undemod, num_subcarrier, num_cp);
        
        Rx_pilot = Rx_piloted_symbols(:, pilot_index);
        Rx_symbols = Rx_piloted_symbols(:, data_index);
        
        H_ideal_full = ofdmChannelResponse(path_gains, pathFilters, num_subcarrier, num_cp, 1:num_subcarrier, toffset);
        subcarrier_idx = (1:num_subcarrier).';
        H_ideal_full = H_ideal_full .* exp(-1j*2*pi*subcarrier_idx*toffset/num_subcarrier);
        H_ideal_pilot = H_ideal_full(:, pilot_index);
        H_ideal_data = H_ideal_full(:, data_index);

        % --- LS 估计 ---
        H_ls = Rx_pilot ./ pilot_patt;
        H_ls_smoothed = movmean(H_ls, 4, 2);
        noise_power = mean(abs(Rx_pilot - H_ideal_pilot .* pilot_patt).^2, 'all');
        
        % --- 当前帧 Rhh ---
        Rhh_current = (H_ls_smoothed * H_ls_smoothed') / num_pilot;
        
        % --- 更新历史（冷启动）---
        count = count + 1;
        Rhh_sum = Rhh_sum + Rhh_current;
        Rhh_avg = Rhh_sum / count;
        
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

        H_lmmse = Rhh_current / (Rhh_current + noise_power/pilot_energy * eye(num_subcarrier)) * H_ls;
        H_lmmse_avg = Rhh_avg / (Rhh_avg + noise_power/pilot_energy * eye(num_subcarrier)) * H_ls;
        H_lmmse_theo = Rhh_theo / (Rhh_theo + noise_power/pilot_energy * eye(num_subcarrier)) * H_ls;
        H_lmmse_dft = Rhh_structured / (Rhh_structured + noise_power/pilot_energy * eye(num_subcarrier)) * H_ls;

        H_LS_full = zeros(num_subcarrier, num_data);
        H_LMMSE_full = zeros(num_subcarrier, num_data);
        H_LMMSE_AVG_full = zeros(num_subcarrier, num_data);
        H_LMMSE_THEO_full = zeros(num_subcarrier, num_data);
        H_LMMSE_DFT_full = zeros(num_subcarrier, num_data);
        
        for ii = 1:num_subcarrier
            H_LS_full(ii, :) = interp1(pilot_index, H_ls(ii, :), 1:num_data, 'pchip', 'extrap');
            H_LMMSE_full(ii, :) = interp1(pilot_index, H_lmmse(ii, :), 1:num_data, 'pchip', 'extrap');
            H_LMMSE_AVG_full(ii, :) = interp1(pilot_index, H_lmmse_avg(ii, :), 1:num_data, 'pchip', 'extrap');
            H_LMMSE_THEO_full(ii, :) = interp1(pilot_index, H_lmmse_theo(ii, :), 1:num_data, 'pchip', 'extrap');
            H_LMMSE_DFT_full(ii, :) = interp1(pilot_index, H_lmmse_dft(ii, :), 1:num_data, 'pchip', 'extrap');
        end
        
        H_data_ls = H_LS_full(:, data_index);
        H_data_lmmse = H_LMMSE_full(:, data_index);
        H_data_lmmse_avg = H_LMMSE_AVG_full(:, data_index);
        H_data_lmmse_theo = H_LMMSE_THEO_full(:, data_index);
        H_data_lmmse_dft = H_LMMSE_DFT_full(:, data_index);
        
        % --- 迫零均衡与解调 ---
        est_ls = Rx_symbols ./ H_data_ls;
        est_lmmse = Rx_symbols ./ H_data_lmmse;
        est_lmmse_avg = Rx_symbols ./ H_data_lmmse_avg;
        est_lmmse_theo = Rx_symbols ./ H_data_lmmse_theo;
        est_lmmse_dft = Rx_symbols ./ H_data_lmmse_dft;
        est_ideal = Rx_symbols ./ H_ideal_data;
        
        demod_ls = qamdemod(est_ls(:).', modulation_mode, 'OutputType', 'bit');
        demod_lmmse = qamdemod(est_lmmse(:).', modulation_mode, 'OutputType', 'bit');
        demod_lmmse_avg = qamdemod(est_lmmse_avg(:).', modulation_mode, 'OutputType', 'bit');
        demod_lmmse_theo = qamdemod(est_lmmse_theo(:).', modulation_mode, 'OutputType', 'bit');
        demod_lmmse_dft = qamdemod(est_lmmse_dft(:).', modulation_mode, 'OutputType', 'bit');
        demod_ideal = qamdemod(est_ideal(:).', modulation_mode, 'OutputType', 'bit');
        
        demod_ls = reshape(demod_ls, 1, num_bits);
        demod_lmmse = reshape(demod_lmmse, 1, num_bits);
        demod_lmmse_avg = reshape(demod_lmmse_avg, 1, num_bits);
        demod_lmmse_theo = reshape(demod_lmmse_theo, 1, num_bits);
        demod_lmmse_dft = reshape(demod_lmmse_dft, 1, num_bits);
        demod_ideal = reshape(demod_ideal, 1, num_bits);
        
        BER_ls_all(u, f) = sum(demod_ls ~= dataTx) / num_bits;
        BER_lmmse_all(u, f) = sum(demod_lmmse ~= dataTx) / num_bits;
        BER_lmmse_avg_all(u, f) = sum(demod_lmmse_avg ~= dataTx) / num_bits;
        BER_lmmse_theo_all(u, f) = sum(demod_lmmse_theo ~= dataTx) / num_bits;
        BER_lmmse_dft_all(u, f) = sum(demod_lmmse_dft ~= dataTx) / num_bits;
        BER_ideal_all(u, f) = sum(demod_ideal ~= dataTx) / num_bits;

        % 注意：MSE 在 data_index 对应的子载波上计算（与 BER 一致）
        num_data_sc = size(H_data_ls, 1) * size(H_data_ls, 2);  % 总数据子载波数
        
        MSE_ls_all(u, f)          = norm(H_data_ls - H_ideal_data, 'fro')^2 / num_data_sc;
        MSE_lmmse_all(u, f)       = norm(H_data_lmmse - H_ideal_data, 'fro')^2 / num_data_sc;
        MSE_lmmse_avg_all(u, f)   = norm(H_data_lmmse_avg - H_ideal_data, 'fro')^2 / num_data_sc;
        MSE_lmmse_theo_all(u, f)  = norm(H_data_lmmse_theo - H_ideal_data, 'fro')^2 / num_data_sc;
        MSE_lmmse_dft_all(u, f)  = norm(H_data_lmmse_dft - H_ideal_data, 'fro')^2 / num_data_sc;
    end
end

% ==================== 剪裁异常值并计算平均 BER ====================
percent_trim = 0.1;
num_trim = round(num_realization * percent_trim);

BER_ls_trimmed = zeros(1, num_frames);
BER_lmmse_trimmed = zeros(1, num_frames);
BER_lmmse_avg_trimmed = zeros(1, num_frames);
BER_lmmse_theo_trimmed = zeros(1, num_frames);
BER_lmmse_dft_trimmed = zeros(1, num_frames);
BER_ideal_trimmed = zeros(1, num_frames);

for f = 1:num_frames
    % LS
    sorted_val = sort(BER_ls_all(:, f));
    BER_ls_trimmed(f) = mean(sorted_val(1:end-num_trim));
    
    % LMMSE
    sorted_val = sort(BER_lmmse_all(:, f));
    BER_lmmse_trimmed(f) = mean(sorted_val(1:end-num_trim));

    % LMMSE
    sorted_val = sort(BER_lmmse_dft_all(:, f));
    BER_lmmse_dft_trimmed(f) = mean(sorted_val(1:end-num_trim));
    
    % LMMSE-avg
    sorted_val = sort(BER_lmmse_avg_all(:, f));
    BER_lmmse_avg_trimmed(f) = mean(sorted_val(1:end-num_trim));
    
    % LMMSE-CKM
    sorted_val = sort(BER_lmmse_theo_all(:, f));
    BER_lmmse_theo_trimmed(f) = mean(sorted_val(1:end-num_trim));
    
    % Ideal
    sorted_val = sort(BER_ideal_all(:, f));
    BER_ideal_trimmed(f) = mean(sorted_val(1:end-num_trim));
end

% 保存 BER 结果
save('BER_ls_trimmed.mat', 'BER_ls_trimmed');
save('BER_lmmse_trimmed.mat', 'BER_lmmse_trimmed');
save('BER_lmmse_avg_trimmed.mat', 'BER_lmmse_avg_trimmed');
save('BER_lmmse_theo_trimmed.mat', 'BER_lmmse_theo_trimmed');
save('BER_ideal_trimmed.mat', 'BER_ideal_trimmed');
save('BER_lmmse_dft_trimmed.mat', 'BER_lmmse_dft_trimmed');

% 保存 MSE 结果
save('MSE_ls_all.mat', 'MSE_ls_all');
save('MSE_lmmse_all.mat', 'MSE_lmmse_all');
save('MSE_lmmse_avg_all.mat', 'MSE_lmmse_avg_all');
save('MSE_lmmse_theo_all.mat', 'MSE_lmmse_theo_all');
save('MSE_lmmse_dft_all.mat', 'MSE_lmmse_dft_all');

function Rhh = calcRhh(pathGains, pathDelays, fs, num_subcarrier, toffset)
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
