clear; close all; rng(42);

num_samples = 50000;        % 样本数
num_subcarrier = 512;       % 子载波数
num_cp = 32;                % 循环前缀
modulation_mode = 2;        % QPSK
fs = num_subcarrier*40e3;   % 采样频率
numPaths = 3;               % 路径数目

X_data = zeros(num_samples,num_subcarrier,2);  % 实部+虚部
Y_data = zeros(num_samples,3);                 % 只保存3条路径延迟（已修正群时延）

% ----------- 导频符号（固定发送端） ----------
pilot_bits = randi([0 1], log2(modulation_mode)*num_subcarrier, 1);
pilot_symbols = qammod(pilot_bits, modulation_mode, 'InputType', 'bit'); 

for n = 1:num_samples 
    pathDelays = zeros(1,numPaths);
    pathDelays(1) = (0 + (10-0)*rand) * 1e-8;
    pathDelays(2) = (1 + (3-1)*rand) * 1e-7;
    pathDelays(3) = (3 + (6-3)*rand) * 1e-7;
    pathDelays = sort(pathDelays, 'ascend');

    pathGains = zeros(1,numPaths);
    pathGains(1) = -5  + (-0  - (-5))*rand;
    pathGains(2) = -10 + (-5  - (-10))*rand;
    pathGains(3) = -20 + (-10 - (-20))*rand;

    chan = comm.RayleighChannel( ...
        PathGainsOutputPort = true, ...
        SampleRate = fs, ...
        MaximumDopplerShift = 0, ...
        PathDelays = pathDelays, ...
        AveragePathGains = pathGains ...
    );
    release(chan);
    reset(chan);
    channelInfo = info(chan);
    pathFilters = channelInfo.ChannelFilterCoefficients;
    toffset = channelInfo.ChannelFilterDelay;  

    % OFDM 调制
    tx_ofdm = ofdmmod(pilot_symbols, num_subcarrier, num_cp);

    % 通过信道
    [rx_chan, ~] = chan(tx_ofdm);

    % 添加噪声
    SNR = 10 + (15-10)*rand;
    rx_awgn = awgn(rx_chan, SNR, 'measured');

    % OFDM 解调
    rx_ofdm = ofdmdemod(rx_awgn, num_subcarrier, num_cp);

    H_ls = rx_ofdm ./ pilot_symbols; % LS估计

    % 输入：信道估计值的实部+虚部
    X_data(n,:,1) = real(H_ls);
    X_data(n,:,2) = imag(H_ls);

    % 修正路径延迟：考虑滤波器的群时延
    pathDelays_eff = pathDelays + toffset/fs;

    % 输出：保存修正后的路径延迟参数
    Y_data(n,:) = pathDelays_eff*1e8;
end

save('channel_dataset_sim.mat','X_data','Y_data');
fprintf('训练数据已保存到 channel_dataset_sim.mat\n');
fprintf('X_data维度: [%d, %d, %d]\n', size(X_data));
fprintf('Y_data维度: [%d, %d] (只包含%d条修正后的路径延迟)\n', size(Y_data), numPaths);