clear; clc; close all;rng(42);

% ================= 参数 =================
N = 64;              % 子载波
L = 3;
SNR_dB = 6:2:24;
MC = 2000;

df = 15e3;
f = (0:N-1)' * df;

% ===== 信道 =====
tau_true = [0.5, 1.2, 2.0]*1e-6;
p_true = [1, 0.7, 0.4];

% ===== CKM =====
tau_ckm = tau_true + 0.3e-6;
p_ckm = p_true + 0.2;
sigma_tau_ckm = 1e-7;
sigma_tau_ckm_1 = 3e-9;
sigma_tau_ckm_2 = 1e-5;

BER_LS = zeros(size(SNR_dB));
BER_det = zeros(size(SNR_dB));
BER_prob = zeros(size(SNR_dB));
BER_prob1 = zeros(size(SNR_dB));
BER_prob2 = zeros(size(SNR_dB));
BER_ideal = zeros(size(SNR_dB));

MSE_LS = zeros(size(SNR_dB));
MSE_det = zeros(size(SNR_dB));
MSE_prob = zeros(size(SNR_dB));
MSE_prob1 = zeros(size(SNR_dB));
MSE_prob2 = zeros(size(SNR_dB));

% --- Probabilistic CKM ---
R_prob = build_probabilistic_CKM_optimized(N, f, tau_ckm, p_ckm, sigma_tau_ckm);
% --- Prob CKM 1 (小sigma) ---
R_prob1 = build_probabilistic_CKM_optimized(N, f, tau_ckm, p_ckm, sigma_tau_ckm_1);
% --- Prob CKM 2 (大sigma) ---
R_prob2 = build_probabilistic_CKM_optimized(N, f, tau_ckm, p_ckm, sigma_tau_ckm_2);

%% ================= CKM构造 =================
% --- Deterministic CKM ---
R_det = zeros(N,N);
for l = 1:L
    a = exp(-1j*2*pi*f*tau_ckm(l));
    R_det = R_det + p_ckm(l)*(a*a');
end

% ================= 主循环 =================
for snr_idx = 1:length(SNR_dB)

    snr = 10^(SNR_dB(snr_idx)/10);
    noise_var = 1/snr;

    err_ls = 0;
    err_det = 0;
    err_prob = 0;
    err_ideal = 0;

    errMSE_ls = 0;
    errMSE_det = 0;
    errMSE_prob = 0;

    err_prob1 = 0;
    err_prob2 = 0;
    
    errMSE_prob1 = 0;
    errMSE_prob2 = 0;

    for mc = 1:MC

        % ===== 生成信道（整个frame固定）=====
        h = zeros(N,1);
        for l = 1:L
            alpha = sqrt(p_true(l)/2)*(randn + 1j*randn);
            h = h + alpha * exp(-1j*2*pi*f*tau_true(l));
        end

        %% ================= Pilot OFDM =================
        pilot = ones(N,1);  % block pilot

        noise_p = sqrt(noise_var/2)*(randn(N,1)+1j*randn(N,1));
        y_pilot = pilot .* h + noise_p;

        % ===== 信道估计 =====
        h_ls = y_pilot ./ pilot;
        h_prob = R_prob * ((R_prob + noise_var*eye(N)) \ h_ls);
        h_prob1 = R_prob1 * ((R_prob1 + noise_var*eye(N)) \ h_ls);
        h_prob2 = R_prob2 * ((R_prob2 + noise_var*eye(N)) \ h_ls);
        h_det = R_det * ((R_det + noise_var*eye(N)) \ h_ls);

        %% ================= Data OFDM =================
        bits = randi([0 1], 2*N, 1);

        % QPSK
        symbols = (2*bits(1:2:end)-1) + 1j*(2*bits(2:2:end)-1);
        symbols = symbols / sqrt(2);

        noise_d = sqrt(noise_var/2)*(randn(N,1)+1j*randn(N,1));
        y_data = symbols .* h + noise_d;

        %% ================= 均衡 =================
        x_ls   = y_data ./ h_ls;
        x_det  = y_data ./ h_det;
        x_prob = y_data ./ h_prob;
        x_ideal = y_data ./ h;
        x_prob1 = y_data ./ h_prob1;
        x_prob2 = y_data ./ h_prob2;

        %% ================= 解调 =================
        bits_ls   = qpsk_demod(x_ls);
        bits_det  = qpsk_demod(x_det);
        bits_prob = qpsk_demod(x_prob);
        bits_ideal= qpsk_demod(x_ideal);
        bits_prob1 = qpsk_demod(x_prob1);
        bits_prob2 = qpsk_demod(x_prob2);

        %% ================= 误码 =================
        err_ls   = err_ls   + sum(bits ~= bits_ls);
        err_det  = err_det  + sum(bits ~= bits_det);
        err_prob = err_prob + sum(bits ~= bits_prob);
        err_ideal = err_ideal + sum(bits ~= bits_ideal);
        err_prob1 = err_prob1 + sum(bits ~= bits_prob1);
        err_prob2 = err_prob2 + sum(bits ~= bits_prob2);

        % ===== MSE统计（信道估计误差）=====
        mse_ls   = norm(h - h_ls)^2;
        mse_det  = norm(h - h_det)^2;
        mse_prob = norm(h - h_prob)^2;
        
        errMSE_ls   = errMSE_ls   + mse_ls;
        errMSE_det  = errMSE_det  + mse_det;
        errMSE_prob = errMSE_prob + mse_prob;

        mse_prob1 = norm(h - h_prob1)^2;
        mse_prob2 = norm(h - h_prob2)^2;
        
        errMSE_prob1 = errMSE_prob1 + mse_prob1;
        errMSE_prob2 = errMSE_prob2 + mse_prob2;

    end
    BER_LS(snr_idx)   = err_ls   / (MC * 2*N);
    BER_det(snr_idx)  = err_det  / (MC * 2*N);
    BER_prob(snr_idx) = err_prob / (MC * 2*N);
    BER_ideal(snr_idx) = err_ideal / (MC * 2*N);
    
    MSE_LS(snr_idx)   = errMSE_ls   / (MC * N);
    MSE_det(snr_idx)  = errMSE_det  / (MC * N);
    MSE_prob(snr_idx) = errMSE_prob / (MC * N);

    BER_prob1(snr_idx) = err_prob1 / (MC * 2*N);
    BER_prob2(snr_idx) = err_prob2 / (MC * 2*N);
    
    MSE_prob1(snr_idx) = errMSE_prob1 / (MC * N);
    MSE_prob2(snr_idx) = errMSE_prob2 / (MC * N);
end

% ================= 画图 =================
figure('Position', [100, 100, 600, 450], 'Color', 'w');
% 定义滑动平均窗口大小（可根据曲线平滑度需求调整）
windowSize = 2;  % 窗口大小，奇数较好

% 对每条BER曲线进行滑动平均平滑（注意BER是对数坐标，需要在线性域平滑）
% LS
BER_LS_smooth = movmean(BER_LS, windowSize);
semilogy(SNR_dB, BER_LS_smooth, '-o', 'LineWidth', 1.2, 'MarkerFaceColor', [0,0.447,0.741]); 
hold on;

% 确定型CKM
BER_det_smooth = movmean(BER_det, windowSize);
semilogy(SNR_dB, BER_det_smooth, '-v', 'LineWidth', 1.2, 'MarkerFaceColor', [0.850,0.325,0.098]);

% 概率型CKM
BER_prob_smooth = movmean(BER_prob, windowSize);
semilogy(SNR_dB, BER_prob_smooth, '-s', 'LineWidth', 1.2, 'MarkerFaceColor', [0.929,0.694,0.125]);

% Prob CKM (sigma小)
BER_prob1_smooth = movmean(BER_prob1, windowSize);
semilogy(SNR_dB, BER_prob1_smooth, '--^', 'LineWidth', 1.2, 'MarkerFaceColor', [0.494,0.184,0.556]);

% Prob CKM (sigma大)
BER_prob2_smooth = movmean(BER_prob2, windowSize);
semilogy(SNR_dB, BER_prob2_smooth, '--d', 'LineWidth', 1.2, 'MarkerFaceColor', [0.466,0.674,0.188]);

% 理想CFR（可选平滑或不平滑）
BER_ideal_smooth = movmean(BER_ideal, windowSize);
semilogy(SNR_dB, BER_ideal_smooth, '-rp', 'LineWidth', 1.2, 'MarkerFaceColor', [1,0,0]);

grid on;
xlabel('SNR (dB)','FontName','Times New Roman');
ylabel('BER','FontName','Times New Roman');
legend('LS', 'Det-CKM', 'Prob-CKM(\sigma=0.1\mus)','Prob-CKM(\sigma=0.001\mus)', 'Prob-CKM(\sigma=10\mus)','Ideal CFR','FontName','Times New Roman');
xlim([6 24]);

% 计算纵轴范围时使用平滑后的数据
allBER_smooth = [BER_ideal_smooth, BER_prob_smooth, BER_det_smooth, BER_LS_smooth];
BERmin = min(allBER_smooth);
BERmax = max(allBER_smooth);
ylim([BERmin*0.7, BERmax*1.3]);

figure('Position', [100, 100, 600, 450], 'Color', 'w');
plot(SNR_dB, 10*log10(MSE_LS), '-o', 'LineWidth', 1.2,'MarkerFaceColor', [0,0.447,0.741]); hold on;
plot(SNR_dB, 10*log10(MSE_det), '-v', 'LineWidth', 1.2,'MarkerFaceColor', [0.850,0.325,0.098]);
plot(SNR_dB, 10*log10(MSE_prob), '-s', 'LineWidth', 1.2,'MarkerFaceColor', [0.929,0.694,0.125]);
plot(SNR_dB, 10*log10(MSE_prob1), '--^', 'LineWidth', 1.2,'MarkerFaceColor', [0.494,0.184,0.556]);
plot(SNR_dB, 10*log10(MSE_prob2), '--d', 'LineWidth', 1.2,'MarkerFaceColor', [0.466,0.674,0.188]);

xlabel('SNR (dB)','FontName','Times New Roman');
ylabel('MSE (dB)','FontName','Times New Roman');
legend('LS','Det-CKM','Prob-CKM(\sigma=0.1\mus)','Prob-CKM(\sigma=0.001\mus)','Prob-CKM(\sigma=10\mus)','FontName','Times New Roman');
xlim([6 24]);
allMSE = [10*log10(MSE_prob), 10*log10(MSE_det), 10*log10(MSE_LS)];
MSEmin = min(allMSE);
MSEmax = max(allMSE);
ylim([-35 0])
grid on;

%% ================= 函数 =================

function bits_hat = qpsk_demod(x)
    bits_hat = zeros(2*length(x),1);
    bits_hat(1:2:end) = real(x) > 0;
    bits_hat(2:2:end) = imag(x) > 0;
end

function R_prob = build_probabilistic_CKM_optimized(N, f, tau_mean, p, sigma_tau)
    df_matrix = f - f';
    
    R_prob = zeros(N,N);
    for l = 1:length(tau_mean)
        phase_term = exp(-1j*2*pi*df_matrix*tau_mean(l));
        mag_term = exp(-2*pi^2*(df_matrix.^2)*(sigma_tau^2));
        R_prob = R_prob + p(l) * phase_term .* mag_term;
    end
end