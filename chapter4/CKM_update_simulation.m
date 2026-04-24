clc; clear; close all; rng(3);

% =========================
% 参数
% =========================
N = 64;
L = 4;
f = (0:N-1)' * 15e3;

T = 200;              % 总时间长度
T_change = 10;        % 每10个slot变化一次

SNR_dB = 20;
noise_var = 10^(-SNR_dB/10);

gamma1 = 0;
gamma2 = 75;
gamma3 = 150;

num_realizations = 500;  % 蒙特卡罗次数

% 存储所有 realization 的结果
MSE_no_all = zeros(num_realizations, T);
MSE_kl1_all = zeros(num_realizations, T);
MSE_kl2_all = zeros(num_realizations, T);
MSE_kl3_all = zeros(num_realizations, T);

% =========================
% 蒙特卡罗循环
% =========================
for n_real = 1:num_realizations
    
    if mod(n_real, 10) == 0
        fprintf('正在运行第 %d / %d 次蒙特卡罗...\n', n_real, num_realizations);
    end

    MSE_no = zeros(1,T);
    MSE_kl1 = zeros(1,T);
    MSE_kl2 = zeros(1,T);
    MSE_kl3 = zeros(1,T);

    % 记录更新点
    update_points1 = [];
    update_points2 = [];
    update_points3 = [];

    % =========================
    % 初始化 old CKM
    % =========================
    mu_old = sort(rand(1,L)*3e-6);
    sigma_old = 0.1e-6 + rand(1,L)*0.2e-6;
    p_old = exp(-sort(rand(1,L)*2));
    p_old = p_old / sum(p_old);

    % 当前真实CKM初始化
    mu_true = mu_old;
    sigma_true = sigma_old;
    p_true = p_old;

    mu = mu_old;
    sigma = sigma_old;
    p = p_old;
    R = build_probabilistic_CKM_optimized(N,f,mu,p,sigma);

    % 为三个曲线分别保存状态
    mu_old1 = mu_old; sigma_old1 = sigma_old; p_old1 = p_old;
    mu_old2 = mu_old; sigma_old2 = sigma_old; p_old2 = p_old;
    mu_old3 = mu_old; sigma_old3 = sigma_old; p_old3 = p_old;

    % =========================
    % 时间演化
    % =========================
    for t = 1:T

        % =========================
        % 1️⃣ 信道统计变化（慢变化）
        % =========================
        if mod(t, T_change) == 1 && t > 1
            scale = 0.3;

            mu_true = mu_true + randn(1,L)*scale*2e-6;
            sigma_true = sigma_true + randn(1,L)*scale*0.3e-6;
            p_true = p_true .* exp(randn(1,L)*scale*0.3);

            mu_true = max(mu_true,0);
            sigma_true = max(sigma_true,0.05e-6);
            p_true = max(p_true,1e-4);
            p_true = p_true / sum(p_true);
        end

        % =========================
        % 2️⃣ 生成瞬时路径（小尺度变化）
        % =========================
        tau_true = mu_true + sigma_true .* randn(1,L);

        % =========================
        % 3️⃣ 构造真实信道
        % =========================
        h = zeros(N,1);
        for l = 1:L
            alpha = sqrt(p_true(l)/2)*(randn + 1j*randn);
            h = h + alpha * exp(-1j*2*pi*f*tau_true(l));
        end

        % =========================
        % 4️⃣ 导频发送
        % =========================
        noise = sqrt(noise_var/2)*(randn(N,1)+1j*randn(N,1));
        y = h + noise;
        h_ls = y;

        % =========================
        % 5️⃣ 构造 CKM
        % =========================

        R_old1 = build_probabilistic_CKM_optimized(N,f,mu_old1,p_old1,sigma_old1);
        R_old2 = build_probabilistic_CKM_optimized(N,f,mu_old2,p_old2,sigma_old2);
        R_old3 = build_probabilistic_CKM_optimized(N,f,mu_old3,p_old3,sigma_old3);
        R_true = build_probabilistic_CKM_optimized(N,f,mu_true,p_true,sigma_true);

        % =========================
        % ❌ No Update
        % =========================
        h_no = R / (R + noise_var*eye(N)) * h_ls;

        % =========================
        % ✅ KL Update - gamma1
        % =========================
        D_ckm1 = symmetric_KL(mu_old1,sigma_old1,mu_true,sigma_true,p_old1);

        if D_ckm1 > gamma1
            mu_old1 = mu_true;
            sigma_old1 = sigma_true;
            p_old1 = p_true;
            
            update_points1 = [update_points1, t];
            
            R_kl1 = build_probabilistic_CKM_optimized(N,f,mu_old1,p_old1,sigma_old1);
        else
            R_kl1 = R_old1;
        end

        h_kl1 = R_kl1 / (R_kl1+ noise_var*eye(N)) * h_ls;

        % =========================
        % ✅ KL Update - gamma2
        % =========================
        D_ckm2 = symmetric_KL(mu_old2,sigma_old2,mu_true,sigma_true,p_old2);

        if D_ckm2 > gamma2
            mu_old2 = mu_true;
            sigma_old2 = sigma_true;
            p_old2 = p_true;
            
            update_points2 = [update_points2, t];
            
            R_kl2 = build_probabilistic_CKM_optimized(N,f,mu_old2,p_old2,sigma_old2);
        else
            R_kl2 = R_old2;
        end

        h_kl2 = R_kl2 / (R_kl2+ noise_var*eye(N)) * h_ls;

        % =========================
        % ✅ KL Update - gamma3
        % =========================
        D_ckm3 = symmetric_KL(mu_old3,sigma_old3,mu_true,sigma_true,p_old3);

        if D_ckm3 > gamma3
            mu_old3 = mu_true;
            sigma_old3 = sigma_true;
            p_old3 = p_true;
            
            update_points3 = [update_points3, t];
            
            R_kl3 = build_probabilistic_CKM_optimized(N,f,mu_old3,p_old3,sigma_old3);
        else
            R_kl3 = R_old3;
        end

        h_kl3 = R_kl3 / (R_kl3+ noise_var*eye(N)) * h_ls;

        % =========================
        % 6️⃣ 误差统计（转换为dB）
        % =========================
        % 计算线性MSE
        mse_no_linear = norm(h - h_no)^2 / norm(h)^2;
        mse_kl1_linear = norm(h - h_kl1)^2 / norm(h)^2;
        mse_kl2_linear = norm(h - h_kl2)^2 / norm(h)^2;
        mse_kl3_linear = norm(h - h_kl3)^2 / norm(h)^2;
        
        % 转换为dB单位 (10*log10)
        MSE_no(t) = 10 * log10(mse_no_linear);
        MSE_kl1(t) = 10 * log10(mse_kl1_linear);
        MSE_kl2(t) = 10 * log10(mse_kl2_linear);
        MSE_kl3(t) = 10 * log10(mse_kl3_linear);
    end
    
    % 存储本次 realization 的结果
    MSE_no_all(n_real, :) = MSE_no;
    MSE_kl1_all(n_real, :) = MSE_kl1;
    MSE_kl2_all(n_real, :) = MSE_kl2;
    MSE_kl3_all(n_real, :) = MSE_kl3;
end

% =========================
% 计算平均值
% =========================
MSE_no_avg = mean(MSE_no_all, 1);
MSE_kl1_avg = mean(MSE_kl1_all, 1);
MSE_kl2_avg = mean(MSE_kl2_all, 1);
MSE_kl3_avg = mean(MSE_kl3_all, 1);

% =========================
% 平滑处理（多重平滑，让曲线更平滑）
% =========================
win1 = 10;   % 第一轮平滑窗口
win2 = 20;  % 第二轮平滑窗口

% 对平均曲线进行两次平滑
MSE_no_s = movmean(MSE_no_avg, win1);
MSE_no_s = movmean(MSE_no_s, win2);

MSE_kl1_s = movmean(MSE_kl1_avg, win1);
MSE_kl1_s = movmean(MSE_kl1_s, win2);

MSE_kl2_s = movmean(MSE_kl2_avg, win1);
MSE_kl2_s = movmean(MSE_kl2_s, win2);

MSE_kl3_s = movmean(MSE_kl3_avg, win1);
MSE_kl3_s = movmean(MSE_kl3_s, win2);

% =========================
% 绘图
% =========================
% figure;
% plot(1:T, MSE_no_s, '-','LineWidth',1.4); hold on;
% plot(1:T, MSE_kl1_s, '--','LineWidth',1.4);
% plot(1:T, MSE_kl2_s, ':','LineWidth',1.4);
% plot(1:T, MSE_kl3_s, '-.','LineWidth',1.4);
% 
% grid on;
% 定义下采样间隔
decimation_factor = 20;

% 生成包含最后一个点的索引
selected_idx = 1:decimation_factor:T;
if selected_idx(end) ~= T
    selected_idx = [selected_idx, T];
end

% 绘制四条曲线，添加不同的标记符号
plot(1:T, MSE_no_s, '-o', 'LineWidth', 1.4, 'MarkerIndices', selected_idx,'MarkerFaceColor', [0,0.447,0.741]); hold on;
plot(1:T, MSE_kl1_s, '-s', 'LineWidth', 1.4, 'MarkerIndices', selected_idx,'MarkerFaceColor', [0.850,0.325,0.098]);
plot(1:T, MSE_kl2_s, '-^', 'LineWidth', 1.4, 'MarkerIndices', selected_idx,'MarkerFaceColor', [0.929,0.694,0.125]);
plot(1:T, MSE_kl3_s, '-d', 'LineWidth', 1.4, 'MarkerIndices', selected_idx,'MarkerFaceColor', [0.494,0.184,0.556]);
grid on;

legend('无更新', 'SKL更新 (\gamma=0)', 'SKL更新 (\gamma=75)', 'SKL更新 (\gamma=150)', 'Location', 'best');
xlabel('时间索引');
ylabel('MSE (dB)', 'FontName', 'Times New Roman');
ylim([-35,-5])

function R_prob = build_probabilistic_CKM_optimized(N, f, tau_mean, p, sigma_tau)
    % 利用向量化避免三重循环
    df_matrix = f - f';  % N×N矩阵，df_ij = f(i)-f(j)
    
    R_prob = zeros(N,N);
    for l = 1:length(tau_mean)
        % 相位项：exp(-j2π*df*μ_τ)
        phase_term = exp(-1j*2*pi*df_matrix*tau_mean(l));
        
        % 幅度项：exp(-2π²*df²*σ_τ²)
        mag_term = exp(-2*pi^2*(df_matrix.^2)*(sigma_tau(l)^2));
        
        % 组合
        R_prob = R_prob + p(l) * phase_term .* mag_term;
    end
end

function D = symmetric_KL(mu_old,sigma_old,mu_new,sigma_new,p)

eps = 1e-12;

sigma_old = max(sigma_old,eps);
sigma_new = max(sigma_new,eps);

w = p / sum(p);

L = length(mu_old);
D_l = zeros(1,L);

for l = 1:L

    term1 = (mu_old(l)-mu_new(l))^2/2 * ...
        (1/sigma_old(l)^2 + 1/sigma_new(l)^2);

    term2 = 0.5 * ( ...
        sigma_old(l)^2/sigma_new(l)^2 + ...
        sigma_new(l)^2/sigma_old(l)^2 - 2 );

    D_l(l) = term1 + term2;
end

D = sum(w .* D_l);
end