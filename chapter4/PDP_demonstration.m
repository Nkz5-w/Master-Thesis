clear; clc; close all;

% 参数设置
L = 3;  % 3条路径

% 确定性PDP参数
tau_det = [0.5, 1.2, 2.0] * 1e-6;  % 时延 (秒)
p = [1.0, 0.7, 0.4];               % 功率

% 概率性PDP参数
mu_tau = tau_det;                   % 时延均值 (秒)
sigma_tau = [0.08, 0.12, 0.15] * 1e-6;  % 时延标准差 (秒)

% 时延网格
tau_grid = linspace(0, 3.5, 1000) * 1e-6;  % 0 到 3.5 μs

% 计算概率性PDP (高斯分布)
pdp_prob = zeros(size(tau_grid));
for l = 1:L
    % 高斯分布
    gaussian = exp(-(tau_grid - mu_tau(l)).^2 / (2 * sigma_tau(l)^2));
    gaussian = gaussian / (sqrt(2*pi) * sigma_tau(l));
    % 加权
    pdp_prob = pdp_prob + p(l) * gaussian;
end
% 归一化使峰值合理显示
pdp_prob = pdp_prob / max(pdp_prob) * max(p);

% 画图
figure('Position', [100, 100, 700, 500]);

% 确定性PDP (离散脉冲)
stem(tau_det*1e6, p, 'b', 'LineWidth', 1.2, 'MarkerSize', 6, 'MarkerFaceColor', 'b',...
     'DisplayName', '确定性PDP');
hold on;

% 概率性PDP (连续曲线)
plot(tau_grid*1e6, pdp_prob, 'r-', 'LineWidth', 1.2, ...
     'DisplayName', '概率性PDP');

grid on;
xlabel('路径时延 (μs)','FontSize', 12);
ylabel('平均功率', 'FontSize', 12);
legend('确定型CKM','概率型CKM','Location', 'northeast', 'FontSize', 12);
xlim([0, 3]);
ylim([0, 1.2]);

% 打印参数
fprintf('\n=== 确定性PDP ===\n');
fprintf('路径\t时延(μs)\t功率\n');
for l = 1:L
    fprintf('%d\t%.2f\t\t%.2f\n', l, tau_det(l)*1e6, p(l));
end

fprintf('\n=== 概率性PDP ===\n');
fprintf('路径\t均值(μs)\t标准差(ns)\t功率\n');
for l = 1:L
    fprintf('%d\t%.2f\t\t%.0f\t\t%.2f\n', l, mu_tau(l)*1e6, sigma_tau(l)*1e6*1000, p(l));
end