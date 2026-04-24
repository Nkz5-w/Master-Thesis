# test_model_simple.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


# =========================
# Dataset
# =========================
class CKMResidualDataset:
    def __init__(self, file_path):
        data = np.load(file_path)

        self.H_ls = data["H_ls"]  # (num_samples, K, N)
        self.old_ckm = data["old_ckm"]  # (num_samples, 3L)
        self.new_ckm = data["new_ckm"]  # (num_samples, 3L)
        self.L = self.old_ckm.shape[1] // 3

        print(f"Dataset loaded: {len(self.H_ls)} samples total")
        print(f"Parameters per CKM: {self.L} paths × 3 = {3 * self.L}")

    def __len__(self):
        return len(self.H_ls)

    def __getitem__(self, idx):
        H = self.H_ls[idx]
        old = self.old_ckm[idx]
        new = self.new_ckm[idx]

        # 复数拆分 (K, N, 2)
        H_input = np.stack([H.real, H.imag], axis=-1)

        return H_input, old, new


# =========================
# 模型定义
# =========================
class ImprovedCKMTransformer(nn.Module):
    def __init__(self, N=64, K=5, L=3, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.L = L

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(N * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # CKM投影
        self.ckm_proj = nn.Sequential(
            nn.Linear(3 * L, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # 可学习的位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, K + 1, d_model) * 0.02)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # 分参数预测头
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, L)
        )

        self.sigma_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, L)
        )

        self.power_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, L)
        )

    def forward(self, H, old_ckm):
        B, K, N, _ = H.shape

        # 编码H
        H = H.reshape(B, K, N * 2)
        H_encoded = self.input_proj(H)

        # 编码old_ckm
        ckm_token = self.ckm_proj(old_ckm).unsqueeze(1)

        # 拼接并添加位置编码
        x = torch.cat([ckm_token, H_encoded], dim=1)
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Transformer编码
        x = self.transformer(x)

        # 提取特征
        ckm_feat = x[:, 0, :]
        h_feat = x[:, 1:, :].mean(dim=1)

        # 特征融合
        combined = torch.cat([ckm_feat, h_feat], dim=-1)
        fused = self.fusion(combined)

        # 分头预测残差
        mu_delta = self.mu_head(fused)
        sigma_delta = self.sigma_head(fused)
        power_delta = self.power_head(fused)

        # 拼接残差
        delta = torch.cat([mu_delta, sigma_delta, power_delta], dim=-1)

        return delta


# =========================
# 预测函数（带物理约束）
# =========================
def predict_ckm(model, H, old_ckm, device='cuda'):
    """
    预测新的CKM参数

    Args:
        model: 训练好的模型
        H: 信道响应 (K, N) 复数或 (K, N, 2) 实数
        old_ckm: 旧的CKM参数 (3L,)
        device: 设备

    Returns:
        new_ckm: 预测的新CKM参数 (3L,)
    """
    model.eval()

    # 准备H输入
    if isinstance(H, np.ndarray):
        if H.ndim == 2:
            # H是复数 (K, N)
            H_input = np.stack([H.real, H.imag], axis=-1)  # (K, N, 2)
        elif H.ndim == 3 and H.shape[-1] == 2:
            # H已经是 (K, N, 2)
            H_input = H
        else:
            raise ValueError(f"Unexpected H shape: {H.shape}")

        H_tensor = torch.tensor(H_input, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        H_tensor = H.to(device)

    # 准备old_ckm输入
    if isinstance(old_ckm, np.ndarray):
        if old_ckm.ndim == 1:
            old_tensor = torch.tensor(old_ckm, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            old_tensor = torch.tensor(old_ckm, dtype=torch.float32).to(device)
    else:
        old_tensor = old_ckm.to(device)

    # 预测
    with torch.no_grad():
        delta = model(H_tensor, old_tensor)
        new_ckm = old_tensor + delta

    new_ckm = new_ckm.squeeze(0).cpu().numpy()

    # 物理约束
    L = 3
    mu = new_ckm[:L]
    sigma = new_ckm[L:2 * L]
    power = new_ckm[2 * L:]

    # 确保非负
    mu = np.maximum(mu, 0)
    sigma = np.maximum(sigma, 0.05)  # 最小0.05μs
    power = np.maximum(power, 1e-4)

    # 确保μ递增
    idx = np.argsort(mu)
    mu = mu[idx]
    sigma = sigma[idx]
    power = power[idx]

    # 确保σ不超过相邻μ间隔
    for i in range(L - 1):
        max_sigma = (mu[i + 1] - mu[i]) / 3
        if max_sigma > 0 and sigma[i] > max_sigma:
            sigma[i] = max_sigma

    return np.concatenate([mu, sigma, power])


# =========================
# 主测试函数
# =========================
def test_model(model_path="best_ckm_model.pth",
               file_path="ckm_residual_dataset.npz",
               num_samples=1000):
    """测试模型（只测试1000个样本）"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print("\n" + "=" * 60)
    print("Loading model...")
    model = ImprovedCKMTransformer(N=64, K=5, L=3).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path} (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # 加载数据
    print("\n" + "=" * 60)
    print("Loading dataset...")
    dataset = CKMResidualDataset(file_path)

    # 随机选择num_samples个样本
    total_samples = len(dataset)
    test_indices = np.random.choice(total_samples, min(num_samples, total_samples), replace=False)
    print(f"Testing on {len(test_indices)} random samples")

    # 存储结果
    mu_errors = []
    sigma_errors = []
    power_errors = []
    mu_rel_errors = []
    sigma_rel_errors = []
    power_rel_errors = []

    predictions = []
    truths = []
    olds = []

    print("\n" + "=" * 60)
    print("Testing...")

    for i, idx in enumerate(test_indices):
        H, old_ckm, true_new = dataset[idx]

        # 预测
        H_np = H
        old_np = old_ckm
        true_np = true_new

        pred_new = predict_ckm(model, H_np, old_np, device)

        # 保存
        predictions.append(pred_new)
        truths.append(true_np)
        olds.append(old_np)

        # 计算误差
        mu_error = np.abs(pred_new[:3] - true_np[:3])
        sigma_error = np.abs(pred_new[3:6] - true_np[3:6])
        power_error = np.abs(pred_new[6:] - true_np[6:])

        mu_errors.append(mu_error)
        sigma_errors.append(sigma_error)
        power_errors.append(power_error)

        # 相对误差
        mu_rel_error = mu_error / (np.abs(true_np[:3]) + 1e-6)
        sigma_rel_error = sigma_error / (np.abs(true_np[3:6]) + 1e-6)
        power_rel_error = power_error / (np.abs(true_np[6:]) + 1e-6)

        mu_rel_errors.append(mu_rel_error)
        sigma_rel_errors.append(sigma_rel_error)
        power_rel_errors.append(power_rel_error)

        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_indices)} samples")

    # 转换为numpy数组
    mu_errors = np.array(mu_errors).flatten()
    sigma_errors = np.array(sigma_errors).flatten()
    power_errors = np.array(power_errors).flatten()
    mu_rel_errors = np.array(mu_rel_errors).flatten()
    sigma_rel_errors = np.array(sigma_rel_errors).flatten()
    power_rel_errors = np.array(power_rel_errors).flatten()

    predictions = np.array(predictions)
    truths = np.array(truths)
    olds = np.array(olds)

    # ========== 统计结果 ==========
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"\n📊 Absolute Errors (MAE ± Std):")
    print(f"  μ (delay)      : {np.mean(mu_errors):.6f} ± {np.std(mu_errors):.6f}")
    print(f"  σ (delay spread): {np.mean(sigma_errors):.6f} ± {np.std(sigma_errors):.6f}")
    print(f"  Power          : {np.mean(power_errors):.6f} ± {np.std(power_errors):.6f}")
    print(f"  Overall        : {np.mean(np.concatenate([mu_errors, sigma_errors, power_errors])):.6f}")

    print(f"\n📈 Relative Errors (MAE ± Std):")
    print(f"  μ (delay)      : {np.mean(mu_rel_errors):.4f} ± {np.std(mu_rel_errors):.4f}")
    print(f"  σ (delay spread): {np.mean(sigma_rel_errors):.4f} ± {np.std(sigma_rel_errors):.4f}")
    print(f"  Power          : {np.mean(power_rel_errors):.4f} ± {np.std(power_rel_errors):.4f}")

    print(f"\n📊 Error Percentiles:")
    print(
        f"  μ  : 25%={np.percentile(mu_errors, 25):.6f}, 50%={np.percentile(mu_errors, 50):.6f}, 75%={np.percentile(mu_errors, 75):.6f}")
    print(
        f"  σ  : 25%={np.percentile(sigma_errors, 25):.6f}, 50%={np.percentile(sigma_errors, 50):.6f}, 75%={np.percentile(sigma_errors, 75):.6f}")
    print(
        f"  Pow: 25%={np.percentile(power_errors, 25):.6f}, 50%={np.percentile(power_errors, 50):.6f}, 75%={np.percentile(power_errors, 75):.6f}")

    # 找出最佳和最差预测
    total_errors = np.mean(np.abs(predictions - truths), axis=1)
    best_idx = np.argmin(total_errors)
    worst_idx = np.argmax(total_errors)

    # ========== 显示示例 ==========
    print("\n" + "=" * 60)
    print("PREDICTION EXAMPLES")
    print("=" * 60)

    # 显示5个随机示例
    print("\n📌 Random Examples:")
    random_indices = np.random.choice(len(test_indices), min(5, len(test_indices)), replace=False)

    for i, sample_idx in enumerate(random_indices):
        print(f"\n--- Example {i + 1} (Dataset Index: {test_indices[sample_idx]}) ---")
        print(f"Old CKM:     μ=[{olds[sample_idx, 0]:.4f}, {olds[sample_idx, 1]:.4f}, {olds[sample_idx, 2]:.4f}] "
              f"σ=[{olds[sample_idx, 3]:.4f}, {olds[sample_idx, 4]:.4f}, {olds[sample_idx, 5]:.4f}] "
              f"P=[{olds[sample_idx, 6]:.4f}, {olds[sample_idx, 7]:.4f}, {olds[sample_idx, 8]:.4f}]")

        print(f"True New:    μ=[{truths[sample_idx, 0]:.4f}, {truths[sample_idx, 1]:.4f}, {truths[sample_idx, 2]:.4f}] "
              f"σ=[{truths[sample_idx, 3]:.4f}, {truths[sample_idx, 4]:.4f}, {truths[sample_idx, 5]:.4f}] "
              f"P=[{truths[sample_idx, 6]:.4f}, {truths[sample_idx, 7]:.4f}, {truths[sample_idx, 8]:.4f}]")

        print(
            f"Predicted:   μ=[{predictions[sample_idx, 0]:.4f}, {predictions[sample_idx, 1]:.4f}, {predictions[sample_idx, 2]:.4f}] "
            f"σ=[{predictions[sample_idx, 3]:.4f}, {predictions[sample_idx, 4]:.4f}, {predictions[sample_idx, 5]:.4f}] "
            f"P=[{predictions[sample_idx, 6]:.4f}, {predictions[sample_idx, 7]:.4f}, {predictions[sample_idx, 8]:.4f}]")

        print(f"Error:       μ=[{abs(predictions[sample_idx, 0] - truths[sample_idx, 0]):.4f}, "
              f"{abs(predictions[sample_idx, 1] - truths[sample_idx, 1]):.4f}, "
              f"{abs(predictions[sample_idx, 2] - truths[sample_idx, 2]):.4f}] "
              f"σ=[{abs(predictions[sample_idx, 3] - truths[sample_idx, 3]):.4f}, "
              f"{abs(predictions[sample_idx, 4] - truths[sample_idx, 4]):.4f}, "
              f"{abs(predictions[sample_idx, 5] - truths[sample_idx, 5]):.4f}] "
              f"P=[{abs(predictions[sample_idx, 6] - truths[sample_idx, 6]):.4f}, "
              f"{abs(predictions[sample_idx, 7] - truths[sample_idx, 7]):.4f}, "
              f"{abs(predictions[sample_idx, 8] - truths[sample_idx, 8]):.4f}]")

        print(f"Total MAE: {total_errors[sample_idx]:.6f}")

    # 显示最佳预测
    print("\n" + "=" * 60)
    print(f"✨ BEST PREDICTION (Total MAE: {total_errors[best_idx]:.6f})")
    print(f"Dataset Index: {test_indices[best_idx]}")
    print(f"Old CKM:     μ={olds[best_idx, :3]}, σ={olds[best_idx, 3:6]}, P={olds[best_idx, 6:]}")
    print(f"True New:    μ={truths[best_idx, :3]}, σ={truths[best_idx, 3:6]}, P={truths[best_idx, 6:]}")
    print(f"Predicted:   μ={predictions[best_idx, :3]}, σ={predictions[best_idx, 3:6]}, P={predictions[best_idx, 6:]}")
    print(f"Error:       μ={np.abs(predictions[best_idx, :3] - truths[best_idx, :3])}")
    print(f"             σ={np.abs(predictions[best_idx, 3:6] - truths[best_idx, 3:6])}")
    print(f"             P={np.abs(predictions[best_idx, 6:] - truths[best_idx, 6:])}")

    # 显示最差预测
    print("\n" + "=" * 60)
    print(f"⚠️  WORST PREDICTION (Total MAE: {total_errors[worst_idx]:.6f})")
    print(f"Dataset Index: {test_indices[worst_idx]}")
    print(f"Old CKM:     μ={olds[worst_idx, :3]}, σ={olds[worst_idx, 3:6]}, P={olds[worst_idx, 6:]}")
    print(f"True New:    μ={truths[worst_idx, :3]}, σ={truths[worst_idx, 3:6]}, P={truths[worst_idx, 6:]}")
    print(
        f"Predicted:   μ={predictions[worst_idx, :3]}, σ={predictions[worst_idx, 3:6]}, P={predictions[worst_idx, 6:]}")
    print(f"Error:       μ={np.abs(predictions[worst_idx, :3] - truths[worst_idx, :3])}")
    print(f"             σ={np.abs(predictions[worst_idx, 3:6] - truths[worst_idx, 3:6])}")
    print(f"             P={np.abs(predictions[worst_idx, 6:] - truths[worst_idx, 6:])}")

    return predictions, truths, olds


# =========================
# 主函数
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test CKM Model')
    parser.add_argument('--model', type=str, default='best_ckm_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='ckm_residual_dataset.npz',
                        help='Path to dataset')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of test samples')

    args = parser.parse_args()

    # 运行测试
    test_model(
        model_path=args.model,
        file_path=args.data,
        num_samples=args.num_samples
    )