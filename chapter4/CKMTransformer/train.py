import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import warnings

warnings.filterwarnings('ignore')


# =========================
# 增强版 Dataset（带数据增强）
# =========================
class AugmentedCKMResidualDataset(Dataset):
    def __init__(self, file_path, augment=False):
        data = np.load(file_path)

        self.H_ls = data["H_ls"]  # (num_samples, K, N)
        self.old_ckm = data["old_ckm"]  # (num_samples, 3L)
        self.new_ckm = data["new_ckm"]  # (num_samples, 3L)
        self.L = self.old_ckm.shape[1] // 3
        self.augment = augment

        print(f"Dataset loaded: {len(self.H_ls)} samples")
        print(f"Parameters per CKM: {self.L} paths × 3 = {3 * self.L}")

    def __len__(self):
        return len(self.H_ls)

    def __getitem__(self, idx):
        H = self.H_ls[idx].copy()  # (K, N)
        old = self.old_ckm[idx]
        new = self.new_ckm[idx]

        # 数据增强
        if self.augment:
            # 频域增强：随机丢弃部分子载波
            if np.random.rand() < 0.3:
                N = H.shape[1]
                mask = np.random.binomial(1, 0.95, N)
                H = H * mask.reshape(1, -1)

            # 时域增强：随机打乱时隙顺序
            if np.random.rand() < 0.2:
                K = H.shape[0]
                perm = np.random.permutation(K)
                H = H[perm]

            # 添加高斯噪声
            if np.random.rand() < 0.3:
                noise_std = 0.05 * np.std(H)
                H = H + noise_std * np.random.randn(*H.shape)

        # 计算残差
        delta = new - old

        # 复数拆分 (K, N, 2)
        H_input = np.stack([H.real, H.imag], axis=-1)

        return (
            torch.tensor(H_input, dtype=torch.float32),
            torch.tensor(old, dtype=torch.float32),
            torch.tensor(delta, dtype=torch.float32)
        )


# =========================
# 改进的 Transformer 模型
# =========================
class ImprovedCKMTransformer(nn.Module):
    def __init__(self, N=64, K=5, L=3, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.L = L
        self.d_model = d_model
        self.K = K

        # 输入投影（带层归一化和GELU）
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

        # Transformer编码器（Pre-LN结构）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN，训练更稳定
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

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, H, old_ckm):
        B, K, N, _ = H.shape

        # 编码H
        H = H.reshape(B, K, N * 2)
        H_encoded = self.input_proj(H)  # (B, K, d_model)

        # 编码old_ckm
        ckm_token = self.ckm_proj(old_ckm).unsqueeze(1)  # (B, 1, d_model)

        # 拼接并添加位置编码
        x = torch.cat([ckm_token, H_encoded], dim=1)  # (B, K+1, d_model)
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Transformer编码
        x = self.transformer(x)  # (B, K+1, d_model)

        # 提取CKM token和H特征
        ckm_feat = x[:, 0, :]  # (B, d_model)
        h_feat = x[:, 1:, :].mean(dim=1)  # (B, d_model)

        # 特征融合
        combined = torch.cat([ckm_feat, h_feat], dim=-1)  # (B, 2*d_model)
        fused = self.fusion(combined)  # (B, d_model)

        # 分头预测
        mu_delta = self.mu_head(fused)
        sigma_delta = self.sigma_head(fused)
        power_delta = self.power_head(fused)

        # 拼接
        delta = torch.cat([mu_delta, sigma_delta, power_delta], dim=-1)

        return delta


# =========================
# 改进的损失函数
# =========================
class ImprovedLoss(nn.Module):
    def __init__(self, L, lambda_mu=1.0, lambda_sigma=0.8, lambda_power=0.5,
                 lambda_ordering=0.1, lambda_smooth=0.05):
        super().__init__()
        self.L = L
        self.lambda_mu = lambda_mu
        self.lambda_sigma = lambda_sigma
        self.lambda_power = lambda_power
        self.lambda_ordering = lambda_ordering
        self.lambda_smooth = lambda_smooth

    def forward(self, pred, target, old_ckm=None):
        # 分离参数
        pred_mu = pred[:, :self.L]
        pred_sigma = pred[:, self.L:2 * self.L]
        pred_power = pred[:, 2 * self.L:]

        target_mu = target[:, :self.L]
        target_sigma = target[:, self.L:2 * self.L]
        target_power = target[:, 2 * self.L:]

        # MSE损失
        mu_mse = F.mse_loss(pred_mu, target_mu)
        sigma_mse = F.mse_loss(pred_sigma, target_sigma)
        power_mse = F.mse_loss(pred_power, target_power)

        # Smooth L1损失（对异常值更鲁棒）
        mu_smooth = F.smooth_l1_loss(pred_mu, target_mu)
        sigma_smooth = F.smooth_l1_loss(pred_sigma, target_sigma)
        power_smooth = F.smooth_l1_loss(pred_power, target_power)

        # 组合基础损失
        base_loss = (
                self.lambda_mu * (mu_mse + self.lambda_smooth * mu_smooth) +
                self.lambda_sigma * (sigma_mse + self.lambda_smooth * sigma_smooth) +
                self.lambda_power * (power_mse + self.lambda_smooth * power_smooth)
        )

        # 物理约束损失
        ordering_loss = 0
        smoothness_loss = 0

        if old_ckm is not None:
            # 预测的new_ckm
            new_mu = old_ckm[:, :self.L] + pred_mu
            new_sigma = old_ckm[:, self.L:2 * self.L] + pred_sigma
            new_power = old_ckm[:, 2 * self.L:] + pred_power

            # 排序约束：μ应该递增
            mu_diff = new_mu[:, 1:] - new_mu[:, :-1]
            ordering_loss = F.relu(-mu_diff).mean()

            # 平滑性约束：相邻路径的μ应该有一定间隔
            mu_gap = mu_diff
            smoothness_loss = F.relu(0.1 - mu_gap).mean()  # 至少间隔0.1μs

        total_loss = base_loss + self.lambda_ordering * ordering_loss + 0.05 * smoothness_loss

        return total_loss


# =========================
# 训练函数
# =========================
def train_model(file_path="ckm_residual_dataset.npz",
                batch_size=64,
                epochs=50,
                learning_rate=1e-3,
                use_cuda=True):
    # 设置设备
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    print("\n" + "=" * 50)
    print("Loading dataset...")
    dataset = AugmentedCKMResidualDataset(file_path, augment=False)

    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # 创建模型
    print("\n" + "=" * 50)
    print("Creating model...")
    model = ImprovedCKMTransformer(
        N=64, K=5, L=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数
    criterion = ImprovedLoss(L=3, lambda_mu=1.0, lambda_sigma=0.8, lambda_power=0.5)

    # 优化器（AdamW + 权重衰减）
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # 训练记录
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\n" + "=" * 50)
    print("Starting training...")

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0
        train_mu_loss = 0
        train_sigma_loss = 0
        train_power_loss = 0

        for batch_idx, (H, old_ckm, target) in enumerate(train_loader):
            H = H.to(device)
            old_ckm = old_ckm.to(device)
            target = target.to(device)

            # 前向传播
            pred = model(H, old_ckm)
            loss = criterion(pred, target, old_ckm)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            # 统计各分量损失
            with torch.no_grad():
                mu_mse = F.mse_loss(pred[:, :3], target[:, :3]).item()
                sigma_mse = F.mse_loss(pred[:, 3:6], target[:, 3:6]).item()
                power_mse = F.mse_loss(pred[:, 6:], target[:, 6:]).item()
                train_mu_loss += mu_mse
                train_sigma_loss += sigma_mse
                train_power_loss += power_mse

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch + 1:2d} [{batch_idx + 1:4d}/{len(train_loader)}] Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_mu = train_mu_loss / len(train_loader)
        avg_train_sigma = train_sigma_loss / len(train_loader)
        avg_train_power = train_power_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0
        val_mu_loss = 0
        val_sigma_loss = 0
        val_power_loss = 0

        with torch.no_grad():
            for H, old_ckm, target in val_loader:
                H = H.to(device)
                old_ckm = old_ckm.to(device)
                target = target.to(device)

                pred = model(H, old_ckm)
                loss = criterion(pred, target, old_ckm)

                val_loss += loss.item()

                mu_mse = F.mse_loss(pred[:, :3], target[:, :3]).item()
                sigma_mse = F.mse_loss(pred[:, 3:6], target[:, 3:6]).item()
                power_mse = F.mse_loss(pred[:, 6:], target[:, 6:]).item()
                val_mu_loss += mu_mse
                val_sigma_loss += sigma_mse
                val_power_loss += power_mse

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mu = val_mu_loss / len(val_loader)
        avg_val_sigma = val_sigma_loss / len(val_loader)
        avg_val_power = val_power_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 打印结果
        print(f"\nEpoch {epoch + 1:2d}/{epochs}")
        print(
            f"  Train Loss: {avg_train_loss:.6f} (μ:{avg_train_mu:.6f}, σ:{avg_train_sigma:.6f}, P:{avg_train_power:.6f})")
        print(f"  Val   Loss: {avg_val_loss:.6f} (μ:{avg_val_mu:.6f}, σ:{avg_val_sigma:.6f}, P:{avg_val_power:.6f})")
        print(f"  LR: {current_lr:.2e}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, "best_ckm_model.pth")
            print(f"  -> Best model saved! (val_loss: {best_val_loss:.6f})")

        print("-" * 50)

    print("\n" + "=" * 50)
    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")

    return model, train_losses, val_losses


# =========================
# 测试函数
# =========================
def test_model(model_path="best_ckm_model.pth", file_path="ckm_residual_dataset.npz"):
    """测试模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = ImprovedCKMTransformer(N=64, K=5, L=3).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载测试数据
    dataset = AugmentedCKMResidualDataset(file_path, augment=False)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # 测试
    test_loss = 0
    mu_errors = []
    sigma_errors = []
    power_errors = []

    with torch.no_grad():
        for H, old_ckm, target in test_loader:
            H = H.to(device)
            old_ckm = old_ckm.to(device)
            target = target.to(device)

            pred = model(H, old_ckm)

            # 计算误差
            mu_error = torch.abs(pred[:, :3] - target[:, :3]).cpu().numpy()
            sigma_error = torch.abs(pred[:, 3:6] - target[:, 3:6]).cpu().numpy()
            power_error = torch.abs(pred[:, 6:] - target[:, 6:]).cpu().numpy()

            mu_errors.extend(mu_error.flatten())
            sigma_errors.extend(sigma_error.flatten())
            power_errors.extend(power_error.flatten())

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"μ (delay) MAE: {np.mean(mu_errors):.6f} ± {np.std(mu_errors):.6f}")
    print(f"σ (delay spread) MAE: {np.mean(sigma_errors):.6f} ± {np.std(sigma_errors):.6f}")
    print(f"Power MAE: {np.mean(power_errors):.6f} ± {np.std(power_errors):.6f}")
    print(f"Overall MAE: {np.mean(mu_errors + sigma_errors + power_errors):.6f}")

    return mu_errors, sigma_errors, power_errors


# =========================
# 预测函数（带约束）
# =========================
def predict_ckm(model, H, old_ckm, device='cuda'):
    """
    预测新的CKM参数（带物理约束）

    Args:
        model: 训练好的模型
        H: 信道响应 (K, N) 复数
        old_ckm: 旧的CKM参数 (3L,)
        device: 设备

    Returns:
        new_ckm: 预测的新CKM参数 (3L,)
    """
    model.eval()

    # 准备输入
    H_input = np.stack([H.real, H.imag], axis=-1)  # (K, N, 2)
    H_tensor = torch.tensor(H_input, dtype=torch.float32).unsqueeze(0).to(device)
    old_tensor = torch.tensor(old_ckm, dtype=torch.float32).unsqueeze(0).to(device)

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

    new_ckm_constrained = np.concatenate([mu, sigma, power])

    return new_ckm_constrained


# =========================
# 主函数
# =========================
if __name__ == "__main__":
    # 训练模型
    model, train_losses, val_losses = train_model(
        file_path="ckm_residual_dataset.npz",
        batch_size=64,
        epochs=50,
        learning_rate=1e-3,
        use_cuda=True
    )