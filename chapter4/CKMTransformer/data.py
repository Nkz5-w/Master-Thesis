import numpy as np


def generate_ckm_residual_dataset(
        save_path="ckm_residual_dataset.npz",
        num_samples=50000,
        N=64,
        L=3,
        K=5
):
    """
    生成 CKM 更新数据集（Residual Learning）：
    1️⃣ 生成 old CKM
    2️⃣ 在 old CKM 上加随机扰动生成 new CKM
    3️⃣ 用 new CKM 生成信道 H
    4️⃣ LS估计 H_ls
    5️⃣ 输出 old_ckm 和 new_ckm

    old_ckm: μ/σ/power
    new_ckm: μ/σ/power
    H_ls: LS估计 (K, N)
    """

    df = 15e3
    f = np.arange(N) * df

    H_ls_all = []
    old_ckm_all = []
    new_ckm_all = []

    for _ in range(num_samples):

        # =========================
        # 1️⃣ 生成 old CKM
        # =========================
        mu_old = np.sort(np.random.uniform(0, 3e-6, L))  # μ
        sigma_old = np.random.uniform(0.1e-6, 0.3e-6, L)  # σ
        power_old = np.exp(-np.sort(np.random.uniform(0, 2, L)))  # power

        # =========================
        # 2️⃣ 生成扰动生成 new CKM
        # =========================
        scale = 0.3

        mu_new = mu_old + np.random.randn(L) * scale * 2e-6
        sigma_new = sigma_old + np.random.randn(L) * scale * 0.3e-6
        power_new = power_old * np.exp(np.random.randn(L) * scale * 0.3)

        # 合法化
        mu_new = np.clip(mu_new, 0, None)
        sigma_new = np.clip(sigma_new, 0.05e-6, None)
        power_new = np.clip(power_new, 1e-4, None)

        # 排序
        idx = np.argsort(mu_new)
        mu_new = mu_new[idx]
        sigma_new = sigma_new[idx]
        power_new = power_new[idx]

        # =========================
        # 3️⃣ 用 new CKM 生成信道 H_ls
        # =========================
        H_seq = []

        # 随机 SNR
        SNR_dB = np.random.uniform(0, 30)
        noise_var = 10 ** (-SNR_dB / 10)

        for t in range(K):
            h = np.zeros(N, dtype=np.complex64)

            for l in range(L):
                # 路径扩展 σ 作为随机时延扰动
                tau_sample = mu_new[l] + np.random.randn() * sigma_new[l]

                alpha = np.sqrt(power_new[l] / 2) * (
                        np.random.randn() + 1j * np.random.randn()
                )

                h += alpha * np.exp(-1j * 2 * np.pi * f * tau_sample)

            # 添加噪声
            noise = np.sqrt(noise_var / 2) * (
                    np.random.randn(N) + 1j * np.random.randn(N)
            )

            # LS估计
            h_ls = h + noise
            H_seq.append(h_ls)

        H_seq = np.stack(H_seq, axis=0)  # (K, N)

        # =========================
        # 4️⃣ 保存 CKM
        # =========================
        old_ckm = np.concatenate([mu_old / 1e-6, sigma_old / 1e-6, power_old])
        new_ckm = np.concatenate([mu_new / 1e-6, sigma_new / 1e-6, power_new])

        H_ls_all.append(H_seq)
        old_ckm_all.append(old_ckm)
        new_ckm_all.append(new_ckm)

    # =========================
    # 保存为 npz
    # =========================
    np.savez_compressed(
        save_path,
        H_ls=np.array(H_ls_all),
        old_ckm=np.array(old_ckm_all),
        new_ckm=np.array(new_ckm_all)
    )

    print(f"Dataset saved to {save_path}")

    # sanity check
    data = np.load(save_path)
    print("Keys:", data.files)
    print("H_ls shape:", data["H_ls"].shape)
    print("old_ckm shape:", data["old_ckm"].shape)
    print("new_ckm shape:", data["new_ckm"].shape)


if __name__ == "__main__":
    generate_ckm_residual_dataset()
