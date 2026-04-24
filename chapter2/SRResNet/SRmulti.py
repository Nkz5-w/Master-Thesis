import torch
import numpy as np
from PIL import Image
from utils import convert_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== scale列表 =====
#scales = [2, 4, 8, 16]
scales = [2, 4, 8]

# ===== 加载模型 =====
srresnet_models = {}
for s in scales:
    ckpt_path = f'./checkpoints/checkpoint_srresnet_{s}_aoa.pth.tar'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = ckpt['model'].to(device)
    model.eval()
    srresnet_models[s] = model

print("Models loaded!")


# =========================
# 稀疏上采样
# =========================
def sparse_upsample(lr_img, scale):
    lr_np = np.array(lr_img)
    h, w = lr_np.shape

    hr_np = np.ones((h * scale, w * scale), dtype=np.uint8) * 255

    # ===== 根据scale决定点大小 =====
    if scale == 4:
        point_size = 2
    elif scale == 2:
        point_size = 1
    elif scale == 8:
        point_size = 4
    else:  # 16
        point_size = 8

    # ===== 放置点 =====
    for i in range(h):
        for j in range(w):
            val = lr_np[i, j]

            x = i * scale
            y = j * scale

            hr_np[
                x:x + point_size,
                y:y + point_size
            ] = val

    return Image.fromarray(hr_np)


# =========================
# 主函数
# =========================
def visualize_multi_scale(img_path):

    hr_img = Image.open(img_path).convert("L")

    sparse_imgs = []
    nearest_imgs = []
    sr_imgs = []

    # ===== 每个scale =====
    for s in scales:

        # LR
        lr_img = hr_img.resize(
            (hr_img.width // s, hr_img.height // s),
            Image.BICUBIC
        )

        # ===== sparse =====
        sparse = sparse_upsample(lr_img, s)
        sparse_imgs.append(sparse)

        # ===== nearest =====
        nearest = lr_img.resize(
            (hr_img.width, hr_img.height),
            Image.NEAREST
        )
        nearest_imgs.append(nearest)

        # ===== SRResNet =====
        with torch.no_grad():
            sr = srresnet_models[s](
                convert_image(lr_img, source='pil', target='imagenet-norm')
                .unsqueeze(0).to(device)
            )

        sr = sr.squeeze(0).cpu().detach()
        sr_img = convert_image(sr, source='[-1, 1]', target='pil').convert("L")
        sr_imgs.append(sr_img)

    # =========================
    # 拼图（3行）
    # =========================
    margin = 20
    W, H = hr_img.width, hr_img.height

    grid = Image.new(
        'RGB',
        (4 * W + 5 * margin, 3 * H + 4 * margin),
        (255, 255, 255)
    )

    # ===== 第1行：sparse =====
    for i in range(3):
        grid.paste(
            sparse_imgs[i].convert("RGB"),
            (i * W + (i + 1) * margin, margin)
        )

    # ===== 第2行：nearest =====
    for i in range(3):
        grid.paste(
            nearest_imgs[i].convert("RGB"),
            (i * W + (i + 1) * margin, H + 2 * margin)
        )

    # ===== 第3行：SRResNet =====
    for i in range(3):
        grid.paste(
            sr_imgs[i].convert("RGB"),
            (i * W + (i + 1) * margin, 2 * H + 3 * margin)
        )

    # =========================
    # 高清放大（论文用）
    # =========================
    grid = grid.resize(
        (grid.width * 4, grid.height * 4),
        Image.NEAREST
    )

    grid.save("grid_multi_scale.png", dpi=(300, 300))
    print("Saved: grid_multi_scale.png")


# =========================
if __name__ == "__main__":
    visualize_multi_scale('datasets/train_ckm_aoa/NJ6_128_BS6_4_0_AoA.png')