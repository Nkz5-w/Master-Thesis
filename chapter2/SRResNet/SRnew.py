import torch
import numpy as np
from PIL import Image

# =========================
# 参数
# =========================
scaling_factor = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

srgan_checkpoint = "./checkpoints/checkpoint_srgan_4_single_ckm_pathloss.pth.tar"
srresnet_checkpoint = "./checkpoints/checkpoint_srresnet_4_single_ckm_pathloss.pth.tar"

srresnet = torch.load(srresnet_checkpoint, map_location=torch.device('cpu'))['model'].to(device)
srresnet.eval()
srgan_generator = torch.load(srgan_checkpoint, map_location=torch.device('cpu'))['generator'].to(device)
srgan_generator.eval()


# =========================
# 稀疏上采样（灰度）
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
# PIL灰度 → tensor (1,1,H,W)
# =========================
def img_to_tensor(img):
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)


# =========================
# tensor → PIL灰度
# =========================
def tensor_to_img(tensor):
    img = tensor.squeeze().cpu().detach().numpy()
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


# =========================
# 可视化函数
# =========================
def visualize_sr(img_path):
    # ===== 1. 读取HR（灰度）=====
    hr_img = Image.open(img_path).convert("L")

    # ===== 2. 生成LR =====
    lr_img = hr_img.resize(
        (hr_img.width // scaling_factor, hr_img.height // scaling_factor),
        Image.BICUBIC
    )

    # ===== 3. baseline =====
    sparse_lr_img = sparse_upsample(lr_img, scaling_factor)
    nearest_img = lr_img.resize((hr_img.width, hr_img.height), Image.NEAREST)
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    # ===== 4. 转tensor =====
    lr_tensor = img_to_tensor(lr_img).to(device)

    # ===== 5. SRResNet =====
    with torch.no_grad():
        sr_resnet = srresnet(lr_tensor)

    sr_img_srresnet = tensor_to_img(sr_resnet)

    # ===== 6. SRGAN =====
    with torch.no_grad():
        sr_gan = srgan_generator(lr_tensor)

    sr_img_srgan = tensor_to_img(sr_gan)

    # ===== 7. 拼图 =====
    margin = 20
    grid_img = Image.new(
        'RGB',
        (6 * hr_img.width + 7 * margin, hr_img.height + 2 * margin),
        (255, 255, 255)
    )

    # 转RGB再贴（避免模式冲突）
    grid_img.paste(sparse_lr_img.convert("RGB"), (margin, margin))
    grid_img.paste(nearest_img.convert("RGB"), (hr_img.width + 2 * margin, margin))
    grid_img.paste(bicubic_img.convert("RGB"), (2 * hr_img.width + 3 * margin, margin))
    grid_img.paste(sr_img_srgan.convert("RGB"), (3 * hr_img.width + 4 * margin, margin))
    grid_img.paste(sr_img_srresnet.convert("RGB"), (4 * hr_img.width + 5 * margin, margin))
    grid_img.paste(hr_img.convert("RGB"), (5 * hr_img.width + 6 * margin, margin))

    # ===== 8. 放大（关键，论文用）=====
    scale_up = 4
    grid_img = grid_img.resize(
        (grid_img.width * scale_up, grid_img.height * scale_up),
        Image.NEAREST
    )

    # ===== 9. 保存高清 =====
    grid_img.save('grid.png', dpi=(300, 300))

    print("Saved: grid.png")


# =========================
# 主函数
# =========================
if __name__ == '__main__':
    visualize_sr('datasets/train_ckm_pathloss/Suzhou_1_128_BS5_6_2.png')
