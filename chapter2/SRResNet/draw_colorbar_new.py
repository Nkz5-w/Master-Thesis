from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt

# ===================== 参数 =====================
scale = 4          # 分辨率放大倍数（推荐2~4）
num_labels = 6     # 色条刻度数量

# ===================== 读取并放大图像 =====================
pil_image = Image.open('datasets/test_ckm_aoa/BJ_128_BS5_1_1_AoA.png').convert('L')

pil_image = pil_image.resize(
    (pil_image.width * scale, pil_image.height * scale),
    Image.Resampling.NEAREST
)

gray_image = np.array(pil_image)

# ===================== 灰度范围 =====================
min_pixel = gray_image.min()
max_pixel = gray_image.max()

# ⭐ 映射到实际数值（但这里只是数值，不显示单位）
min_val = min_pixel * 380 / 255 - 200
max_val = max_pixel * 380 / 255 - 200

# ===================== 构建灰度 colorbar =====================
color_bar_height = gray_image.shape[0]
color_bar_width = 20 * scale

# 用原始灰度范围生成色条（保证一致）
color_bar = np.linspace(min_pixel, max_pixel, color_bar_height).reshape(-1, 1)
color_bar = np.flipud(color_bar)

# 归一化到0~1
color_bar_norm = color_bar / 255

# ⭐ 使用灰度 colormap（关键修复点）
cmap = plt.get_cmap('gray')
color_bar_rgb = cmap(color_bar_norm)[:, :, :3]

# 扩展宽度
color_bar_rgb = np.tile(color_bar_rgb, (1, color_bar_width, 1))

color_bar_pil = Image.fromarray((color_bar_rgb * 255).astype(np.uint8))

# ===================== 拼接 =====================
spacing_width = 10 * scale
extra_space_width = 70 * scale

spacing = np.ones((color_bar_height, spacing_width, 3)) * 255
spacing_pil = Image.fromarray(spacing.astype(np.uint8))

gray_image_rgb = np.stack([gray_image] * 3, axis=-1)
gray_image_pil = Image.fromarray(gray_image_rgb)

combined_width = (
    gray_image_pil.width +
    spacing_pil.width +
    color_bar_pil.width +
    extra_space_width
)

combined_height = gray_image_pil.height + 20 * scale

combined_image_pil = Image.new(
    'RGB',
    (combined_width, combined_height),
    color=(255, 255, 255)
)

# 粘贴
combined_image_pil.paste(gray_image_pil, (0, 10 * scale))
combined_image_pil.paste(spacing_pil, (gray_image_pil.width, 0))
combined_image_pil.paste(color_bar_pil, (gray_image_pil.width + spacing_width, 10 * scale))

# ===================== 绘制刻度 =====================
draw = ImageDraw.Draw(combined_image_pil)

# 字体
try:
    font = ImageFont.truetype("times.ttf", 14 * scale)
except IOError:
    font = ImageFont.load_default()

# ⭐ 仅数值（无单位）
for value in np.linspace(min_val, max_val, num_labels):
    y = int((1 - (value - min_val) / (max_val - min_val)) * (color_bar_height - 1))
    y += 10 * scale
    y -= 7 * scale

    text_position = (
        gray_image_pil.width + spacing_width + color_bar_width + 10 * scale,
        y
    )

    draw.text(
        text_position,
        f'{value:.0f}°',
        fill='black',
        font=font
    )

# ===================== 保存 =====================
combined_image_pil.save(
    'example_draw.png',
    dpi=(600, 600),   # 论文级
    quality=95
)

print("✅ 已生成论文级灰度图：example_draw.png")