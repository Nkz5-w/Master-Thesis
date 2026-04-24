from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt

# 打开你的PIL图片
pil_image = Image.open('datasets/train_ckm_aoa/BJ_128_BS1_0_3_AoA.png').convert('L')  # 确保是灰度图像
gray_image = np.array(pil_image)  # 转换为NumPy数组

# 获取图像的最小和最大灰度值（假设范围是0到255）
min_val = gray_image.min()
max_val = gray_image.max()

# 创建色卡（垂直方向），色卡的范围从0到255
color_bar_height = gray_image.shape[0]
color_bar_width = 10  # 色卡的宽度
color_bar = np.linspace(min_val, max_val, color_bar_height).reshape(-1, 1)

# 将色卡上下颠倒
color_bar = np.flipud(color_bar)

# 将色卡标准化到0-1范围，以便使用灰度颜色映射
color_bar = color_bar / 255
cmap = plt.get_cmap('gray')
color_bar_rgb = cmap(color_bar)[:, :, :3]  # 转换为RGB
color_bar_rgb = np.tile(color_bar_rgb, (1, color_bar_width, 1))  # 扩展色卡宽度

# # path loss
# min_val = min_val * 200 / 255 - 250
# max_val = max_val * 200 / 255 - 250
# 角度
min_val = min_val * 380 / 255 - 200
max_val = max_val * 380 / 255 - 200

# 将色卡转换为PIL图像
color_bar_pil = Image.fromarray((color_bar_rgb * 255).astype(np.uint8))

# 创建一个间隔区域
spacing_width = 10  # 间隔宽度（可以调整）
spacing = np.ones((color_bar_height, spacing_width, 3)) * 255  # 创建白色间隔区域
spacing_pil = Image.fromarray(spacing.astype(np.uint8))

# 将灰度图像转换为RGB图像
gray_image_rgb = np.stack([gray_image] * 3, axis=-1)
gray_image_pil = Image.fromarray(gray_image_rgb)

# 增加图像宽度，给数字留出空间，并设置背景颜色为白色
extra_space_width = 50  # 增加的额外宽度，用于放置刻度数字
combined_image_pil = Image.new('RGB',
                               (gray_image_pil.width + spacing_pil.width + color_bar_pil.width + extra_space_width, gray_image_pil.height+10),
                               color=(255, 255, 255))  # 设置背景颜色为白色

# 拼接灰度图像、间隔和色卡
combined_image_pil.paste(gray_image_pil, (0, 5))
combined_image_pil.paste(spacing_pil, (gray_image_pil.width, 0))
combined_image_pil.paste(color_bar_pil, (gray_image_pil.width + spacing_pil.width, 5))

# 使用 ImageDraw.Draw 绘制文本
draw = ImageDraw.Draw(combined_image_pil)

# 设置字体大小（如果没有字体文件，可以使用默认字体）
try:
    font = ImageFont.truetype("times.ttf", 11)  # 你可以选择字体和大小
except IOError:
    font = ImageFont.load_default()  # 如果找不到字体文件，加载默认字体

# 添加数值标记，确保数值与色卡上的颜色对齐
num_labels = 6  # 例如，从0到255的范围内添加6个标记
for i, value in enumerate(np.linspace(min_val, max_val, num_labels)):  # 根据图像的灰度范围生成标记
    # 调整y位置确保文本在图片范围内
    y = int((1 - (value - min_val) / (max_val - min_val)) * (color_bar_height - 1))  # 均匀分布刻度
    text_position = (gray_image_pil.width + color_bar_width + spacing_width + 10, y)  # 调整x, y 位置
    # 角度
    draw.text(text_position, f'{value:.0f}\u00b0', fill='black', font=font)


# 保存并显示最终的图像
combined_image_pil.save('example_draw.png')
combined_image_pil.show()

combined_image_pil.show()

