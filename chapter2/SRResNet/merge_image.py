from PIL import Image

# 加载图片
image1 = Image.open("AoA_example_draw.png")
image2 = Image.open("CGM_example.png")
# image3 = Image.open("grid2.png")

# 获取每张图片的宽度和高度（假设图片宽度相同）
width1, height1 = image1.size
width2, height2 = image2.size
# width3, height3 = image3.size

# 拼接后的宽度取所有图片的最大宽度，总高度为所有图片高度之和
total_width = width1+width2
total_height = max(height1, height2)

# 创建一个空白的拼接图像
new_image = Image.new('RGB', (total_width, total_height))

# 将三张图片依次粘贴到新图像上
new_image.paste(image1, (0, 0))                      # 粘贴第1张图片
new_image.paste(image2, (width1, 0))                # 粘贴第2张图片（y坐标为第1张图片的高度）

# 保存拼接后的图片
new_image.save("3.png")

# 可选：展示拼接后的图片
new_image.show()
