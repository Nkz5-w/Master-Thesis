import os
import shutil


def merge_images(source_folder, target_folder):
    # 创建目标文件夹，如果不存在的话
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 检查文件扩展名是否是图片格式
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # 构建完整的源文件路径
                source_path = os.path.join(root, file)
                # 构建目标文件路径
                target_path = os.path.join(target_folder, file)
                # 复制文件到目标文件夹
                shutil.copy2(source_path, target_path)


# 使用示例
source_folder = 'D:/wzj/image/RGB'  # 源文件夹路径
target_folder = 'D:/wsy/Codes/SR/datasets/train_ckm_RGB'  # 目标文件夹路径
merge_images(source_folder, target_folder)
