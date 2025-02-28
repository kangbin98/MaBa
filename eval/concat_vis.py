import os
from PIL import Image

# 设置文件路径
folder_path = '/home/kb/TBPR/MaBa/vis/p318_s580'

# 获取包含'cls'的图像文件
cls_images = [f for f in os.listdir(folder_path) if 'cls' in f and f.endswith('.png')]

# 按照文件名中的数字排序
cls_images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

# 调整图像尺寸并将其横向拼接
images = []
for image_name in cls_images:
    image_path = os.path.join(folder_path, image_name)
    img = Image.open(image_path)

    # 调整大小为106x280
    img_resized = img.resize((106, 280))

    # 将调整后的图像添加到列表
    images.append(img_resized)

# 设定每张图像之间的空白间距（例如10个像素）
padding = 10

# 计算拼接后图像的宽度
concatenated_width = 106 * len(images) + padding * (len(images) - 1)
concatenated_image = Image.new('RGB', (concatenated_width, 280), (255, 255, 255))  # 设置背景色为白色

# 横向拼接图像并留空白
x_offset = 0
for img in images:
    concatenated_image.paste(img, (x_offset, 0))
    x_offset += img.width + padding  # 添加间隔

# 保存拼接后的图像
output_path = '/home/kb/TBPR/MaBa/vis/concatenated_s580.png'
concatenated_image.save(output_path)

print(f"拼接后的图像已保存为 {output_path}")
