import os
import random
from PIL import Image, ImageDraw, ImageFont

# 設定字型資料夾路徑
font_folder = 'fonts'
font_files = ['HanyiSentyTang.ttf', 'SentyZHAO.ttf', 'SNsanafonGyou.ttf', '行書字體.ttf', '潦草字體.ttf']

# 設定生成圖像的參數
image_size = (64, 64)
text_color = (0, 0, 0)  # 黑色
output_folder = 'generated_images'
os.makedirs(output_folder, exist_ok=True)

# 從 common_characters.txt 讀取
with open('common_characters.txt', 'r', encoding='utf-8') as f:
    characters = f.read().strip().splitlines()

# 隨機生成圖像的函數
def generate_images(character, font_path):
    for i in range(5):
        image = Image.new('RGB', image_size, (255, 255, 255))  # 白色背景
        draw = ImageDraw.Draw(image)
        
        # 載入字型
        font_size = random.randint(30, 50)
        font = ImageFont.truetype(font_path, font_size)

        # 計算文本位置以居中
        text_width, text_height = draw.textsize(character, font=font)
        position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
        
        # 在圖像上繪製文本
        draw.text(position, character, text=text_color, font=font)

        # 保存圖像
        image_name = f"{character}_{os.path.basename(font_path).split('.')[0]}_{i + 1}.png"
        image.save(os.path.join(output_folder, image_name))

# 生成所有字符的圖像
for character in characters:
    for font_file in font_files:
        font_path = os.path.join(font_folder, font_file)
        generate_images(character, font_path)

print("generate success")
