import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_img_path='images/content.jpg'
style_img_path='images/style.jpg'
out_path='out.jpg'

img_size=512

ckpt_dir="model_state"
alpha=1

