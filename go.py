import torch
from depthtest import process_depth_images

###
import requests
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
###

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
response = requests.get(url, stream=True)
image = Image.open(response.raw)
transform = transforms.ToTensor()
tensor_image = transform(image).cuda()

to_pil = transforms.ToPILImage()
img_pil = to_pil(tensor_image)
img_pil.save('image.jpg')

print(tensor_image.shape)
depth = process_depth_images(tensor_image, 'vits')


print(depth.shape)
img_pil = to_pil(depth)
img_pil.save('depth.jpg')