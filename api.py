from transformers import pipeline
from PIL import Image
import requests
import torchvision.transforms as transforms


# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
to_pil = transforms.ToPILImage()
depth = pipe(image)['predicted_depth']
img_pil = to_pil(depth)
img_pil.save('api_depth.jpg')
print(depth.shape)
