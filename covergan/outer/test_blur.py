from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms.functional import gaussian_blur

im = Image.open("../dataset_emoji_4/clean_covers/demo_track_1.jpg")
convert_tensor = transforms.ToTensor()
tens = convert_tensor(im)
blur = gaussian_blur(tens, kernel_size=29)
to_pil = transforms.ToPILImage()
pil = to_pil(blur)
pil.save('out2.png')
