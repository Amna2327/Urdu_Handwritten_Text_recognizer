from PIL import Image

for i in range(1,6):
    img=Image.open(f'data/{i}.png')
    print(img.size)
    print(img.mode)
    img.show()