from PIL import Image, ImageDraw

palettes = [
    [(185.83932795535875, 245.76842360647782, 158.07709104142657),
     (65.17575993764615, 65.35268901013251, 69.9871395167576),
     (208.78896882494004, 247.26139088729016, 192.2170263788969),
     (60.50174825174825, 184.7937062937063, 186.37179487179486),
     (242.23333333333332, 211.1, 184.26666666666668),
     (213.14456391875746, 250.02867383512546, 195.69534050179212)],
    [(190, 244, 159), (61, 112, 115), (100, 99, 103),
     (114, 170, 157), (196, 244, 217), (94, 132, 84)]
]
imsize = 1024
for palette in palettes:
    palette = [tuple(map(int, c)) for c in palette]
    im = Image.new('RGB', (imsize, imsize))
    draw = ImageDraw.Draw(im)
    cur_h = 0
    width = imsize / len(palette)
    for i, p in enumerate(palette):
        print(i, p)
        colorval = "#%02x%02x%02x" % p
        draw.rectangle((0, cur_h, imsize, cur_h + width), fill=colorval)
        cur_h += width
    im.show()
