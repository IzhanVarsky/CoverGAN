from colorthief import ColorThief

jpg_filename = 'A S T R O - Change.jpg'
color_thief = ColorThief(jpg_filename)
# get the dominant color
dominant_color = color_thief.get_color(quality=1)
palette = color_thief.get_palette(color_count=6)
print(palette)
# [(190, 244, 159), (61, 112, 115), (100, 99, 103), (114, 170, 157), (196, 244, 217), (94, 132, 84)]

