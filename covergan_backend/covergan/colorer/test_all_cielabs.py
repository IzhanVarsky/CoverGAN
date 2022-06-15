import numpy as np

from colorer.colors_transforms import rgb_to_cielab

arange = np.arange(0, 256, 1)

lab_l = []
lab_a = []
lab_b = []
step = 1
for r in arange:
    for g in arange:
        for b in arange:
            print(r, g, b)
            cielab = rgb_to_cielab([r, g, b])
            lab_l.append(cielab[0])
            lab_a.append(cielab[1])
            lab_b.append(cielab[2])
sorted_lab_l = sorted(lab_l)
sorted_lab_a = sorted(lab_a)
sorted_lab_b = sorted(lab_b)
print(sorted_lab_l[0], sorted_lab_l[-1])
print(sorted_lab_a[0], sorted_lab_a[-1])
print(sorted_lab_b[0], sorted_lab_b[-1])
# 0.0 99.99998453333127
# -86.1829494051608 98.23532017664644
# -107.86546414496824 94.47731817969378