import json
import base64
import os

out_dir = "out"
os.makedirs(out_dir, exist_ok=True)
for f_ind, f_name in enumerate(['output1.json',
                                'output2.json',
                                'output3.json',
                                'output4.json']):
    with open(f_name, encoding="utf-8") as f:
        d = json.load(f)
        for ind, x in enumerate(d):
            with open(f"{out_dir}/svg_out_{f_ind}_{ind}.svg", 'w', encoding="utf-8") as f_svg:
                f_svg.write(x["svg"])
            if "base64" in x:
                with open(f"{out_dir}/png_out_{f_ind}_{ind}.png", 'wb') as f_png:
                    f_png.write(base64.b64decode(x["base64"]))
