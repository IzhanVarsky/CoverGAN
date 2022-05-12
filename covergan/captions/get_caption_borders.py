from typing import Optional

import torch.cuda
from PIL import Image
from torchvision.transforms.functional import to_tensor

from captions.dataset import plot_img_with_bboxes, image_file_to_tensor
from captions.models.captioner import Captioner
from utils.bboxes import BBox

device = "cuda" if torch.cuda.is_available() else "cpu"
captioner_weights_path = "../weights/captioner.pt"

cover_paths = ["0_s_1_g2.png", "0_s_6_g2.png"]
artist_name = ""
track_name = ""


def png_data_to_pil_image(f_name, canvas_size: Optional[int] = None) -> Image:
    result = Image.open(f_name).convert('RGB')
    if canvas_size is not None:
        result = result.resize((canvas_size, canvas_size))
    return result


def main():
    captioner_canvas_size_ = 256
    num_captioner_conv_layers = 3
    num_captioner_linear_layers = 2
    captioner = Captioner(
        canvas_size=captioner_canvas_size_,
        num_conv_layers=num_captioner_conv_layers,
        num_linear_layers=num_captioner_linear_layers
    ).to(device)
    captioner_weights = torch.load(captioner_weights_path, map_location=device)
    captioner.load_state_dict(captioner_weights["0_state_dict"])
    captioner.eval()
    with torch.no_grad():
        covs = []
        for cover_path in cover_paths:
            covs.append(to_tensor(png_data_to_pil_image(cover_path, captioner_canvas_size_)))
        pos_pred, color_pred = captioner(torch.stack(covs))
        pos_preds = torch.round(pos_pred * captioner_canvas_size_).to(int).numpy()
        color_preds = torch.round(color_pred * 255).to(int).numpy()
        for i, cover_path in enumerate(cover_paths):
            x_pos_pred = pos_preds[i]
            x_color_pred = color_preds[i]
            print(x_pos_pred)
            print(x_color_pred)
            bbox1 = BBox(*x_pos_pred[:4])
            col1 = x_color_pred[:3]
            bbox2 = BBox(*x_pos_pred[4:])
            col2 = x_color_pred[3:]
            original_cover = image_file_to_tensor(cover_path, captioner_canvas_size_)
            plot_img_with_bboxes(original_cover, [(bbox1, col1), (bbox2, col2)])


if __name__ == '__main__':
    main()
