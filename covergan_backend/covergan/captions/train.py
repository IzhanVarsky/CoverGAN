import logging

import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.ops.boxes as bops

from tqdm.auto import tqdm

from .models.captioner import Captioner

from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.plotting import plot_losses, plot_grad_flow

logger = logging.getLogger("trainer")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def unstack(t: torch.Tensor, dim: int) -> torch.Tensor:
    # a x (dim * b) -> (a * b) x dim
    width = t.shape[1]
    if width == dim:
        return t

    assert width % dim == 0
    result = []
    for i in range(width // dim):
        result.append(t[:, i * dim: (i + 1) * dim])
    return torch.cat(result)


def calc_iou(a: torch.Tensor, b: torch.Tensor):
    pos_dim = 4
    a = unstack(a, pos_dim)
    b = unstack(b, pos_dim)
    a = bops.box_convert(a, in_fmt='xywh', out_fmt='xyxy')
    b = bops.box_convert(b, in_fmt='xywh', out_fmt='xyxy')
    iou = bops.generalized_box_iou(a, b).diagonal()  # We only need GIoU of corresponding boxes
    return iou


def make_models(canvas_size: int, num_conv_layers: int, num_linear_layers: int,
                device: torch.device) -> Captioner:
    captioner = Captioner(
        canvas_size=canvas_size,
        num_conv_layers=num_conv_layers,
        num_linear_layers=num_linear_layers
    ).to(device)

    return captioner


def train(dataloader: DataLoader, captioner: Captioner, device: torch.device, training_params: dict):
    logger.info(captioner)

    n_epochs = training_params["n_epochs"]
    lr = training_params["lr"]
    checkpoint_root = training_params["checkpoint_root"]
    display_steps = training_params["display_steps"]
    bin_steps = training_params["bin_steps"]
    plot_grad = training_params["plot_grad"]

    opt = torch.optim.Adam(captioner.parameters(), lr=lr, betas=(0.5, 0.999))  # momentum=0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, verbose=True)
    criterion = torch.nn.MSELoss()

    captioner_name = 'captioner'
    epochs_done = load_checkpoint(checkpoint_root, captioner_name, [captioner, opt])
    if epochs_done:
        logger.info(f"Loaded a checkpoint with {epochs_done} epochs done")

    cur_step = 0
    pos_losses, color_losses = [], []
    pos_val_metrics, color_val_losses = [], []

    for epoch in range(epochs_done + 1, epochs_done + n_epochs + 1):
        for cover, edges, pos_truth, color_truth in tqdm(dataloader):
            cover = cover.to(device)
            edges = edges.to(device)
            pos_truth = pos_truth.to(device)
            color_truth = color_truth.to(device)

            opt.zero_grad()

            pos_pred, color_pred = captioner(cover, edges)
            assert len(pos_pred) == len(pos_truth)
            assert len(color_pred) == len(color_truth)

            pos_loss = criterion(pos_pred, pos_truth)
            pos_loss.backward(retain_graph=True)
            color_loss = criterion(color_pred, color_truth)
            color_loss.backward()
            if plot_grad and cur_step % display_steps == 0:
                plot_grad_flow(captioner.named_parameters(), "Captioner")

            opt.step()

            pos_losses.append(pos_loss.item())
            color_losses.append(color_loss.item())
            # plot_losses(epoch, cur_step, display_steps, bin_steps,
            #             [("Positioning", pos_losses), ("Coloring", color_losses)])
            cur_step += 1

        captioner.eval()
        pos_val_metric, color_val_loss = 0.0, 0.0
        for cover, edges, pos_truth, color_truth in tqdm(dataloader):
            cover = cover.to(device)
            edges = edges.to(device)
            pos_truth = pos_truth.to(device)
            color_truth = color_truth.to(device)

            pos_pred, color_pred = captioner(cover, edges)
            batch_iou = calc_iou(pos_pred, pos_truth).mean()
            batch_criterion = criterion(color_pred, color_truth).mean()
            pos_val_metric += batch_iou.item()
            color_val_loss += batch_criterion.item()
        pos_val_metric /= len(dataloader)
        color_val_loss /= len(dataloader)
        pos_val_metrics.append(pos_val_metric)
        color_val_losses.append(color_val_loss)
        plot_losses(epoch, 1, 1, 1, [("Pos IOU Metric", pos_val_metrics)])
        plot_losses(epoch, 1, 1, 1, [("Coloring", color_val_losses)])
        captioner.train()

        scheduler.step(pos_val_metric - color_val_loss)  # max-mode

        save_checkpoint(checkpoint_root, captioner_name, epoch, 0, [captioner, opt])
