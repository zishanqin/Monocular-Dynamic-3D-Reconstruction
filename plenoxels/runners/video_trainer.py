import logging as log
import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Any, List
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from PIL import Image

from plenoxels.datasets.video_datasets import Video360Dataset
from plenoxels.utils.ema import EMA
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_video_to_file
from plenoxels.models.lowrank_model import LowrankModel
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model
from .regularization import (
    PlaneTV, TimeSmoothness, HistogramLoss, L1TimePlanes, DistortionLoss
)

def save_images_to_folder(image_array, folder_path):
    for i, image in enumerate(image_array):
        image_path = os.path.join(folder_path, f"{i}.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = Image.fromarray(image)
        image.save(image_path)

class VideoTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 trdset_save: torch.utils.data.TensorDataset,
                 tr_dset: torch.utils.data.TensorDataset,
                 ts_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 depth_gt_dir: str,
                 val_indices: list,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 isg_step: int,
                 ist_step: int,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.train_dataset = tr_dset
        self.train_dataset_original = trdset_save
        self.test_dataset = ts_dset
        self.ist_step = ist_step
        self.isg_step = isg_step
        self.save_video = save_outputs
        # Switch to compute extra video metrics (FLIP, JOD)
        self.compute_video_metrics = False
        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            depth_gt_dir=depth_gt_dir,
            val_indices=val_indices,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=False,  # False since we're saving video
            device=device,
            **kwargs)

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]

            timestamp = data["timestamps"]
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                timestamps_d_b = timestamp.expand(rays_o_b.shape[0]).to(self.device)
                outputs = self.model(
                    rays_o_b, rays_d_b, timestamps=timestamps_d_b, bg_color=bg_color,
                    near_far=near_far)
                for k, v in outputs.items():
                    if "rgb" in k or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        scale_ok = super().train_step(data, **kwargs)

        if self.global_step == self.isg_step:
            self.train_dataset.enable_isg()
            raise StopIteration  # Whenever we change the dataset
        if self.global_step == self.ist_step:
            self.train_dataset.switch_isg2ist()
            raise StopIteration  # Whenever we change the dataset

        return scale_ok

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        # Reset randomness in train-dataset
        self.train_dataset.reset_iter()


    def save_renders(self, dataset, vis=None): 
        # print('Dataset size', len(dataset))
        per_scene_metrics: Dict[str, Union[float, List]] = defaultdict(list)
        pred_frames, out_depths = [], []
        pb = tqdm(total=len(dataset), desc=f"Validate scene ({dataset.name})")
        for img_idx, data in enumerate(dataset):
            preds = self.eval_step(data)
            out_metrics, out_img, out_depth = self.evaluate_metrics(
                data["imgs"], preds, dset=dataset, img_idx=img_idx, name=None,
                save_outputs=self.save_outputs, train=vis)
            pred_frames.append(out_img)
            if out_depth is not None:
                out_depths.append(out_depth)

            # print('out_depth_shape', out_depth)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        pb.close()
        if self.save_video:
            import cv2

            out_name = f"step{self.global_step}"

            # Save pred_frames as separate images
            pred_frames_folder = os.path.join(self.log_dir, "pred_frames")
            os.makedirs(pred_frames_folder, exist_ok=True)
            for i, frame in enumerate(pred_frames):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_path = os.path.join(pred_frames_folder, f"{out_name}_frame{i}.png")
                cv2.imwrite(frame_path, frame)

            # Save out_depths as separate images
            if len(out_depths) > 0:
                depths_folder = os.path.join(self.log_dir, "depths")
                os.makedirs(depths_folder, exist_ok=True)
                for i, depth in enumerate(out_depths):
                    depth_path = os.path.join(depths_folder, f"{out_name}_depth{i}.png")
                    cv2.imwrite(depth_path, depth)

        if not vis:
            val_metrics = [
                self.report_test_metrics(per_scene_metrics, extra_name=None),
            ]
            df = pd.DataFrame.from_records(val_metrics)
            df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))


    @torch.no_grad()
    def validate(self, show_training_renders=False):
        if show_training_renders:
            self.save_renders(self.train_dataset_original, vis=True)
        self.save_renders(self.test_dataset, vis=False)


    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def load_model(self, checkpoint_data, training_needed: bool = True):
        super().load_model(checkpoint_data, training_needed)
        if self.train_dataset is not None:
            if -1 < self.isg_step < self.global_step < self.ist_step:
                self.train_dataset.enable_isg()
            elif -1 < self.ist_step < self.global_step:
                self.train_dataset.switch_isg2ist()

    def init_epoch_info(self):
        ema_weight = 0.9
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)

    def get_regularizers(self, **kwargs):
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            L1TimePlanes(kwargs.get('l1_time_planes', 0.0), what='field'),
            L1TimePlanes(kwargs.get('l1_time_planes_proposal_net', 0.0), what='proposal_network'),
            TimeSmoothness(kwargs.get('time_smoothness_weight', 0.0), what='field'),
            TimeSmoothness(kwargs.get('time_smoothness_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
        ]

    @property
    def calc_metrics_every(self):
        return 5


def init_tr_data(data_downsample, data_dir, truncation, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    log.info(f"Loading Video360Dataset with downsample={data_downsample}")
    tr_dset = Video360Dataset(
        data_dir, split='train', downsample=data_downsample, truncation=truncation,
        batch_size=batch_size,
        max_cameras=kwargs.get('max_train_cameras', None),
        max_tsteps=kwargs['max_train_tsteps'] if keyframes else None,
        isg=isg, keyframes=keyframes, contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],depth_gt_dir=kwargs.get("depth_gt_dir"), val_indices=kwargs.get("val_indices")
    )
    if ist:
        tr_dset.switch_isg2ist()  # this should only happen in case we're reloading


    g = torch.Generator()
    g.manual_seed(0)
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None, num_workers=0,  prefetch_factor=None, pin_memory=True,
        worker_init_fn=init_dloader_random, generator=g)
    return {"tr_loader": tr_loader, "tr_dset": tr_dset}


def init_ts_data(data_dir, split, truncation, **kwargs):
    downsample = 1.0 # Both D-NeRF and DyNeRF use downsampling by 2
    # batch=4096 if split == 'train' else None
    dset = Video360Dataset(
        data_dir, split=split, downsample=downsample, truncation=truncation,
        max_cameras=kwargs.get('max_test_cameras', None), max_tsteps=kwargs.get('max_test_tsteps', None),
        contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'], depth_gt_dir=kwargs.get("depth_gt_dir"), val_indices=kwargs.get("val_indices")
    )
    # return {"ts_dset": ts_dset}
    if split == 'train_vis':
        key = 'trdset_save'
    else:
        key = 'ts_dset'

    # exit()
    return {key: dset}


def load_data(data_downsample, data_dirs, validate_only, render_only, truncation, **kwargs):
    assert len(data_dirs) == 1
    od: Dict[str, Any] = {}
    if not validate_only and not render_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], truncation=truncation, **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if render_only else 'test'
    od.update(init_ts_data(data_dirs[0], truncation=truncation, split=test_split, **kwargs))
    # Use the same code to initilize the training data
    od.update(init_ts_data(data_dirs[0], truncation=truncation, split='train_vis', **kwargs))
    print("video 370 od", od.keys())
    return od