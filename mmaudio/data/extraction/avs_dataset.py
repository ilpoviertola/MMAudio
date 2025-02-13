import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from PIL import Image

from mmaudio.utils.dist_utils import local_rank

log = logging.getLogger()

_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0

_EXPANSION_PIX = 3


class AVS(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        tsv_path: Union[str, Path] = "sets/vgg3-train.tsv",
        sample_rate: int = 44_100,
        duration_sec: float = 5.0,
        audio_samples: Optional[int] = None,
        normalize_audio: bool = False,
    ):
        self.root = Path(root)
        self.mask_root = Path(str(root).replace("raw_videos", "gt_masks"))
        self.normalize_audio = normalize_audio
        if audio_samples is None:
            self.audio_samples = int(sample_rate * duration_sec)
        else:
            self.audio_samples = audio_samples
            effective_duration = audio_samples / sample_rate
            # make sure the duration is close enough, within 16ms
            assert (
                abs(effective_duration - duration_sec) < 0.016
            ), f"audio_samples {audio_samples} does not match duration_sec {duration_sec}"

        videos = sorted(list(self.root.rglob("*.mp4")))
        videos = set([Path(v).stem for v in videos])  # remove extensions
        self.labels = {}
        self.videos = []
        missing_videos = []

        # read the tsv for subset information
        df_list = pd.read_csv(tsv_path, sep="\t", dtype={"id": str}).to_dict("records")
        for record in df_list:
            id = record["id"]
            label = record["label"]
            if id in videos:
                self.labels[id] = label
                self.videos.append(id)
            else:
                missing_videos.append(id)

        if local_rank == 0:
            log.info(f"{len(videos)} videos found in {root}")
            log.info(f"{len(self.videos)} videos found in {tsv_path}")
            log.info(f"{len(missing_videos)} videos missing in {root}")

        self.sample_rate = sample_rate
        self.duration_sec = duration_sec

        self.expected_audio_length = audio_samples
        self.clip_expected_length = int(_CLIP_FPS * self.duration_sec)
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.clip_transform = v2.Compose(
            [
                v2.Resize(
                    (_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.sync_transform = v2.Compose(
            [
                v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop(_SYNC_SIZE),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.to_tensor_transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.mask_video_transform = v2.Compose(
            [v2.UniformTemporalSubsample(self.sync_expected_length)]
        )

        self.resampler = {}

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_id = self.videos[idx]
        label = self.labels[video_id]

        reader = StreamingMediaDecoder(self.root / label / (video_id + ".mp4"))
        reader.add_basic_video_stream(
            frames_per_chunk=int(_CLIP_FPS * self.duration_sec),
            frame_rate=_CLIP_FPS,
            format="rgb24",
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format="rgb24",
        )
        reader.add_basic_audio_stream(
            frames_per_chunk=2**30,
        )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        clip_chunk = data_chunk[0]
        sync_chunk = data_chunk[1]
        audio_chunk = data_chunk[2]

        if clip_chunk is None:
            raise RuntimeError(f"CLIP video returned None {video_id}")
        if clip_chunk.shape[0] < self.clip_expected_length:
            raise RuntimeError(
                f"CLIP video too short {video_id}, expected {self.clip_expected_length}, got {clip_chunk.shape[0]}"
            )

        if sync_chunk is None:
            raise RuntimeError(f"Sync video returned None {video_id}")
        if sync_chunk.shape[0] < self.sync_expected_length:
            raise RuntimeError(
                f"Sync video too short {video_id}, expected {self.sync_expected_length}, got {sync_chunk.shape[0]}"
            )

        # process audio
        sample_rate = int(reader.get_out_stream_info(2).sample_rate)
        audio_chunk = audio_chunk.transpose(0, 1)
        audio_chunk = audio_chunk.mean(dim=0)  # mono
        if self.normalize_audio:
            abs_max = audio_chunk.abs().max()
            audio_chunk = audio_chunk / abs_max * 0.95
            if abs_max <= 1e-6:
                raise RuntimeError(f"Audio is silent {video_id}")

        # resample
        if sample_rate == self.sample_rate:
            audio_chunk = audio_chunk
        else:
            if sample_rate not in self.resampler:
                # https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html#kaiser-best
                self.resampler[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate,
                    self.sample_rate,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                )
            audio_chunk = self.resampler[sample_rate](audio_chunk)

        if audio_chunk.shape[0] < self.expected_audio_length:
            raise RuntimeError(f"Audio too short {video_id}")
        audio_chunk = audio_chunk[: self.expected_audio_length]

        # truncate the video
        clip_chunk = clip_chunk[: self.clip_expected_length]
        if clip_chunk.shape[0] != self.clip_expected_length:
            raise RuntimeError(
                f"CLIP video wrong length {video_id}, "
                f"expected {self.clip_expected_length}, "
                f"got {clip_chunk.shape[0]}"
            )
        clip_chunk = self.clip_transform(clip_chunk)

        sync_chunk = sync_chunk[: self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(
                f"Sync video wrong length {video_id}, "
                f"expected {self.sync_expected_length}, "
                f"got {sync_chunk.shape[0]}"
            )
        sync_chunk = self.sync_transform[0:2](sync_chunk)

        # mask video
        mask_chunk = self.get_extended_masks(self.mask_root / label / video_id)
        mask_chunk = self.mask_video_transform(mask_chunk)
        mask_chunk = mask_chunk[: self.sync_expected_length]
        if mask_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(
                f"Mask video wrong length {video_id}, "
                f"expected {self.sync_expected_length}, "
                f"got {mask_chunk.shape[0]}"
            )
        mask_chunk = self.sync_transform(mask_chunk)

        data = {
            "id": video_id,
            "caption": label,
            "audio": audio_chunk,
            "clip_video": clip_chunk,
            "sync_video": sync_chunk,
            "mask_video": mask_chunk,
        }

        return data

    def get_extended_masks(self, sample_dir: Path) -> torch.Tensor:
        """Get extended masks for the sample.

        Args:
            item (dict): Single sample.

        Returns:
            torch.Tensor: Extended masks.
        """
        mask_dir = sample_dir / "extended_mask"
        if not mask_dir.exists() and not mask_dir.is_file():
            raise FileNotFoundError(f"Extended mask dir not found: {mask_dir}")

        ret = []
        for mask_file in sorted(
            mask_dir.glob("*.png"), key=lambda x: int(x.stem.split("_")[-1])
        ):
            ret.append(self._load_png_to_tensor(mask_file.as_posix(), convert_type="L"))
        mask_video = torch.stack(ret, dim=0)  # (T, C, H, W)
        return mask_video

    def _load_png_to_tensor(
        self, image_path: str, convert_type: str = "L", expand_mask: bool = False
    ) -> torch.Tensor:
        image = Image.open(image_path).convert(convert_type)
        image_t = self.to_tensor_transform(image)
        if expand_mask:
            # Convert to binary mask
            binary_mask = image_t > 0.5
            if not torch.any(binary_mask):
                return image_t
            if binary_mask.ndim == 3:
                binary_mask = binary_mask.squeeze(0)
            # Find bounding box
            non_zero_indices = torch.nonzero(binary_mask)
            min_y, min_x = torch.min(non_zero_indices, dim=0)[0]
            max_y, max_x = torch.max(non_zero_indices, dim=0)[0]
            # Expand bounding box by _EXPANSION_PIX pixels
            min_y = max(min_y - _EXPANSION_PIX, 0)
            min_x = max(min_x - _EXPANSION_PIX, 0)
            max_y = min(max_y + _EXPANSION_PIX, image_t.shape[1] - 1)
            max_x = min(max_x + _EXPANSION_PIX, image_t.shape[2] - 1)
            # Create new mask with expanded bounding box
            image_t = torch.zeros_like(image_t)
            image_t[:, min_y : max_y + 1, min_x : max_x + 1] = 1

        return image_t

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f"Error loading video {self.videos[idx]}: {e}")
            return None

    def __len__(self):
        return len(self.labels)
