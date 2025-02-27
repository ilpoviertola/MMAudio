from pathlib import Path
import typing as tp
from argparse import ArgumentParser
from random import sample

import torch
import torchvision.transforms.v2 as Tv
from torchvision.io import decode_image, ImageReadMode
from torio.io import StreamingMediaDecoder, StreamingMediaEncoder

from mmaudio.utils.video_joiner import merge_audio_into_video
import multiprocessing as mp
from tqdm import tqdm


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input video dir",
    )
    parser.add_argument(
        "-m",
        "--mask_dir",
        type=Path,
        required=True,
        help="Mask dir",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output video dir",
    )
    parser.add_argument(
        "-c",
        "--count_per_file",
        type=int,
        default=3,
        help="Amount of new files per input file",
    )
    return parser


def get_video_transforms(width: int, height: int, num_frames: int) -> Tv.Compose:
    return Tv.Compose(
        [Tv.Resize((height, width)), Tv.UniformTemporalSubsample(num_frames)]
    )


def sample_list(lst: tp.List[tp.Any], count: int) -> tp.List[tp.Any]:
    return sample(lst, count)


def convert_to_box_masks(masks: torch.Tensor, enlarge_by: int = 5) -> torch.Tensor:
    """
    Convert strict masks to box masks and enlarge the masked area.

    Args:
        masks (torch.Tensor): Tensor of binary masks with shape (T, C, H, W).
        enlarge_by (int): Number of pixels to enlarge the masked area by.

    Returns:
        torch.Tensor: Tensor of box masks with enlarged masked area.
    """
    T, C, H, W = masks.shape
    box_masks = torch.zeros_like(masks)

    for t in range(T):
        for c in range(C):
            mask = masks[t, c]
            if mask.sum() == 0:
                continue

            # Find the corner coordinates of the masked area
            coords = torch.nonzero(mask)
            y_min, x_min = coords.min(dim=0)[0]
            y_max, x_max = coords.max(dim=0)[0]

            # Enlarge the masked area
            y_min = max(y_min - enlarge_by, 0)
            x_min = max(x_min - enlarge_by, 0)
            y_max = min(y_max + enlarge_by, H - 1)
            x_max = min(x_max + enlarge_by, W - 1)

            # Set the masked area to 1
            box_masks[t, c, y_min : y_max + 1, x_min : x_max + 1] = 1

    return box_masks


def extract_object_from_video(
    video_chunk: torch.Tensor,
    mask_path: Path,
    video_transforms: Tv.Compose,
    bb_masks: bool = True,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    masks = sorted(
        list(mask_path.glob("*.png")), key=lambda x: int(x.stem.split("_")[-1])
    )
    assert len(masks) == video_chunk.shape[0], f"{len(masks)} != {video_chunk.shape[0]}"
    masks_l: tp.List[torch.Tensor] = []
    for m in masks:
        mask = decode_image(m, mode=ImageReadMode.GRAY)
        masks_l.append(mask)
    masks_t = torch.stack(masks_l)
    # convert to binary mask
    masks_t = masks_t > 128
    masks_t = video_transforms(masks_t)
    if bb_masks:
        masks_t = convert_to_box_masks(masks_t, 2)
    # get inverse of the mask
    masks_t_inv = ~masks_t
    return video_chunk * masks_t, masks_t_inv


def handle_single_file(
    video: Path, output_dir: Path, mask_dir: Path, count_per_file: int
):
    background_videos = list(video.parent.glob("*.mp4"))
    background_videos.remove(video)
    assert len(background_videos) > 0, f"No background videos found for {video}"
    assert (
        len(background_videos) >= count_per_file
    ), f"Not enough background videos found for {video}"
    mask_path = (
        mask_dir
        / video.parent.parent.name
        / video.parent.name
        / video.stem
        / "extended_mask"
    )
    background_videos = sample_list(background_videos, count_per_file)
    reader_video = StreamingMediaDecoder(video)
    video_info = reader_video.get_src_stream_info(0)
    audio_info = reader_video.get_src_stream_info(1)
    duration = video_info.num_frames / video_info.frame_rate
    reader_video.add_basic_video_stream(
        frames_per_chunk=video_info.num_frames,
        format="rgb24",
    )
    reader_video.add_basic_audio_stream(
        frames_per_chunk=round(duration * audio_info.sample_rate)
    )
    reader_video.fill_buffer()
    video_transforms = get_video_transforms(
        video_info.width, video_info.height, video_info.num_frames
    )
    data_chunk = reader_video.pop_chunks()
    video_chunk = data_chunk[0]
    audio_chunk = data_chunk[1]
    video_chunk, inv_mask = extract_object_from_video(
        video_chunk, mask_path, video_transforms
    )

    for i in range(count_per_file):
        reader_background = StreamingMediaDecoder(background_videos[i])
        background_video_info = reader_background.get_src_stream_info(0)
        reader_background.add_basic_video_stream(
            frames_per_chunk=background_video_info.num_frames,
            format="rgb24",
        )
        reader_background.fill_buffer()
        background_chunk = reader_background.pop_chunks()[0]
        background_chunk = video_transforms(background_chunk)
        background_chunk = background_chunk * inv_mask
        background_chunk = background_chunk + video_chunk
        output_name = f"{video.stem}_{background_videos[i].stem}.mp4"
        output_path = (
            output_dir / video.parent.parent.name / video.parent.name / output_name
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = StreamingMediaEncoder(output_path)
        writer.add_video_stream(
            frame_rate=video_info.frame_rate,
            width=video_info.width,
            height=video_info.height,
            format="rgb24",
            encoder="libx264",
            encoder_format="yuv420p",
        )
        writer.add_audio_stream(
            sample_rate=round(audio_info.sample_rate),
            num_channels=audio_chunk.shape[-1],
            encoder="aac",
        )
        with writer.open():
            writer.write_video_chunk(0, background_chunk)
            writer.write_audio_chunk(1, audio_chunk)


def generate_data(
    input_dir: Path, output_dir: Path, mask_dir: Path, count_per_file: int
):
    assert input_dir.exists(), f"Input dir {input_dir} does not exist"
    assert mask_dir.exists(), f"Mask dir {mask_dir} does not exist"
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = list(input_dir.rglob("*.mp4"))
    with mp.Pool(processes=16) as pool:
        for _ in tqdm(
            pool.starmap(
                handle_single_file,
                [(video, output_dir, mask_dir, count_per_file) for video in videos],
            ),
            total=len(videos),
            position=0,
            leave=True,
        ):
            pass


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    generate_data(args.input, args.output, args.mask_dir, args.count_per_file)
