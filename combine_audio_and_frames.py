"""Merge (generated) audio with corresponding video files."""

from typing import Any, Dict, Optional, Union, Tuple
from fractions import Fraction
import argparse
import random
from pathlib import Path
import json

import av
import torch
from torch import Tensor
import torchaudio
from torchvision.io import read_video
import numpy as np
from tqdm import tqdm


def read_video_to_frames_and_audio_streams(
    fn: str,
    start_pts: Optional[Union[float, Fraction]] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "sec",
    output_format: str = "TCHW",
) -> Tuple[Tensor, Tensor, dict]:
    """Read video file to frames and audio streams.

    Args:
        fn (str): Path to video file.
        start_pts (Optional[Union[float, Fraction]], optional): Where to start reading video from. Defaults to 0.
        end_pts (Optional[Union[float, Fraction]], optional): Where to end reading video to. Defaults to None.
        pts_unit (str, optional): Unit of measurement. Defaults to "sec".
        output_format (str, optional): Output dim order. Defaults to "TCHW".

    Raises:
        FileNotFoundError: If video does not exist.

    Returns:
        (Tensor, Tensor, dict): Frames (Tv, C, H, W), audio streams (Ta, ), and metadata.
    """
    if not Path(fn).is_file():
        # warnings.warn(f"File {fn} does not exist.")
        return torch.empty(0), torch.empty(0), {}

    frames, audio, metadata = read_video(
        fn, start_pts, end_pts, pts_unit, output_format
    )
    audio = audio.mean(dim=0)  # (2, T) -> (T,)
    return frames, audio, metadata


def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: Optional[Dict[str, Any]] = None,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = np.round(fps)

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate=audio_fps)
            a_stream.options = audio_options or {}

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = a_stream.format.name

            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy().astype(format_dtype)

            frame = av.AudioFrame.from_ndarray(
                audio_array, format=audio_sample_fmt, layout=audio_layout
            )

            frame.sample_rate = audio_fps

            for packet in a_stream.encode(frame):
                container.mux(packet)

            for packet in a_stream.encode():
                container.mux(packet)

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = 0
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--samples-dir", "-i", type=str, nargs="+", required=True)
    args.add_argument("--video-dir", "-v", type=str, required=True)
    args.add_argument("--output-path", "-o", type=str, nargs="+", required=False)
    args.add_argument("-vfps", type=float, required=False, default=25)
    args.add_argument("-afps", type=int, required=False, default=24000)
    args.add_argument("--num-videos", type=int, required=False, default=None)
    return args


def main():
    args = get_args().parse_args()
    if args.output_path and len(args.output_path) != len(args.samples_dir):
        raise ValueError(
            "Number of output paths should be equal to number of input paths or none."
        )
    input_paths = [Path(dir) for dir in args.samples_dir]
    video_path = Path(args.video_dir)
    output_paths = (
        [Path(path) for path in args.output_path] if args.output_path else input_paths
    )

    for input_path, output_path in zip(input_paths, output_paths):
        audio_samples = input_path.rglob("*.flac")

        if args.num_videos is not None and args.num_videos > 0:
            audio_samples = random.sample(list(audio_samples), args.num_videos)

        for audio in tqdm(
            audio_samples,
            desc=f"Generating videos from {input_path.name}",
            # total=len(list(audio_samples)),
        ):
            start_pts = 0
            video_name = f"{audio.parent.name}/{audio.stem}.mp4"

            audio_tensor, sr = torchaudio.load(audio, channels_first=True)
            duration = audio_tensor.shape[-1] / sr

            frames, _, _ = read_video_to_frames_and_audio_streams(
                (video_path / video_name).as_posix(),
                start_pts=start_pts,
                end_pts=start_pts + duration,
            )

            write_video(
                filename=(output_path / video_name).as_posix(),
                video_array=frames.permute(0, 2, 3, 1),
                audio_array=audio_tensor,
                fps=args.vfps,
                video_codec="h264",
                options={"crf": "10", "pix_fmt": "yuv420p"},
                audio_fps=args.afps,
                audio_codec="aac",
            )


if __name__ == "__main__":
    main()
