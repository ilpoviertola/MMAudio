import argparse
import json
from pathlib import Path
import csv
from copy import deepcopy
import random

import cv2
import torch
import einops
import numpy as np
from pycocotools import mask as maskUtils

import sys

sys.path.append(".")
sys.path.append("./scripts/data_prep/track_sam")
sys.path.append("./scripts/data_prep/track_sam/aot")
sys.path.append(str(Path(sys.executable).parent))
from scripts.data_prep.track_sam.SegTracker import SegTracker
from scripts.data_prep.track_sam.model_args import (
    aot_args,
    sam_args,
    segtracker_args,
    detector_args,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--dataset_root")
    parser.add_argument("--audioset_split")
    args = parser.parse_args()

    ### Paths
    dataset_root = Path(args.dataset_root)
    dataset_name = args.dataset

    if dataset_name == "audioset":
        split = args.audioset_split
        split2subfolder = {
            "test": "eval_segments",
            "train": "unbalanced_train_segments",
            "valid": "balanced_train_segments",
        }
        video_root = (
            dataset_root
            / "h264_video_25fps_256side_16000hz_aac"
            / split2subfolder[split]
        )
        if split == "test":
            strong_meta_path = Path("./data/audioset_eval_strong.tsv")
        else:
            strong_meta_path = Path("./data/audioset_train_strong.tsv")
        save_root = (
            dataset_root
            / f"track_sam_masks_for_{dataset_name}_strong"
            / split2subfolder[split]
        )
        prompt_annot_path = Path("./data/audioset_strong_ontology.csv")
        labels_path = Path("./data/audioset_labels.csv")
        weak_meta_path = Path("./data/audioset.csv")
    elif dataset_name == "lrs3":
        split = None
        video_root = (
            dataset_root / "h264_uncropped_25fps_256side_16000hz_aac" / "pretrain"
        )
        strong_meta_path = Path("./data/lrs3_strong_attrib_meta.csv")
        save_root = (
            dataset_root / f"track_sam_masks_for_{dataset_name}_strong" / "pretrain"
        )
        prompt_annot_path = None  # Path('./data/audioset_strong_ontology.csv')
        labels_path = None  # Path('./data/audioset_labels.csv')
        weak_meta_path = None  # Path('./data/audioset.csv')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    assert dataset_root.exists()
    save_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to {str(save_root)}")

    track_sam_root = Path("./scripts/data_prep/track_sam")

    # Grounding DINO checkpoints
    detector_args["config_file"] = (
        track_sam_root
        / "src/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    detector_args["grounding_dino_ckpt"] = (
        track_sam_root / "ckpt/groundingdino_swint_ogc.pth"
    )
    # detector_args['config_file'] = track_sam_root / 'src/groundingdino/groundingdino/config/GroundingDINO_SwinB_cfg.py'
    # detector_args['grounding_dino_ckpt'] = track_sam_root / 'ckpt/groundingdino_swinb_cogcoor.pth'
    # inference configs
    box_threshold, text_threshold, box_size_threshold, reset_image = (
        0.35,
        0.35,
        0.9,
        True,
    )
    # NMS
    do_nms = True
    iou_threshold = 0.5

    # SAM ckpt (91M; 308M; 636M)
    # sam_args['sam_checkpoint'] = track_sam_root / 'ckpt/sam_vit_b_01ec64.pth'
    # sam_args['model_type'] = 'vit_b'
    sam_args["sam_checkpoint"] = track_sam_root / "ckpt/sam_vit_l_0b3195.pth"
    sam_args["model_type"] = "vit_l"
    # sam_args['sam_checkpoint'] = track_sam_root / 'ckpt/sam_vit_h_4b8939.pth'
    # sam_args['model_type'] = 'vit_h'

    # AOT
    aot_args["model_path"] = track_sam_root / "ckpt/R50_DeAOTL_PRE_YTB_DAV.pth"
    sam_args["generator_args"] = {
        "points_per_side": 30,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.9,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 200,
    }
    # segtracker_args['max_obj_num'] = 20
    segtracker_args["sam_gap"] = 250  # the interval to run sam to segment new objects

    strong_annotations, mid2label, _, ytidms2dir = load_strong_annotations(
        strong_meta_path,
        prompt_annot_path,
        labels_path,
        weak_meta_path,
        dataset_name,
    )
    ytidms_vidpath = make_strong_unique_vid_paths(
        strong_annotations,
        video_root,
        ytidms2dir,
        split,
        dataset_name,
    )

    print("Loading the model...")
    segtracker = SegTracker(segtracker_args, sam_args, aot_args, detector_args)

    ### iterate over the videos
    for ytid_ms, video_path in ytidms_vidpath:
        if dataset_name == "audioset":
            fstem = f"{ytid_ms}_{int(ytid_ms[12:]) + 10000}"  # add 10000 to the start to match file naming
        elif dataset_name == "lrs3":
            fstem = "/".join(ytid_ms.split("/")[1:])
        # checking if the video was attempted already (skip if so, otherwise add to the attempted list)
        attempted_list_path = Path("./todel_attempted_ytids.txt")
        if attempted_list_path.exists() and fstem in set(
            [row.strip() for row in open(attempted_list_path)]
        ):
            print(f"Skipped: {ytid_ms}: attempted already")
            continue
        else:
            with open(attempted_list_path, "a") as f:
                f.write(f"{fstem}\n")
        ### TODO:
        # ytid_ms, video_path = [(ytid_ms, video_path) for ytid_ms, video_path in ytidms_vidpath if ytid_ms == 'A43JOxLa5MM_23000'][0]
        ### TODO:
        if not video_path.exists():
            print("File does not exist:", video_path)
            continue
        save_path_json = save_root / f"{fstem}.json"
        if save_path_json.exists():
            print(f"Skipped: {ytid_ms}: already exists")
            continue

        # get the frames and the fps
        frames, vfps = read_video_cv2(video_path)
        if len(frames) == 0:
            print(f"Skipped: {ytid_ms}: empty video")
            continue
        H, W, C = frames[0].shape

        # get the all strong annotations for the video
        vid_strong_annot = [
            item for item in strong_annotations if item["segment_id"] == ytid_ms
        ]
        vid_strong_annot = deepcopy(vid_strong_annot)

        # placeholder for masks for each frame
        masks = torch.zeros(
            (len(vid_strong_annot), len(frames), H, W), dtype=torch.uint8
        )
        all_box_annots = [None] * len(vid_strong_annot)
        all_grounding_captions = [
            mid2label[item["label"]]["prompt"] for item in vid_strong_annot
        ]
        for i, item in enumerate(vid_strong_annot):
            mid = item["label"]
            start, end = float(item["start_time_seconds"]), float(
                item["end_time_seconds"]
            )
            vstart_frames = int(vfps * start)
            vend_frames = int(vfps * end)
            if vstart_frames >= len(frames):
                print(
                    f"Skipped {ytid_ms}: shorter than the start ({start}) of the annotation {len(frames)}"
                )
                continue
            elif vstart_frames == vend_frames:
                vend_frames = vstart_frames + 1
                print(
                    f"Extended {ytid_ms}: zero duration {vstart_frames} -> {vend_frames}"
                )
                if vend_frames >= len(frames):
                    print(
                        f"But skipped {ytid_ms}: it became longer than the video {len(frames)}"
                    )
                    continue
            elif len(frames) < vend_frames:
                vend_frames = len(frames)
                print(
                    f"Trimmed {ytid_ms}: shorter than the end ({end}) of the annotation {len(frames)}"
                )
            label_prompt = mid2label[mid]
            grounding_caption = label_prompt["prompt"]
            if grounding_caption.startswith("skip/"):
                continue
            obj_masks, box_annots = get_obj_mask(
                frames,
                vstart_frames,
                vend_frames,
                segtracker,
                grounding_caption,
                box_threshold,
                text_threshold,
                box_size_threshold,
                reset_image,
                do_nms,
                iou_threshold,
                segtracker_args["sam_gap"],
            )
            if obj_masks is not None and box_annots is not None:
                all_box_annots[i] = box_annots
                masks[i, vstart_frames:vend_frames] = obj_masks

        # vid_annot = {'meta': vid_strong_annot, 'masks': masks, 'box_annots': all_box_annots}
        make_json_w_annots(
            vid_strong_annot,
            masks,
            all_box_annots,
            save_path_json,
            vfps,
            all_grounding_captions,
        )

    print("done")
    return "done"


def load_strong_annotations(
    strong_meta_path, prompt_annot_path, labels_path, weak_meta_path, dataset_name
):
    print("Loading annotations...")
    if dataset_name == "audioset":
        # load strong annotations
        with open(strong_meta_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            strong_annotations = []
            for row in reader:
                strong_annotations.append({k: v for k, v in zip(header, row)})
        # load prompt strong_annotations
        with open(prompt_annot_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            header = next(reader)
            mid2label = {}
            for indent, label, mid, prompt in reader:
                mid2label[mid] = {"prompt": prompt, "label": label}
        # load weak labels for each video to `ytidms2label`
        label2target = {l: int(t) for t, _, l in csv.reader(open(labels_path))}
        target2label = {t: label for label, t in label2target.items()}
        ytidms2label = {}
        ytidms2dir = {}
        short2long = {
            "unbalanced": "unbalanced_train_segments",
            "balanced": "balanced_train_segments",
            "eval": "eval_segments",
        }
        for shortdir_vid, start, _, targets, _ in csv.reader(
            open(weak_meta_path), quotechar='"'
        ):
            shortdir, ytid = shortdir_vid.split("/")
            ytid_ms = f"{ytid}_{int(float(start)*1000)}"
            targets = list(map(int, targets.split(",")))
            ytidms2label[ytid_ms] = [target2label[t] for t in targets]
            ytidms2dir[ytid_ms] = short2long[shortdir]
    elif dataset_name == "lrs3":
        # load strong annotations
        strong_annotations = []
        with open(strong_meta_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            header = ["segment_id", "start_time_seconds", "end_time_seconds", "label"]
            header_ = next(reader)  # remove the first row (header)
            for row in reader:
                row = row[:-1]  # remove the last column with prompt
                strong_annotations.append({k: v for k, v in zip(header, row)})
        # load prompt strong_annotations
        speech_mid = "/m/09x0r"
        speech_label = "speech"
        speech_prompt = "face"
        mid2label = {speech_mid: {"prompt": speech_prompt, "label": speech_label}}
        # load weak labels for each video to `ytidms2label`
        target2label = {0: speech_label}  # trivial case, just a plug
        ytidms2label = {
            annot["segment_id"]: [
                speech_label,
            ]
            for annot in strong_annotations
        }
        ytidms2dir = {annot["segment_id"]: "pretrain" for annot in strong_annotations}

    return strong_annotations, mid2label, ytidms2label, ytidms2dir


def make_strong_unique_vid_paths(
    strong_annotations, video_root, ytidms2dir, split, dataset_name, shuffle=True
):
    if dataset_name == "audioset":
        split2longdir = {
            "train": "unbalanced_train_segments",
            "valid": "balanced_train_segments",
            "test": "eval_segments",
        }
        longdir = split2longdir[split]
        unique_ytid_ms = list(set([item["segment_id"] for item in strong_annotations]))
        unique_ytid_ms = [
            ytid_ms for ytid_ms in unique_ytid_ms if ytidms2dir[ytid_ms] == longdir
        ]
        # adding 10000 to the end of the ytid_ms to match the file naming convention
        all_fnames = [
            f"{ytid_ms}_{int(ytid_ms[12:]) + 10000}.mp4" for ytid_ms in unique_ytid_ms
        ]
        ytidms_vidpath = [
            (ytid_ms, video_root / fname)
            for ytid_ms, fname in zip(unique_ytid_ms, all_fnames)
        ]
    elif dataset_name == "lrs3":
        unique_path_stubs = list(
            set([item["segment_id"] for item in strong_annotations])
        )
        video_root = video_root.parent
        ytidms_vidpath = [
            (stub, video_root / f"{stub}.mp4") for stub in unique_path_stubs
        ]
    if shuffle:
        random.shuffle(ytidms_vidpath)
    return ytidms_vidpath


def read_video_cv2(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, fps


def get_obj_mask(
    frames,
    vstart_frames,
    vend_frames,
    segtracker: SegTracker,
    grounding_caption,
    box_threshold,
    text_threshold,
    box_size_threshold,
    reset_image,
    do_nms,
    iou_threshold,
    sam_gap,
):
    segtracker.restart_tracker_inc_refs()
    obj_masks = []
    box_annots = None
    for frame_idx, frame in enumerate(deepcopy(frames[vstart_frames:vend_frames])):
        with torch.cuda.amp.autocast():
            # initial detection and segmentation
            if frame_idx == 0:
                pred_mask, _, box_annots = segtracker.detect_and_seg(
                    frame,
                    grounding_caption,
                    box_threshold,
                    text_threshold,
                    box_size_threshold,
                    reset_image,
                    do_nms,
                    iou_threshold,
                )
                if pred_mask is None:
                    return None, None
                segtracker.add_reference(frame, pred_mask)
            # add new reference, potentially with new objects
            # elif (frame_idx % int((vend_frames-vstart_frames)/3)) == 0:
            elif (frame_idx % sam_gap) == 0:
                seg_mask, _, _box_annots = segtracker.detect_and_seg(
                    frame,
                    grounding_caption,
                    box_threshold,
                    text_threshold,
                    box_size_threshold,
                    reset_image,
                    do_nms,
                    iou_threshold,
                )
                # don't add the new objects if they are not detected and continue like w/o making the ref
                if seg_mask is None:
                    track_mask = segtracker.track(frame, update_memory=True)
                else:
                    # if seg_mask is None, then the box_annots are bad as well; using only if successful
                    box_annots = _box_annots
                    track_mask = segtracker.track(frame)
                    # find new objects, and update tracker with new objects
                    new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
                    if np.sum(new_obj_mask > 0) > frame.shape[0] * frame.shape[1] * 0.4:
                        new_obj_mask = np.zeros_like(new_obj_mask)
                    pred_mask = track_mask + new_obj_mask
                    segtracker.add_reference(frame, pred_mask)
            else:
                pred_mask = segtracker.track(frame, update_memory=True)

        obj_masks.append(torch.from_numpy(pred_mask))
    obj_masks = torch.stack(obj_masks) if len(obj_masks) > 0 else None
    return obj_masks, box_annots


def make_json_w_annots(
    vid_strong_annot,
    masks,
    all_box_annots,
    save_path_json,
    vfps,
    all_grounding_captions,
):
    """
    `vid_strong_annot` - list (N) of dicts with strong annotations
                       `segment_id`, `label` (mid), `start_time_seconds`, `end_time_seconds`
    `masks` - tensor (N, Tv, H, W) of masks where elements are in [0, num_objs_in_frame_not_N]
    `all_box_annots` - list (N) of lists (num_objs_in_frame);
                     0: confidences; 1: gdino prompts; 2: obj ids in the frame [0, num_objs_in_frame_not_N];
                     3: bboxes in x0y0, x1y1
    `vfps` - video fps (int)
    `all_grounding_caption`: list of captions for GroundingDINO, for each strong annotation
    """
    assert len(vid_strong_annot) != 0, "No strong annotations for the video"

    for i, (mask, box_annots) in enumerate(zip(masks, all_box_annots)):
        found_objects = []

        if box_annots is not None:
            mask = einops.rearrange(mask, "t h w -> h w t")
            for conf, prompt, obj_id, bbox in zip(*box_annots):
                # maskUtils.encode expects uint8 and F (Fortran) order
                bi_masks = np.asarray((mask == obj_id).byte(), order="F")
                # 'counts' is a RLE bytes object, so we need to decode it to a string for json
                frame_rle_masks = [
                    f["counts"].decode() for f in maskUtils.encode(bi_masks)
                ]
                found_objects.append(
                    {
                        "id": obj_id,
                        "prompt": prompt,
                        "bbox": einops.rearrange(bbox, "x y -> (x y)").tolist(),
                        "conf": f"{conf:.3f}",
                        "rgb_mask": frame_rle_masks,
                    }
                )

        # renaming the keys and adding the annotations of the found objects in RGB
        vid_strong_annot[i]["start"] = vid_strong_annot[i].pop("start_time_seconds")
        vid_strong_annot[i]["end"] = vid_strong_annot[i].pop("end_time_seconds")
        vid_strong_annot[i]["mid"] = vid_strong_annot[i].pop("label")
        vid_strong_annot[i]["full_prompt"] = all_grounding_captions[i]
        vid_strong_annot[i]["found_objects"] = found_objects
        ytid_ms = vid_strong_annot[i].pop("segment_id")

    ytidms_annot = {
        "ytid_ms": ytid_ms,
        "thw": list(masks.shape[-3:]),
        "vfps": vfps,
        "strong_annotations": vid_strong_annot,
    }
    save_path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path_json, "w") as f:
        json.dump(ytidms_annot, f, indent=4)
    print(f"Done: {ytid_ms}")


if __name__ == "__main__":
    # for lrs3, you will need to run `.make_lrs3_strong_w_vad.py` to make './data/lrs3_strong_attrib_meta.csv`
    main()
    # also run the .as_strong_filter_attrib_dataset.py to filter the dataset for non-empty detections
