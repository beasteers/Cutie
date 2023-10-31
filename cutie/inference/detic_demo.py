''''''
import os
import sys
import time
import logging
import contextlib
from collections import defaultdict, Counter
from tqdm.contrib.logging import logging_redirect_tqdm

import cv2
import tqdm
import numpy as np
import torch
import supervision as sv

import cutie
from cutie.inference.assignment import ObjectAssignment
from torchvision.ops import masks_to_boxes

from cutie.inference.detic_demo import Detic


device = 'cuda'


def main(
        src, vocab='lvis', untracked_vocab=None, out_path=None, 
        detect_every=1, skip_frames=0, stop_detecting_after=None,
        fps_down=1, size=420, nms_threshold=0.1,
        limit=None, out_dir='output', out_prefix='cutie_', track_videos=True):
    out_path = out_path or f'{out_dir}/{out_prefix}{os.path.basename(src)}'
    if untracked_vocab:
        vocab = vocab + untracked_vocab
    
    # object detector
    detic_model = Detic(vocab, conf_threshold=0.5, masks=True).to(device)
    print(detic_model.labels)

    # object tracker
    cutie_model = cutie.get_model({}).to(device)

    # object label counter (dict of object_track_id -> {plate: 25, cutting_board: 1})
    # this should have the same keys as ds_tracker
    label_counts = defaultdict(lambda: Counter())

    draw = Drawer()

    # video read-write loop
    video_info, WH = get_video_info(src, size, fps_down, ncols=2)
    afps = video_info.fps / fps_down
    i_detect = int(detect_every*video_info.og_fps//fps_down) or 1
    i_stop_detecting = int(stop_detecting_after*video_info.og_fps//fps_down) or 1 if stop_detecting_after else None
    print('detecting every', i_detect, detect_every, afps, detect_every%(1/afps))
    detect_out_frame = None 
    
    with sv.VideoSink(out_path, video_info) as s, XMemWriter(out_path, video_info) as tw:
        for i, frame in enumerate(tqdm.tqdm(sv.get_video_frames_generator(src), total=video_info.total_frames)):
            if i < skip_frames or i%fps_down: continue
            if limit and i > limit: break

            frame = cv2.resize(frame, WH)

            # run detic
            detections = det_mask = det_labels = None
            track_index = None
            if detect_out_frame is None or not i % i_detect and (not i_stop_detecting or i < i_stop_detecting):
                # get object detections
                outputs = detic_model(frame)
                det_mask = det_masks = outputs["instances"].pred_masks.int()
                detections, det_labels = detectron2_to_sv(outputs, detic_model.labels)

                if nms_threshold:
                    selected_indices, overlap_indices = nms(detections.xyxy, detections.confidence, nms_threshold)
                    det_mask = det_masks[selected_indices]
                    detections = detections[selected_indices]

                # draw detic
                detect_out_frame = draw.draw_detections(frame, detections, det_labels)

                if untracked_vocab:
                    track_index = np.array([l not in untracked_vocab for l in detic_model.labels[detections.class_id]], dtype=bool)
                    # det_mask = det_mask[keep]
                    # detections = detections[keep]

            # run xmem
            pred_mask, track_ids, input_track_ids = cutie_model(
                frame, mask=det_mask, 
                objects=ObjectAssignment(labels=det_labels, track_index=track_index),
                only_confirmed=True)

            # draw xmem
            track_detections, track_labels = cutie_to_sv(pred_mask, cutie_model.object_manager, track_ids)
            track_out_frame = draw.draw_detections(frame, track_detections, track_labels)
            
            # write videos for each track
            if track_videos:
                tw.write_tracks(frame, track_detections)
            # write frame to file
            s.write_frame(np.concatenate([track_out_frame, detect_out_frame], axis=1))

    print("wrote to:", out_path)



class XMemWriter:
    def __init__(self, out_path, video_info, size=200, padding=0):
        # self.sink = sv.VideoSink(target_path=out_path, video_info=video_info)
        self.track_out_format = '{}_track{{}}{}'.format(*os.path.splitext(out_path))
        os.makedirs(os.path.dirname(self.track_out_format) or '.', exist_ok=True)
        self.video_info = sv.VideoInfo(width=size, height=size, fps=video_info.fps)
        self.size = (self.video_info.height, self.video_info.width)
        self.padding = padding
        self.writers = {}
        self.ba = sv.BoxAnnotator()
        self.ma = sv.MaskAnnotator()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for w in self.writers.values():
            w.__exit__(*a)
        # self.sink.__exit__(*a)

    def write_tracks(self, frame, detections):
        for tid, bbox in zip(detections.tracker_id, detections.xyxy):
            if tid not in self.writers:
                self.writers[tid] = sv.VideoSink(self.track_out_format.format(tid), video_info=self.video_info)
                self.writers[tid].__enter__()
            self._write_frame(self.writers[tid], frame, bbox)

    # def write_frame(self, frame, detections):
    #     frame = self._draw_detections(frame, detections)
    #     self.sink.write_frame(frame)

    def draw(self, frame, detections, labels=None):
        frame = frame.copy()
        frame = self.ma.annotate(frame, detections)
        frame = self.ba.annotate(frame, detections, labels=labels)
        return frame

    def _write_frame(self, writer, frame=None, bbox=None):
        if frame is None:
            frame = np.zeros(self.size, dtype='uint8')
        elif bbox is not None:
            x, y, x2, y2 = map(int, bbox)
            frame = frame[y - self.padding:y2 + self.padding, x - self.padding:x2 + self.padding]
        frame = resize_with_pad(frame, self.size)
        writer.write_frame(frame)



def nms(boxes, scores, iou_threshold, verbose=False):
    boxes = np.array(boxes)
    scores = np.array(scores)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Sort boxes by their confidence scores in descending order
    indices = np.argsort(area)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]

    selected_indices = []
    overlap_indices = []
    while len(boxes) > 0:
        # Pick the box with the highest confidence score
        b = boxes[0]
        selected_indices.append(indices[0])

        # Calculate IoU between the picked box and the remaining boxes
        intersection_area = (
            np.maximum(0, np.minimum(b[2], boxes[1:, 2]) - np.maximum(b[0], boxes[1:, 0])) * 
            np.maximum(0, np.minimum(b[3], boxes[1:, 3]) - np.maximum(b[1], boxes[1:, 1]))
        )
        # smaller_box_area = np.minimum(
        #     (b[2] - b[0]) * (b[3] - b[1]),
        #     (boxes[1:, 2] - boxes[1:, 0]) * (boxes[1:, 3] - boxes[1:, 1])
        # )
        smaller_box_area = np.minimum(area[0], area[1:])
        iou = intersection_area / (smaller_box_area + 1e-7)

        # Filter out boxes with IoU above the threshold
        overlap_indices.append(indices[np.where(iou > iou_threshold)[0]])
        filtered_indices = np.where(iou <= iou_threshold)[0]
        indices = indices[filtered_indices + 1]
        boxes = boxes[filtered_indices + 1]
        scores = scores[filtered_indices + 1]
        area = area[filtered_indices + 1]

    return selected_indices, overlap_indices


def resize_with_pad(image, new_shape):
    """Maintains aspect ratio and resizes with padding."""
    original_shape = (image.shape[1], image.shape[0])
    if not all(original_shape):
        return np.zeros(new_shape, dtype=np.uint8)
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)



def detectron2_to_sv(outputs, labels):
    inst = outputs["instances"].to('cpu')
    class_id = inst.pred_classes.int().numpy()
    detections = sv.Detections(
        xyxy=inst.pred_boxes.tensor.numpy(),
        mask=inst.pred_masks.int().numpy(),
        confidence=inst.scores.numpy(),
        class_id=class_id,
    )
    return detections, labels[class_id]


def cutie_to_sv(pred_mask, object_manager, track_ids):
    # convert to Detection object for visualization
    detections = sv.Detections(
        mask=pred_mask.cpu().numpy(),
        xyxy=masks_to_boxes(pred_mask).cpu().numpy(),
        class_id=track_ids,
        tracker_id=track_ids,
    )

    # draw xmem detections
    labels = [
        (object_manager[i].label or "")[:12]
        for i in detections.tracker_id
    ]
    return detections, labels



class Drawer:
    def __init__(self):
        self.box_ann = sv.BoxAnnotator(text_scale=0.4, text_padding=1)
        self.mask_ann = sv.MaskAnnotator()

    def draw_detections(self, frame, detections, labels):
        frame = frame.copy()
        frame = self.mask_ann.annotate(frame, detections)
        frame = self.box_ann.annotate(frame, detections, labels=labels)
        return frame


def get_video_info(src, size, fps_down=1, nrows=1, ncols=1):
    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
    aspect = video_info.width / video_info.height
    video_info.width = int(aspect*size)//16*16
    video_info.height = int(size)//16*16
    WH = video_info.width, video_info.height

    # double width because we have both detic and xmem frames
    video_info.width *= ncols
    video_info.height *= nrows
    # possibly reduce the video frame rate
    video_info.og_fps = video_info.fps
    video_info.fps /= fps_down or 1

    print(f"Input Video {src}\nsize: {WH}  fps: {video_info.fps}")
    return video_info, WH

import ipdb
@ipdb.iex
@torch.no_grad()
def main_profile(*a, profile=False, **kw):
    if not profile:
        return main(*a, **kw)
    from pyinstrument import Profiler
    prof = Profiler(async_mode='disabled')
    try:
        with prof:
            main(*a, **kw)
    finally:
        prof.print()



if __name__ == '__main__':
    import fire
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        fire.Fire(main_profile)
