"""Generate detections.json from video using a trained YOLO26s model and SAHI.
Usage: python process_video.py --model best.pt --video video.mp4 --output detections.json
"""
import argparse
import json
from pathlib import Path
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best.pt')
    parser.add_argument('--video', default='video.mp4')
    parser.add_argument('--output', default='detections.json')
    parser.add_argument('--interval', type=float, default=0.5, help='Seconds between sampled frames')
    parser.add_argument('--slice', type=int, default=640, help='SAHI slice dimension')
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--confidence', type=float, default=0.15, help='Minimum saved confidence; web can filter further')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.interval <= 0 or args.slice <= 0 or not 0 <= args.overlap < 1:
        parser.error('interval/slice must be positive and overlap in [0,1)')
    if not Path(args.model).is_file():
        parser.error(f'Model not found: {args.model}')
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        parser.error(f'Cannot open video: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        parser.error('Video FPS or dimensions invalid')
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics', model_path=args.model,
        confidence_threshold=args.confidence, device=args.device)
    samples = []
    frame_index = 0
    next_t = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_index / fps
            frame_index += 1
            if t + 1e-8 < next_t:
                continue
            next_t += args.interval
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = get_sliced_prediction(
                rgb, detection_model,
                slice_height=args.slice, slice_width=args.slice,
                overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
                postprocess_type='GREEDYNMM', postprocess_match_metric='IOS',
                postprocess_match_threshold=0.5, verbose=0)
            boxes = []
            for p in result.object_prediction_list:
                b = p.bbox
                boxes.append([round(max(0, b.minx), 2), round(max(0, b.miny), 2),
                              round(min(width, b.maxx), 2), round(min(height, b.maxy), 2),
                              round(float(p.score.value), 4)])
            samples.append({'t': round(t, 3), 'c': len(boxes), 'b': boxes})
            print(f'{t:8.2f}s | {len(boxes)} eggs (frame detections)', flush=True)
    finally:
        cap.release()
    payload = {'width': width, 'height': height, 'fps': fps,
               'interval_seconds': args.interval, 'model': 'YOLO26s + SAHI',
               'count_semantics': 'per_frame_not_unique_video_total', 'samples': samples}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f'Saved {len(samples)} samples to {args.output}')

if __name__ == '__main__':
    main()
