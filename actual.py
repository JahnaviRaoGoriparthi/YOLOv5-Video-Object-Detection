import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import sys

# Add YOLOv5 directory to path (if needed)
sys.path.append('yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# Preprocess query to extract meaningful words
def preprocess_query(query):
    words = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    interrogative_terms = {'who', 'what', 'where', 'when', 'why', 'how', 'find'}
    meaningful_words = []
    for word, tag in pos_tag(words):
        if word not in stop_words and word not in interrogative_terms and (tag.startswith('N') or tag.startswith('V')):
            meaningful_words.append(word)
    return meaningful_words

def detect_objects_yolo(frame_path, model, device, meaningful_words):
    img = cv2.imread(frame_path)
    img0 = img.copy()
    img = letterbox(img, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    detected_meaningful_frames_withBB = []
    detected_meaningful_frames_withoutBB = []

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            labels = det[:, -1].cpu().numpy()
            label_names = [model.names[int(label)] for label in labels]

            found_words = set()
            for word in meaningful_words:
                for i, name in enumerate(label_names):
                    if word in name.lower():
                        xyxy = det[i, :4]
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(img0, c1, c2, (255, 0, 0), 2)  # Draw bounding box
                        found_words.add(word)
            if len(found_words) == len(meaningful_words):
                detected_meaningful_frames_withBB.append((frame_path.replace(".jpg", "_detected.jpg"), img0))
                detected_meaningful_frames_withoutBB.append(frame_path)

    return detected_meaningful_frames_withBB, detected_meaningful_frames_withoutBB

def main(video_path, query):
    output_folder = os.path.join('uploads', 'output', query)
    os.makedirs(output_folder, exist_ok=True)

    # Create subfolders for frames with and without bounding boxes
    withBB_folder = os.path.join(output_folder, 'withBB')
    withoutBB_folder = os.path.join(output_folder, 'withoutBB')
    os.makedirs(withBB_folder, exist_ok=True)
    os.makedirs(withoutBB_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting frames from {video_path}...")
    progress_step = max(total_frames // 10, 1)
    pbar = tqdm(total=total_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Save original frames to output folder
        frame_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1
        pbar.update(1)
        pbar.set_description(f"Extracted: {count}/{total_frames}")

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend('yolov5s.pt', device=device)
    model.eval()

    meaningful_words = preprocess_query(query)
    print(f"Meaningful words extracted from query: {meaningful_words}")

    frame_files = sorted([f for f in os.listdir(output_folder) if f.startswith('frame_') and f.endswith('.jpg')])
    detected_frames_withBB = []
    detected_frames_withoutBB = []

    print(f"Detecting objects in frames using YOLOv5...")
    pbar = tqdm(total=len(frame_files))
    for frame_file in frame_files:
        frame_path = os.path.join(output_folder, frame_file)
        detected_frame_paths_withBB, detected_frame_paths_withoutBB = detect_objects_yolo(frame_path, model, device, meaningful_words)
        for detected_frame_path, processed_img in detected_frame_paths_withBB:
            # Save frames with bounding boxes to withBB folder
            cv2.imwrite(os.path.join(withBB_folder, os.path.basename(detected_frame_path)), processed_img)
            detected_frames_withBB.append(detected_frame_path)
        for frame_path_withoutBB in detected_frame_paths_withoutBB:
            original_img = cv2.imread(frame_path_withoutBB)
            save_path_withoutBB = os.path.join(withoutBB_folder, os.path.basename(frame_path_withoutBB))
            cv2.imwrite(save_path_withoutBB, original_img)

            detected_frames_withoutBB.append(frame_path_withoutBB)

        pbar.update(len(detected_frame_paths_withBB))
        pbar.set_description(f"Frames processed: {len(detected_frames_withBB)}/{len(frame_files)}")

    pbar.close()

    # Create output video using frames from the withoutBB directory
    # Create output video using frames from the withoutBB directory
    if len(detected_frames_withoutBB) > 0:
        frame = cv2.imread(os.path.join(withoutBB_folder, os.path.basename(detected_frames_withoutBB[0])))
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(os.path.join('uploads', f'processed_{query}.mp4'), fourcc, 30.0, (width, height))

        for frame_path in detected_frames_withoutBB:
            frame = cv2.imread(os.path.join(withoutBB_folder, os.path.basename(frame_path)))
            if frame is None:
                continue
            out.write(frame)

        out.release()
    else:
        return None


    return os.path.join('uploads', f'processed_{query}.mp4')
