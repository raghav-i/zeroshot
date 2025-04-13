import cv2
import torch
import numpy as np
import time
import json
from datetime import datetime
from typing import List, Dict
import threading
from PIL import Image
import clip

# COCO dataset classes - used to prevent using classes that are already in COCO
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
    'hair drier', 'toothbrush', "stuff"
]

CONFIG = {
    "model_name": "ViT-B/32",  # Faster model
    "threshold": 0.3,
    "default_categories": [
        "lightbulb", "matchstick", "monitor", "lion", "gaming console",
        "water bottle", "saxophone", "ukulele", "yoga mat", "snow globe",
        "retro phone", "vintage camera", "electric toothbrush", "fidget spinner"
    ],
    "log_file": "detections_log.json",
    "target_fps": 30,
    "frame_processing_size": 384, 
    "enable_logging": True
}

# Filter out any default categories that are in COCO
CONFIG["default_categories"] = [cat for cat in CONFIG["default_categories"] 
                               if cat.lower() not in COCO_CLASSES]

class OptimizedZeroShotDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model_lock = threading.Lock()
        self._load_model()
        
        self.default_categories = CONFIG["default_categories"].copy()
        self.categories = self.default_categories.copy()
        self.custom_categories = []
        self.is_running = False
        self.latest_detections = []
        self.frame_count = 0
        self.fps = 0
        self.last_log_time = time.time()
        self.frame_times = []
        
        self.text_features = None
        self._update_text_features()
        
        self.input_buffer = ""
        self.input_mode = False
        self.replace_mode = False
        self.coco_warning = ""

    def _load_model(self):
        with self.model_lock:
            try:
                self.model, self.preprocess = clip.load(
                    CONFIG["model_name"],
                    device=self.device,
                    download_root="./clip_models"
                )
                self.model.eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load CLIP model: {e}")

    def _update_text_features(self):
        with self.model_lock:
            text_inputs = torch.cat([
                clip.tokenize(f"a photo of a {c}") for c in self.categories
            ]).to(self.device)
            with torch.no_grad():
                self.text_features = self.model.encode_text(text_inputs)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def filter_coco_classes(self, categories):
        """Filter out categories that exist in COCO dataset"""
        coco_detected = []
        filtered_categories = []
        
        for cat in categories:
            if cat.lower() in COCO_CLASSES:
                coco_detected.append(cat)
            else:
                filtered_categories.append(cat)
        
        if coco_detected:
            self.coco_warning = f"Skipped COCO classes: {', '.join(coco_detected)}"
            print(self.coco_warning)
        else:
            self.coco_warning = ""
            
        return filtered_categories

    def update_categories(self, new_categories: List[str], replace=False):
        new_categories = [c.strip() for c in new_categories if c.strip()]
        
        if not new_categories:
            print("Warning: No valid categories provided.")
            return
        
        # Filter out COCO classes
        filtered_categories = self.filter_coco_classes(new_categories)
        
        if not filtered_categories:
            print("All provided categories are in COCO dataset. Please provide different classes.")
            return
        
        with self.model_lock:
            self.custom_categories = filtered_categories
            
            if replace:
                self.categories = filtered_categories
                print(f"Categories replaced with: {self.categories}")
            else:
                combined = self.default_categories.copy()
                for cat in filtered_categories:
                    if cat not in combined:
                        combined.append(cat)
                self.categories = combined
                print(f"Categories updated to include both defaults and new items: {self.categories}")
            
            def _update_features():
                try:
                    text_inputs = torch.cat([
                        clip.tokenize(f"a photo of a {c}") for c in self.categories
                    ]).to(self.device)
                    
                    with torch.no_grad():
                        features = self.model.encode_text(text_inputs)
                        features /= features.norm(dim=-1, keepdim=True)
                    
                    with self.model_lock:
                        self.text_features = features
                        print(f"Detection ready for: {self.categories}")
                        
                except Exception as e:
                    print(f"Feature update failed: {e}")
                    with self.model_lock:
                        self.text_features = None

            self.text_features = None
            
            threading.Thread(target=_update_features, daemon=True).start()

    def _async_preprocess(self, frame: np.ndarray) -> torch.Tensor:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = CONFIG["frame_processing_size"] / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        return self.preprocess(Image.fromarray(frame_resized)).unsqueeze(0).to(self.device)

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        if not self.categories:
            return []

        try:
            with self.model_lock:
                text_features = self.text_features
                categories = self.categories.copy()
                
                if text_features is None:
                    return [{
                        "label": "Loading new categories...",
                        "confidence": 1.0,
                        "bbox": [0, 0, frame.shape[1], frame.shape[0]]
                    }]

            image_input = self._async_preprocess(frame)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                k = min(3, len(categories))
                if k > 0:
                    values, indices = similarity[0].topk(k)
                else:
                    return []

            return [
                {
                    "label": categories[idx],
                    "confidence": float(value),
                    "bbox": [0, 0, frame.shape[1], frame.shape[0]]
                }
                for value, idx in zip(values, indices)
                if value.item() > CONFIG["threshold"]
            ]
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        start_time = time.time()
        
        detections = self.detect_objects(frame)
        self.latest_detections = detections
        
        annotated = frame.copy()
        
        y_offset = 30
        if self.text_features is None:
            cv2.putText(
                annotated, 
                "MODEL UPDATING...",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )
            y_offset += 30
        
        # Display COCO warning if present
        if self.coco_warning:
            cv2.putText(
                annotated, 
                self.coco_warning,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )
            y_offset += 30
        
        display_cats = self.custom_categories if self.custom_categories else self.categories[:3]
        cv2.putText(
            annotated,
            f"Tracking: {', '.join(display_cats)}{'...' if len(self.categories) > len(display_cats) else ''}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )
        y_offset += 30
        
        for i, det in enumerate(detections):
            if det['label'] == "Loading new categories...":
                continue
                
            label = f"{det['label']}: {det['confidence']:.2f}"
            cv2.putText(
                annotated, 
                label,
                (10, y_offset + i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        
        self._update_fps(start_time)
        cv2.putText(
            annotated,
            f"FPS: {self.fps:.1f}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        
        return annotated

    def _update_fps(self, start_time: float):
        self.frame_times.append(time.time() - start_time)
        if len(self.frame_times) > 10:
            self.frame_times.pop(0)
        self.fps = 1 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0

    def log_detections(self):
        if not CONFIG["enable_logging"] or not self.latest_detections:
            return
            
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "detections": self.latest_detections,
            "categories": self.categories,
            "custom_categories": self.custom_categories,
            "fps": self.fps
        }
        
        try:
            with open(CONFIG["log_file"], "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
        except Exception as e:
            print(f"Logging error: {e}")

    def handle_keyboard_input(self, key):
        if key == 27:  # ESC key - cancel input
            self.input_mode = False
            self.input_buffer = ""
            print("Input canceled")
            return True
            
        if key == 13:  # Enter key - confirm input
            if self.input_buffer:
                new_categories = [cat.strip() for cat in self.input_buffer.split(",") if cat.strip()]
                if new_categories:
                    print(f"Updating categories to include: {new_categories}")
                    self.update_categories(new_categories, replace=self.replace_mode)
                else:
                    print("No valid categories entered")
            self.input_mode = False
            self.input_buffer = ""
            return True
            
        if key == 8 or key == 127:  # Backspace or Delete
            self.input_buffer = self.input_buffer[:-1] if self.input_buffer else ""
            return True
            
        if 32 <= key <= 126:  # Printable ASCII characters
            self.input_buffer += chr(key)
            return True
            
        return False

    def toggle_replace_mode(self):
        self.replace_mode = not self.replace_mode
        mode_desc = "REPLACE" if self.replace_mode else "ADD TO DEFAULTS"
        print(f"Category update mode set to: {mode_desc}")
        return self.replace_mode

    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error opening video source")
            return

        print("Controls:\n"
            "  Q: Quit\n"
            "  C: Add categories to defaults (press C then type)\n"
            "  R: Replace all categories (press R then type)\n"
            "  L: Toggle logging\n"
            "  F: Show/hide FPS\n"
            "  D: Reset to default categories\n"
            "  Enter: Confirm new categories\n"
            "  ESC: Cancel input mode")
        
        print("\nNOTE: Classes in COCO dataset will be automatically filtered out.")
        
        show_fps = True
        self.is_running = True
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = self.process_frame(frame)
            
            if self.input_mode:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
                
                mode_text = "REPLACE ALL" if self.replace_mode else "ADD TO DEFAULTS"
                input_display = f"[{mode_text}] New categories: {self.input_buffer}_"
                cv2.putText(
                    annotated, 
                    input_display, 
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (255, 255, 255), 
                    2,
                    cv2.LINE_AA
                )
            
            cv2.imshow("Zero-Shot Detection", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            
            if not self.input_mode:
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.input_mode = True
                    self.replace_mode = False
                    self.input_buffer = ""
                    print("Input mode activated: ADD TO DEFAULTS. Type categories separated by commas.")
                elif key == ord('r'):
                    self.input_mode = True
                    self.replace_mode = True
                    self.input_buffer = ""
                    print("Input mode activated: REPLACE ALL. Type categories separated by commas.")
                elif key == ord('d'):
                    print("Resetting to default categories")
                    self.custom_categories = []
                    self.categories = self.default_categories.copy()
                    self._update_text_features()
                elif key == ord('l'):
                    CONFIG["enable_logging"] = not CONFIG["enable_logging"]
                    print(f"Logging {'enabled' if CONFIG['enable_logging'] else 'disabled'}")
                elif key == ord('f'):
                    show_fps = not show_fps
            else:
                self.handle_keyboard_input(key)

            if time.time() - self.last_log_time > 5:
                threading.Thread(target=self.log_detections, daemon=True).start()
                self.last_log_time = time.time()

        cap.release()
        cv2.destroyAllWindows()
        self.is_running = False

if __name__ == "__main__":
    import os
    detector = OptimizedZeroShotDetector()
    
    source = input("Enter video path (or 0 for webcam): ").strip()
    try:
        source = int(source) if source.isdigit() else source
    except ValueError:
        source = source
        
    detector.run(source)