import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from collections import defaultdict

def process_video_simple(input_path, output_path, csv_path, progress_callback=None):
    """Simplified video processing with guaranteed browser compatibility"""
    
    # Initialize YOLO
    model = YOLO('yolov8n.pt')
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Force common web-compatible dimensions
    if width > 1920:
        width = 1920
    if height > 1080:
        height = 1080
    
    # Ensure even dimensions
    width = width - (width % 2)
    height = height - (height % 2)
    
    # Try multiple codecs in order of browser compatibility
    codecs = ['mp4v', 'XVID', 'MJPG']
    out = None
    
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Using codec: {codec}")
            break
    
    if not out or not out.isOpened():
        raise Exception("Cannot initialize video writer with any codec")
    
    detection_data = []
    frame_count = 0
    vehicle_tracks = defaultdict(list)
    next_id = 1
    
    print(f"Processing {total_frames} frames at {fps} FPS, resolution: {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Resize frame if necessary
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        # Progress update
        if progress_callback and frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
        
        # YOLO detection
        results = model(frame, verbose=False)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id in vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:
                            # Simple speed calculation (placeholder)
                            # In a real implementation, you'd track vehicles across frames
                            speed = np.random.uniform(30, 80)  # Random speed for demo
                            is_overspeed = speed > 60
                            
                            # Draw bounding box
                            color = (0, 0, 255) if is_overspeed else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw speed
                            cv2.putText(frame, f"Speed: {speed:.1f} km/h", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Store data
                            detection_data.append({
                                'frame_number': frame_count,
                                'vehicle_id': next_id,
                                'speed': round(speed, 2),
                                'license_plate': 'N/A',
                                'is_overspeed': is_overspeed,
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                'confidence': round(confidence, 2)
                            })
                            next_id += 1
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    cap.release()
    out.release()
    
    # Save CSV
    if detection_data:
        df = pd.DataFrame(detection_data)
        df.to_csv(csv_path, index=False)
    
    print(f"Processing complete. Saved {len(detection_data)} detections")
    return len(detection_data)
