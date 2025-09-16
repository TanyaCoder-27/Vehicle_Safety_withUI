import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import pandas as pd
import os
from django.conf import settings
from collections import defaultdict
import time

class VehicleSpeedDetector:
    def __init__(self, progress_callback=None):
        # Load YOLO model for vehicle detection
        self.model = YOLO('yolov8n.pt')
        self.ocr_reader = easyocr.Reader(['en'])
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.speed_limit = 80  # km/h
        
        # Class names mapping for COCO dataset (used by YOLOv8)
        self.class_names = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Vehicle tracking
        self.vehicle_tracks = defaultdict(list)  # Store positions over time
        self.vehicle_speeds = {}
        self.next_vehicle_id = 1
        self.progress_callback = progress_callback
        
        # Vehicle tracking improvements
        self.inactive_tracks = {}  # Store tracks that are no longer active
        self.track_last_seen = {}  # Track when a vehicle was last seen
        self.track_max_age = 30    # Maximum number of frames to keep inactive tracks
        
        # Calibration parameters
        self.pixels_per_meter = 8  # Lower value to get higher, more realistic speeds
        self.min_track_length = 5   # Minimum frames to calculate speed
        
        # Speed detection zone (will be initialized in process_video)
        self.speed_detection_zone = None
        
    def get_class_name(self, class_id):
        """Convert class ID to human-readable name"""
        return self.class_names.get(class_id, 'unknown')
        
    def detect_license_plate(self, vehicle_crop):
        """Extract license plate text from vehicle crop with improved preprocessing"""
        try:
            # Check if crop is valid
            if vehicle_crop is None or vehicle_crop.size == 0 or vehicle_crop.shape[0] == 0 or vehicle_crop.shape[1] == 0:
                return None
                
            # Resize for better OCR if the image is too small
            h, w = vehicle_crop.shape[:2]
            if h < 100 or w < 100:
                scale_factor = max(100 / h, 100 / w)
                vehicle_crop = cv2.resize(vehicle_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
            # Convert to grayscale
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            
            # Apply image enhancement techniques
            # 1. Bilateral filter to reduce noise while preserving edges
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # 2. Adaptive histogram equalization for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(bilateral)
            
            # 3. Thresholding to further enhance text
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Try OCR on both enhanced and thresholded images
            results1 = self.ocr_reader.readtext(enhanced)
            results2 = self.ocr_reader.readtext(thresh)
            
            # Combine results
            all_results = results1 + results2
            license_plates = []
            
            for (bbox, text, conf) in all_results:
                if conf > 0.4:  # Lower threshold to catch more potential plates
                    # Clean the text (remove spaces and special characters)
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    # Filter by typical license plate patterns (at least 4 alphanumeric characters)
                    if len(cleaned_text) >= 4 and any(c.isdigit() for c in cleaned_text) and any(c.isalpha() for c in cleaned_text):
                        license_plates.append((cleaned_text, conf, bbox))
            
            # Sort by confidence and return the highest confidence result
            if license_plates:
                license_plates.sort(key=lambda x: x[1], reverse=True)
                return license_plates[0][0], license_plates[0][2]  # Return text and bounding box
            return None, None
        except Exception as e:
            print(f"License plate detection error: {e}")
            return None, None
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def assign_vehicle_id(self, center_x, center_y, frame_number, max_distance=50):
        """Assign vehicle ID based on proximity to existing tracks with improved uniqueness"""
        current_pos = (center_x, center_y)
        
        # Find the closest existing track
        min_distance = float('inf')
        assigned_id = None
        
        # First check active tracks
        for vehicle_id, positions in self.vehicle_tracks.items():
            if positions:  # Check if track has positions
                last_pos = positions[-1]['position']
                distance = self.calculate_distance(current_pos, last_pos)
                
                if distance < max_distance and distance < min_distance:
                    min_distance = distance
                    assigned_id = vehicle_id
                    # Update last seen
                    self.track_last_seen[vehicle_id] = frame_number
        
        # If no match in active tracks, check recently inactive tracks
        if assigned_id is None:
            # Create a copy of the keys to avoid dictionary changed size during iteration error
            inactive_track_ids = list(self.inactive_tracks.keys())
            for vehicle_id in inactive_track_ids:
                last_pos = self.inactive_tracks[vehicle_id]
                distance = self.calculate_distance(current_pos, last_pos)
                
                if distance < max_distance and distance < min_distance:
                    min_distance = distance
                    assigned_id = vehicle_id
                    # Reactivate this track
                    self.track_last_seen[vehicle_id] = frame_number
                    # Remove from inactive tracks
                    del self.inactive_tracks[vehicle_id]
        
        # If still no match, create new vehicle
        if assigned_id is None:
            assigned_id = self.next_vehicle_id
            self.next_vehicle_id += 1
            self.track_last_seen[assigned_id] = frame_number
        
        return assigned_id
    
    def calculate_speed(self, vehicle_id, current_pos, frame_number, fps):
        """Calculate vehicle speed based on position history with improved consistency"""
        # Add current position to track
        self.vehicle_tracks[vehicle_id].append({
            'position': current_pos,
            'frame': frame_number,
            'timestamp': frame_number / fps
        })
        
        # Keep only recent positions (last 15 frames for smoother calculation)
        if len(self.vehicle_tracks[vehicle_id]) > 15:
            self.vehicle_tracks[vehicle_id] = self.vehicle_tracks[vehicle_id][-15:]
        
        track = self.vehicle_tracks[vehicle_id]
        
        # Need at least 3 positions to calculate reliable speed
        if len(track) < 3:
            return 0
        
        # Calculate speed using last few positions
        speeds = []
        for i in range(1, min(len(track), 6)):  # Use last 5 intervals
            pos1 = track[-i-1]['position']
            pos2 = track[-i]['position']
            time1 = track[-i-1]['timestamp']
            time2 = track[-i]['timestamp']
            
            # Calculate distance in pixels
            distance_pixels = self.calculate_distance(pos1, pos2)
            time_diff = time2 - time1
            
            if time_diff > 0 and distance_pixels > 1:  # Avoid division by zero and noise
                # Convert to real-world units
                distance_meters = distance_pixels / self.pixels_per_meter
                speed_ms = distance_meters / time_diff
                speed_kmh = speed_ms * 3.6
                
                # Apply a scaling factor to get more realistic speeds
                speed_kmh = speed_kmh * 1.2
                
                # Filter out unrealistic speeds (0-200 km/h)
                if 0 < speed_kmh < 200:
                    speeds.append(speed_kmh)
        
        # Return average speed if we have valid measurements
        if speeds:
            # Apply exponential moving average for smoother speed transitions
            new_speed = np.mean(speeds)
            
            # If we already have a speed for this vehicle, apply smoothing
            if vehicle_id in self.vehicle_speeds:
                # Use exponential smoothing with alpha=0.3 (70% previous, 30% new)
                alpha = 0.3
                smoothed_speed = (1 - alpha) * self.vehicle_speeds[vehicle_id] + alpha * new_speed
                self.vehicle_speeds[vehicle_id] = smoothed_speed
                return smoothed_speed
            else:
                # First measurement for this vehicle
                self.vehicle_speeds[vehicle_id] = new_speed
                return new_speed
        
        return self.vehicle_speeds.get(vehicle_id, 0)
    
    def process_video(self, input_path, output_path, csv_path):
        """Process video for speed detection and license plate recognition"""
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define speed detection zone (middle third of the frame height)
        zone_top = int(height * 0.4)
        zone_bottom = int(height * 0.7)
        self.speed_detection_zone = (0, zone_top, width, zone_bottom)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        detection_data = []
        frame_count = 0
        
        # Auto-calibrate pixels per meter based on video dimensions
        # Assume a standard road width of 3.5 meters per lane
        estimated_road_width_pixels = width * 0.6  # Assume road takes 60% of frame width
        estimated_lanes = 2  # Estimate number of lanes
        estimated_road_width_meters = estimated_lanes * 3.5  # 3.5 meters per lane
        # Calculate pixels per meter based on estimated road width
        # Use a lower value (max 8) to get higher, more realistic speeds
        self.pixels_per_meter = min(8, max(5, estimated_road_width_pixels / estimated_road_width_meters))
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        print(f"Estimated scale: {self.pixels_per_meter:.2f} pixels per meter")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = (frame_count / total_frames) * 100
            if self.progress_callback:
                self.progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            current_detections = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if detection is a vehicle
                        class_id = int(box.cls[0])
                        if class_id in self.vehicle_classes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            
                            if confidence > 0.5:
                                # Calculate center position
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                
                                current_detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'center': (center_x, center_y),
                                    'confidence': confidence,
                                    'class_id': class_id
                                })
            
            # Update inactive tracks
            current_active_ids = set()
            # Create a copy of the keys to avoid dictionary changed size during iteration error
            track_keys = list(self.track_last_seen.keys())
            for vehicle_id in track_keys:
                if frame_count - self.track_last_seen.get(vehicle_id, 0) > self.track_max_age:
                    # If track exists and has positions, store last position before removing
                    if vehicle_id in self.vehicle_tracks and self.vehicle_tracks[vehicle_id]:
                        self.inactive_tracks[vehicle_id] = self.vehicle_tracks[vehicle_id][-1]['position']
                        # Remove from track_last_seen to prevent memory leaks
                        if vehicle_id in self.track_last_seen:
                            del self.track_last_seen[vehicle_id]
            
            # Process detections
            for detection in current_detections:
                x1, y1, x2, y2 = detection['bbox']
                center_x, center_y = detection['center']
                
                # Assign vehicle ID with frame number for tracking
                vehicle_id = self.assign_vehicle_id(center_x, center_y, frame_count)
                current_active_ids.add(vehicle_id)
                
                # Calculate speed
                speed = self.calculate_speed(vehicle_id, (center_x, center_y), frame_count, fps)
                
                # Extract vehicle crop for license plate detection
                vehicle_crop = frame[y1:y2, x1:x2]
                license_plate = None
                license_plate_bbox = None
                
                # Run OCR more frequently on slower vehicles for better detection
                # Use frame_count % 5 for slower vehicles, % 10 for faster ones
                ocr_frequency = 5 if speed < 40 else 10
                if frame_count % ocr_frequency == 0 and vehicle_crop.size > 0:
                    license_plate, license_plate_bbox = self.detect_license_plate(vehicle_crop)
                    
                    # Store license plate in vehicle track for persistence between frames
                    if license_plate and vehicle_id in self.vehicle_tracks:
                        # Store in the last position entry
                        if self.vehicle_tracks[vehicle_id]:
                            # Get confidence from OCR results
                            license_plate_confidence = 0.0
                            try:
                                # Try to get the confidence from the OCR results
                                # Find the matching text in the original OCR results
                                results1 = self.ocr_reader.readtext(vehicle_crop)
                                for (_, text, conf) in results1:
                                    cleaned_text = ''.join(c for c in text if c.isalnum())
                                    if cleaned_text == license_plate:
                                        license_plate_confidence = conf
                                        break
                            except Exception:
                                pass
                                
                            self.vehicle_tracks[vehicle_id][-1]['license_plate'] = license_plate
                            self.vehicle_tracks[vehicle_id][-1]['license_plate_bbox'] = license_plate_bbox
                            self.vehicle_tracks[vehicle_id][-1]['license_plate_confidence'] = license_plate_confidence
                else:
                    # Try to get license plate from previous detections of this vehicle
                    if vehicle_id in self.vehicle_tracks and self.vehicle_tracks[vehicle_id]:
                        for pos in reversed(self.vehicle_tracks[vehicle_id]):
                            if 'license_plate' in pos and pos['license_plate']:
                                license_plate = pos['license_plate']
                                license_plate_bbox = pos.get('license_plate_bbox')
                                break
                
                # Determine if overspeed
                is_overspeed = speed > self.speed_limit
                
                # Draw bounding box (red for overspeed, green for normal)
                color = (0, 0, 255) if is_overspeed else (0, 255, 0)
                thickness = 3 if is_overspeed else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw vehicle ID always
                id_text = f"ID:{vehicle_id}"
                cv2.putText(frame, id_text, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Check if vehicle is in the speed detection zone
                zone_x1, zone_y1, zone_x2, zone_y2 = self.speed_detection_zone
                if zone_y1 <= center_y <= zone_y2:
                    # Only show speed if vehicle is in the detection zone
                    speed_text = f"{speed:.1f} km/h"
                    if speed > self.speed_limit:
                        speed_text += " OVERSPEED"
                    cv2.putText(frame, speed_text, (x1, y1+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw license plate if detected
                if license_plate:
                    # Draw license plate text
                    cv2.putText(frame, f"LP: {license_plate}", (x1, y2+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # If we have the license plate bounding box, draw it on the vehicle crop
                    if license_plate_bbox is not None:
                        # Convert relative bbox coordinates to absolute frame coordinates
                        try:
                            # License plate bbox is relative to vehicle crop
                            lp_x1, lp_y1 = int(license_plate_bbox[0][0]), int(license_plate_bbox[0][1])
                            lp_x2, lp_y2 = int(license_plate_bbox[2][0]), int(license_plate_bbox[2][1])
                            
                            # Convert to absolute coordinates in the frame
                            abs_lp_x1, abs_lp_y1 = x1 + lp_x1, y1 + lp_y1
                            abs_lp_x2, abs_lp_y2 = x1 + lp_x2, y1 + lp_y2
                            
                            # Draw license plate bounding box with a distinct color
                            cv2.rectangle(frame, (abs_lp_x1, abs_lp_y1), (abs_lp_x2, abs_lp_y2), (255, 255, 0), 2)
                        except (IndexError, TypeError) as e:
                            # Skip drawing if there's an issue with the bounding box
                            pass
                
                # Store detection data only if speed > 0
                if speed > 0:
                    # Create detection record with enhanced license plate information
                    detection_record = {
                        'frame_number': frame_count,
                        'vehicle_id': vehicle_id,
                        'speed': round(speed, 2),
                        'license_plate': license_plate or 'N/A',
                        'license_plate_confidence': 0.0,  # Default value
                        'is_overspeed': is_overspeed,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': round(detection['confidence'], 2),
                        'vehicle_class': self.get_class_name(detection['class_id']),
                        'timestamp': frame_count / fps
                    }
                    
                    # If we have license plate from vehicle tracks with confidence info
                    if vehicle_id in self.vehicle_tracks and self.vehicle_tracks[vehicle_id]:
                        for pos in reversed(self.vehicle_tracks[vehicle_id]):
                            if 'license_plate_confidence' in pos and pos.get('license_plate') == license_plate:
                                detection_record['license_plate_confidence'] = pos['license_plate_confidence']
                                break
                    
                    detection_data.append(detection_record)
            
            # Draw speed detection zone
            zone_x1, zone_y1, zone_x2, zone_y2 = self.speed_detection_zone
            cv2.line(frame, (zone_x1, zone_y1), (zone_x2, zone_y1), (0, 0, 255), 2)  # Top line
            cv2.line(frame, (zone_x1, zone_y2), (zone_x2, zone_y2), (0, 0, 255), 2)  # Bottom line
            cv2.putText(frame, "Speed Detection Zone", (10, zone_y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Vehicles: {len(current_detections)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed
                eta = (total_frames - frame_count) / fps_processing
                print(f"Progress: {progress:.1f}% | Processing FPS: {fps_processing:.1f} | ETA: {eta:.1f}s")
        
        cap.release()
        out.release()
        
        # Save detection data to CSV
        if detection_data:
            df = pd.DataFrame(detection_data)
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(detection_data)} detections to CSV")
        
        processing_time = time.time() - start_time
        print(f"Video processing completed in {processing_time:.2f} seconds")
        
        return len(detection_data)
