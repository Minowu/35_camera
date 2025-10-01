import cv2
import numpy as np
import time
from ultralytics import YOLO
import json
from datetime import datetime

class YOLOInference:
    """Class xử lý inference YOLO"""
    
    def __init__(self, model_path):
        """
        Args:
            model_path: Đường dẫn file .pt
        """
        self.model = YOLO(model_path)
        print(f"Đã load model YOLO: {model_path}")
    
    def detect(self, frame):
        """
        Detect objects trong frame
        
        Args:
            frame: OpenCV frame
            
        Returns:
            results: YOLO results object
        """
        try:
            results = self.model(frame)
            return results[0]  # Lấy result đầu tiên
        except Exception as e:
            print(f"Lỗi inference: {e}")
            return None
    
    def draw_results(self, frame, results):
        """
        Vẽ bounding box và label lên frame
        
        Args:
            frame: OpenCV frame
            results: YOLO results
            
        Returns:
            frame: Frame đã vẽ kết quả
        """
        if results is None or len(results.boxes) == 0:
            return frame
        
        # Vẽ từng detection
        for box in results.boxes:
            # Lấy tọa độ
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Lấy tên class
            class_name = results.names[class_id]
            
            # Vẽ bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Vẽ label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def get_detection_info(self, results, frame_shape):
        """
        Lấy thông tin detection theo định dạng JSON yêu cầu
        
        Args:
            results: YOLO results
            frame_shape: Tuple (height, width, channels)
            
        Returns:
            dict: Thông tin detection theo format JSON
        """
        if results is None or len(results.boxes) == 0:
            return {
                "detections": [],
                "detection_count": 0
            }
        
        detections = []
        for box in results.boxes:
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = results.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Tính center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2)
                },
                "center": {
                    "x": float(center_x),
                    "y": float(center_y)
                }
            })
        
        return {
            "detections": detections,
            "detection_count": len(detections)
        }

def ai_inference_worker(shared_dict, result_dict, cam_names=None, model_path="weights/model_vl_0205.pt"):
    """
    AI Inference worker process
    
    Args:
        shared_dict: Dict chứa frame từ camera
        result_dict: Dict để lưu kết quả detection
        cam_names: List tên camera cần process (None để process tất cả)
        model_path: Đường dẫn model YOLO
    """
    print("AI Inference worker: Bắt đầu")
    if cam_names is None:
        print("Processing all cameras")
    else:
        print(f"Processing {len(cam_names)} specific cameras")
    
    # Load YOLO model
    try:
        yolo = YOLOInference('weights/model_vl_0205.pt')
    except Exception as e:
        print(f"Lỗi load model: {e}")
        return
    
    frame_count = 0
    
    try:
        while True:
            # Lấy danh sách camera
            camera_names = list(shared_dict.keys())
            if cam_names is not None:
                camera_names = [cam for cam in camera_names if cam in cam_names]
            
            if not camera_names:
                time.sleep(0.1)
                continue
            
            # Process từng camera
            for cam_name in camera_names:
                cam_data = shared_dict.get(cam_name, {})
                
                # Kiểm tra frame có hợp lệ không
                current_time = time.time()
                frame_age = current_time - cam_data.get('ts', 0)
                
                if (cam_data.get('status') == 'ok' and 
                    cam_data.get('frame') is not None and 
                    frame_age < 2.0):
                    
                    try:
                        # Decode frame từ JPEG
                        jpeg_bytes = cam_data['frame']
                        nparr = np.frombuffer(jpeg_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Chạy inference
                            start_time = time.time()
                            results = yolo.detect(frame)
                            inference_time = time.time() - start_time
                            
                            if results is not None:
                                # Lấy thông tin detection theo format JSON
                                frame_shape = frame.shape
                                detection_info = yolo.get_detection_info(results, frame_shape)
                                
                                # Tạo kết quả JSON theo format yêu cầu
                                json_result = {
                                    "camera_id": cam_name,
                                    "frame_id": frame_count,
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                    "frame_shape": {
                                        "height": int(frame_shape[0]),
                                        "width": int(frame_shape[1]),
                                        "channels": int(frame_shape[2])
                                    },
                                    "detections": detection_info["detections"],
                                    "detection_count": detection_info["detection_count"]
                                }
                                
                                # In kết quả JSON ra console
                                print(json.dumps(json_result, indent=2))
                                
                                # Lưu vào result_dict (để tương thích với code cũ)
                                result_dict[cam_name] = {
                                    'frame': None,  # Không cần frame nữa
                                    'ts': current_time,
                                    'status': 'ok',
                                    'inference_time': inference_time,
                                    'detections': detection_info['detection_count'],
                                    'objects': detection_info['detections']
                                }
                            else:
                                # Inference lỗi
                                result_dict[cam_name] = {
                                    'frame': None,
                                    'ts': current_time,
                                    'status': 'inference_error',
                                    'inference_time': 0,
                                    'detections': 0,
                                    'objects': []
                                }
                        
                    except Exception as e:
                        print(f"Lỗi process camera {cam_name}: {e}")
                        result_dict[cam_name] = {
                            'frame': None,
                            'ts': current_time,
                            'status': 'error',
                            'inference_time': 0,
                            'detections': 0,
                            'objects': []
                        }
                
                else:
                    # Camera không có tín hiệu
                    if cam_name in result_dict:
                        result_dict[cam_name]['status'] = 'no_signal'
            
            frame_count += 1
            
            # Không cần inference quá nhanh
            time.sleep(0.05)  # ~20 FPS
            
    except KeyboardInterrupt:
        print("AI Inference worker: Đang dừng...")
    except Exception as e:
        print(f"AI Inference worker lỗi: {e}")
    finally:
        print("AI Inference worker: Đã dừng")