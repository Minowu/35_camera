import cv2
import numpy as np
import time
from ultralytics import YOLO
import json

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
            results = self.model(frame, verbose=False, device='cuda')
            return results  # Lấy result đầu tiên
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
    
    def get_detection_info(self, results):
        """
        Lấy thông tin detection để in ra
        
        Args:
            results: YOLO results
            
        Returns:
            dict: Thông tin detection
        """
        if results is None or len(results.boxes) == 0:
            return {"detections": 0, "objects": []}
        
        objects = []
        for box in results.boxes:
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = results.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            objects.append({
                "class": class_name,
                "confidence": float(confidence),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
        
        return {
            "detections": len(objects),
            "objects": objects
        }

def ai_inference_worker(shared_dict, result_dict, model_path="weights/model-hanam_0506.pt"):
    """
    AI Inference worker process
    
    Args:
        shared_dict: Dict chứa frame từ camera
        result_dict: Dict để lưu kết quả detection
        model_path: Đường dẫn model YOLO
    """
    print("AI Inference worker: Bắt đầu batch processing")
    print("Processing all cameras in single process for better efficiency")
    
    # Load YOLO model
    try:
        yolo = YOLOInference(model_path)
    except Exception as e:
        print(f"Lỗi load model: {e}")
        return
    
    frame_count = 0
    
    try:
        while True:
            # Lấy danh sách camera
            camera_names = list(shared_dict.keys())

            if not camera_names:
                time.sleep(0.1)
                continue
            
            # === TRUE BATCH PROCESSING ===
            # Bước 1: Thu thập tất cả frame hợp lệ
            valid_frames = {}
            current_time = time.time()
            
            for cam_name in camera_names:
                cam_data = shared_dict.get(cam_name, {})
                frame_age = current_time - cam_data.get('ts', 0)
                
                if (cam_data.get('status') == 'ok' and 
                    cam_data.get('frame') is not None and 
                    frame_age < 5.0):
                    
                    try:
                        # Decode frame từ JPEG
                        jpeg_bytes = cam_data['frame']
                        nparr = np.frombuffer(jpeg_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            valid_frames[cam_name] = {
                                'frame': frame,
                                'timestamp': current_time,
                                'original_data': cam_data
                            }
                    except Exception as e:
                        print(f"Lỗi decode frame {cam_name}: {e}")
            
            # Bước 2: True batch inference nếu có frames hợp lệ
            if valid_frames:
                batch_start_time = time.time()
                
                # Prepare batch data
                frames_list = []
                cam_names_list = []
                
                for cam_name, frame_data in valid_frames.items():
                    frames_list.append(frame_data['frame'])
                    cam_names_list.append(cam_name)
                
                # True batch inference với YOLO
                try:
                    batch_results = yolo.detect(frames_list)
                    batch_inference_time = time.time() - batch_start_time
                    
                    # Bước 3: Xử lý kết quả batch
                    for i, (cam_name, results) in enumerate(zip(cam_names_list, batch_results)):
                        frame_data = valid_frames[cam_name]
                        current_time = frame_data['timestamp']
                        
                        if results is not None:
                            # Vẽ kết quả lên frame
                            frame = frames_list[i]
                            frame_with_results = yolo.draw_results(frame.copy(), results)
                            
                            # Encode frame có kết quả
                            _, buffer = cv2.imencode('.jpg', frame_with_results, 
                                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                            result_jpeg = buffer.tobytes()
                            
                            # Lấy thông tin detection
                            detection_info = yolo.get_detection_info(results)
                            
                            # Lưu vào result_dict
                            result_dict[cam_name] = {
                                'frame': result_jpeg,
                                'ts': current_time,
                                'status': 'ok',
                                'inference_time': batch_inference_time / len(frames_list),
                                'detections': detection_info['detections'],
                                'objects': detection_info['objects']
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
                    
                    # Log batch performance
                    avg_time_per_frame = batch_inference_time / len(frames_list)
                    print(f"True batch: {len(frames_list)} cameras in {batch_inference_time:.3f}s (avg: {avg_time_per_frame:.3f}s/frame)")
                    
                except Exception as e:
                    print(f"Batch inference failed: {e}")
                    # Fallback to sequential processing
                    pass
            
            # Bước 4: Xử lý camera không có tín hiệu
            for cam_name in camera_names:
                if cam_name not in valid_frames:
                    if cam_name in result_dict:
                        result_dict[cam_name]['status'] = 'no_signal'
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("AI Inference worker: Đang dừng...")
    except Exception as e:
        print(f"AI Inference worker lỗi: {e}")
    finally:
        print("AI Inference worker: Đã dừng")