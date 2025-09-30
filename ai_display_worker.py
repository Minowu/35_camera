import cv2
import numpy as np
import time

def ai_display_worker(result_dict):
    """
    AI Display worker - hiển thị mỗi camera với kết quả AI trong window riêng
    
    Args:
        result_dict: Dict chứa kết quả AI detection
    """
    print("AI Display worker: Bắt đầu")
    
    # Kích thước mỗi window
    window_width = 320
    window_height = 240
    
    # Stats tracking
    total_detections = 0
    frame_count = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Lấy danh sách camera
            camera_names = list(result_dict.keys())
            
            if not camera_names:
                time.sleep(0.1)
                continue
            
            frame_total_detections = 0
            
            # Hiển thị từng camera trong window riêng
            for cam_name in camera_names:
                # Lấy data từ result_dict
                cam_data = result_dict.get(cam_name, {})
                
                # Kiểm tra frame có mới không
                frame_age = current_time - cam_data.get('ts', 0)
                
                if (cam_data.get('status') == 'ok' and 
                    cam_data.get('frame') is not None and 
                    frame_age < 5.0):  # Tăng timeout cho AI process
                    
                    try:
                        # Decode frame với kết quả AI
                        jpeg_bytes = cam_data['frame']
                        nparr = np.frombuffer(jpeg_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Resize frame để fit vào window
                            frame = cv2.resize(frame, (window_width, window_height))
                            
                            # Thêm thông tin AI lên frame
                            detections = cam_data.get('detections', 0)
                            inference_time = cam_data.get('inference_time', 0)
                            
                            # Vẽ thông tin AI
                            info_text = f"Det: {detections} | {inference_time*1000:.0f}ms"
                            cv2.putText(frame, info_text, (5, 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                            
                            # Hiển thị frame
                            cv2.imshow(f"AI_{cam_name}", frame)
                            frame_total_detections += detections
                        else:
                            _draw_ai_error_window(f"AI_{cam_name}", window_width, window_height, 
                                                cam_name, "Decode Error")
                            
                    except Exception as e:
                        _draw_ai_error_window(f"AI_{cam_name}", window_width, window_height, 
                                            cam_name, f"Error: {str(e)[:15]}")
                        
                else:
                    # Camera không có tín hiệu hoặc AI chưa xử lý
                    status = cam_data.get('status', 'unknown')
                    if frame_age > 3.0:
                        status_text = f"Timeout: {frame_age:.1f}s"
                    else:
                        status_text = status
                    
                    _draw_ai_error_window(f"AI_{cam_name}", window_width, window_height, 
                                        cam_name, status_text)
            
            total_detections += frame_total_detections
            frame_count += 1
            
            # In stats định kỳ (mỗi 30 frame)
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {len(camera_names)} cameras, "
                      f"Total detections: {total_detections}, "
                      f"This frame: {frame_total_detections}")
            
            # Nhấn 'q' để thoát, 's' để save screenshot tất cả window
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                for cam_name in camera_names:
                    window_name = f"AI_{cam_name}"
                    screenshot_name = f"ai_{cam_name}_{timestamp}.jpg"
                    # Lưu screenshot của window hiện tại
                    img = cv2.getWindowImageRect(window_name)
                    if img[2] > 0 and img[3] > 0:  # Kiểm tra window có tồn tại
                        cv2.imwrite(screenshot_name, img)
                print(f"Đã lưu screenshot tất cả camera tại timestamp {timestamp}")
                
    except KeyboardInterrupt:
        print("AI Display worker: Đang dừng...")
    finally:
        cv2.destroyAllWindows()
        print("AI Display worker: Đã dừng")

def _draw_ai_error_window(window_name, width, height, cam_name, status_text):
    """Vẽ window lỗi cho AI display"""
    # Tạo canvas cho window
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Nền xám đậm với border đỏ
    cv2.rectangle(canvas, (0, 0), (width, height), (40, 40, 40), -1)
    cv2.rectangle(canvas, (0, 0), (width, height), (0, 0, 255), 2)
    
    # Tên camera
    cv2.putText(canvas, cam_name, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Trạng thái
    cv2.putText(canvas, status_text, (10, height-40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
    
    # AI indicator
    cv2.putText(canvas, "AI", (10, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Icon cảnh báo
    cv2.putText(canvas, "!", (width-30, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    cv2.imshow(window_name, canvas)