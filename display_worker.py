import cv2
import numpy as np
import time

def display_worker(shared_dict):
    """
    Display worker process - hiển thị mỗi camera trong một window riêng
    
    Args:
        shared_dict: Multiprocessing.Manager().dict()
    """
    print("Display worker: Bắt đầu")
    
    # Kích thước mỗi window
    window_width = 320
    window_height = 240
    try:
        while True:
            # Lấy danh sách camera hiện có
            camera_names = list(shared_dict.keys())
            
            if not camera_names:
                time.sleep(0.1)
                continue
            
            # Hiển thị từng camera
            for cam_name in camera_names:
                # Lấy data từ shared_dict
                cam_data = shared_dict.get(cam_name, {})
                
                # Kiểm tra frame có mới không (timeout 2 giây)
                current_time = time.time()
                frame_age = current_time - cam_data.get('ts', 0)
                
                if (cam_data.get('status') == 'ok' and 
                    cam_data.get('frame') is not None and 
                    frame_age < 2.0):
                    
                    try:
                        # Decode JPEG bytes thành frame
                        jpeg_bytes = cam_data['frame']
                        nparr = np.frombuffer(jpeg_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Resize frame để fit vào window
                            frame = cv2.resize(frame, (window_width, window_height))
                            cv2.imshow(cam_name, frame)
                        else:
                            # Frame decode lỗi
                            _draw_no_signal_window(cam_name, window_width, window_height, "Decode Error")
                            
                    except Exception as e:
                        # Lỗi decode
                        _draw_no_signal_window(cam_name, window_width, window_height, f"Error: {str(e)[:20]}")
                        
                else:
                    # Camera không có tín hiệu hoặc timeout
                    status_text = f"Age: {frame_age:.1f}s" if frame_age > 2.0 else cam_data.get('status', 'unknown')
                    _draw_no_signal_window(cam_name, window_width, window_height, status_text)
            
            # Nhấn 'q' để thoát
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Display worker: Đang dừng...")
    finally:
        cv2.destroyAllWindows()
        print("Display worker: Đã dừng")

def _draw_no_signal_window(cam_name, width, height, status_text):
    """Vẽ window không có tín hiệu"""
    # Tạo canvas cho window
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Nền xám
    cv2.rectangle(canvas, (0, 0), (width, height), (50, 50, 50), -1)
    
    # Tên camera
    cv2.putText(canvas, cam_name, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Trạng thái
    cv2.putText(canvas, status_text, (10, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imshow(cam_name, canvas)