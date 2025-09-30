import time
from camera_thread import CameraThread

def camera_process_worker(process_id, camera_list, shared_dict, max_retry_attempts=5):
    """
    Worker function cho mỗi process
    
    Args:
        process_id: ID process
        camera_list: Danh sách camera [(name, url), ...]
        shared_dict: Multiprocessing.Manager().dict()
        max_retry_attempts: Số lần thử kết nối lại tối đa cho mỗi camera
    """
    print(f"Process {process_id}: Bắt đầu với {len(camera_list)} camera")
    
    # Dict local trong process
    local_dict = {}
    
    # Tạo và khởi động các camera thread
    threads = []
    for cam_name, cam_url in camera_list:
        thread = CameraThread(cam_name, cam_url, local_dict, max_retry_attempts)
        threads.append(thread)
        thread.start()
        print(f"Process {process_id}: Khởi động thread {cam_name}")
    
    # Vòng lặp cập nhật từ local_dict lên shared_dict
    try:
        while True:
            # Copy dữ liệu từ local_dict lên shared_dict
            for cam_name, data in local_dict.items():
                shared_dict[cam_name] = data
            
            time.sleep(0.1)  # Update mỗi 100ms
            
    except KeyboardInterrupt:
        print(f"Process {process_id}: Đang dừng...")
        
        # Dừng tất cả thread
        for thread in threads:
            thread.stop()
        
        # Đợi thread kết thúc
        for thread in threads:
            thread.join(timeout=1.0)