import multiprocessing as mp
from multiprocessing import Manager, Process
import time
import math
from camera_process import camera_process_worker
from display_worker import display_worker
from ai_inference import ai_inference_worker
from ai_display_worker import ai_display_worker

class CameraOrchestrator:
    """Orchestrator chính quản lý toàn bộ hệ thống"""
    
    def __init__(self, camera_urls, num_processes=4, max_retry_attempts=5, use_ai=True, model_path="yolov8n.pt"):
        """
        Args:
            camera_urls: List các URL camera
            num_processes: Số process (có thể tùy chỉnh)
            max_retry_attempts: Số lần thử kết nối lại tối đa cho mỗi camera
            use_ai: Có sử dụng AI detection không
            model_path: Đường dẫn model YOLO .pt
        """
        self.camera_urls = camera_urls
        self.num_processes = num_processes
        self.max_retry_attempts = max_retry_attempts
        self.use_ai = use_ai
        self.model_path = model_path
        self.manager = Manager()
        self.shared_dict = self.manager.dict()
        self.result_dict = self.manager.dict()  # Dict cho kết quả AI
        self.processes = []
        
    def _divide_cameras(self):
        """Chia nhóm camera cho các process"""
        total_cameras = len(self.camera_urls)
        cameras_per_process = math.ceil(total_cameras / self.num_processes)
        
        camera_groups = []
        for i in range(0, total_cameras, cameras_per_process):
            group = self.camera_urls[i:i + cameras_per_process]
            camera_groups.append(group)
        
        print(f"Chia {total_cameras} camera thành {len(camera_groups)} process:")
        for i, group in enumerate(camera_groups):
            print(f"  Process {i}: {len(group)} camera")
        
        return camera_groups
    
    def start(self):
        """Khởi động hệ thống"""
        print("Bắt đầu khởi động hệ thống camera...")
        print(f"AI Detection: {'BẬT' if self.use_ai else 'TẮT'}")
        if self.use_ai:
            print(f"Model YOLO: {self.model_path}")
        
        # Chia nhóm camera
        camera_groups = self._divide_cameras()
        
        # Tạo và spawn các process camera
        for i, camera_group in enumerate(camera_groups):
            process = Process(
                target=camera_process_worker,
                args=(i, camera_group, self.shared_dict, self.max_retry_attempts)
            )
            self.processes.append(process)
            process.start()
        
        if self.use_ai:
            print("Khởi động AI inference processes...")
            for i, camera_group in enumerate(camera_groups):
                ai_cam_names = [cam[0] for cam in camera_group]
                ai_process = Process(
                    target=ai_inference_worker,
                    args=(self.shared_dict, self.result_dict, ai_cam_names, self.model_path)
                )
                self.processes.append(ai_process)
                ai_process.start()
            
            # AI display worker (hiển thị kết quả có AI)
            ai_display_process = Process(
                target=ai_display_worker,
                args=(self.result_dict,)
            )
            self.processes.append(ai_display_process)
            ai_display_process.start()
            
        else:
            # Display worker thường (hiển thị frame gốc)
            display_process = Process(
                target=display_worker,
                args=(self.shared_dict,)
            )
            self.processes.append(display_process)
            display_process.start()
        
        print(f"Đã khởi động {len(self.processes)} process")
    
    def run_lifecycle(self):
        """Chạy vòng đời hệ thống"""
        try:
            print("Hệ thống đang chạy. Nhấn Ctrl+C để dừng...")
            while True:
                time.sleep(1)
                
                # Hiển thị thống kê (optional)
                active_cameras = len(self.shared_dict)
                print(f"Camera hoạt động: {active_cameras}", end='\r')
                
        except KeyboardInterrupt:
            print("\nĐang dừng hệ thống...")
            self._stop()
    
    def _stop(self):
        """Dừng tất cả process"""
        for process in self.processes:
            process.terminate()
        
        for process in self.processes:
            process.join(timeout=5.0)
        
        print("Đã dừng hệ thống")

def main():

    camera_urls = [
      ("Camera_01","rtsp://192.168.1.252:8554/live/cam1"),
      ("Camera_02","rtsp://192.168.1.252:8554/live/cam2"),
      ("Camera_03","rtsp://192.168.1.252:8554/live/cam3"),
      ("Camera_04","rtsp://192.168.1.252:8554/live/cam4"),
      ("Camera_05","rtsp://192.168.1.252:8554/live/cam5"),
      ("Camera_06","rtsp://192.168.1.252:8554/live/cam6"),
      ("Camera_07","rtsp://192.168.1.252:8554/live/cam7"),
      ("Camera_08","rtsp://192.168.1.252:8554/live/cam8"),
      ("Camera_09","rtsp://192.168.1.252:8554/live/cam9"),
      ("Camera_10","rtsp://192.168.1.252:8554/live/cam10"),
      ("Camera_11","rtsp://192.168.1.252:8554/live/cam11"),
      ("Camera_12","rtsp://192.168.1.252:8554/live/cam12"),
      ("Camera_13","rtsp://192.168.1.252:8554/live/cam13"),
      ("Camera_14","rtsp://192.168.1.252:8554/live/cam14"),
      ("Camera_15","rtsp://192.168.1.252:8554/live/cam15"),
      ("Camera_16","rtsp://192.168.1.252:8554/live/cam16"),
      ("Camera_17","rtsp://192.168.1.252:8554/live/cam17"),
      ("Camera_18","rtsp://192.168.1.252:8554/live/cam18"),
      ("Camera_19","rtsp://192.168.1.252:8554/live/cam19"),
      ("Camera_20","rtsp://192.168.1.252:8554/live/cam20"),
      ("Camera_21","rtsp://192.168.1.252:8554/live/cam21"),
      ("Camera_22","rtsp://192.168.1.252:8554/live/cam22"),
      ("Camera_23","rtsp://192.168.1.252:8554/live/cam23"),
      ("Camera_24","rtsp://192.168.1.252:8554/live/cam24"),
      ("Camera_25","rtsp://192.168.1.252:8554/live/cam25"),
      ("Camera_26","rtsp://192.168.1.252:8554/live/cam26"),
      ("Camera_27","rtsp://192.168.1.252:8554/live/cam27"),
      ("Camera_28","rtsp://192.168.1.252:8554/live/cam28"),
      ("Camera_29","rtsp://192.168.1.252:8554/live/cam29"),
      ("Camera_30","rtsp://192.168.1.252:8554/live/cam30"),
      ("Camera_31","rtsp://192.168.1.252:8554/live/cam31"),
      ("Camera_32","rtsp://192.168.1.252:8554/live/cam32"),
      ("Camera_33","rtsp://192.168.1.252:8554/live/cam33"),
      ("Camera_34","rtsp://192.168.1.252:8554/live/cam34"),
      ("Camera_35","rtsp://192.168.1.252:8554/live/cam35")
    ]
    
    # Tạo orchestrator với số process tùy chỉnh
    NUM_PROCESSES = 5  # Có thể thay đổi số này
    MAX_RETRY_ATTEMPTS = 5  # Số lần thử kết nối lại tối đa
    USE_AI = True  # Bật/tắt AI detection
    MODEL_PATH = "weights/model_vl_0205.pt"  # Đường dẫn model YOLO
    
    orchestrator = CameraOrchestrator(camera_urls, NUM_PROCESSES, MAX_RETRY_ATTEMPTS, USE_AI, MODEL_PATH)
    
    # Khởi động và chạy
    orchestrator.start()
    orchestrator.run_lifecycle()

if __name__ == "__main__":
    main()