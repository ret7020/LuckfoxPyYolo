import ctypes
import os
from od_structs import ObjectDetectResultList
from pathlib import Path

class YOLO:
    def __init__(self, model: str, img_size: int = 640):
        self.so_lib = Path(__file__).absolute().parent / "libYoloBindings.so"
        self.yolo_api = ctypes.CDLL(self.so_lib)

        # Setup types
        self.yolo_api.init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.yolo_api.init.restype = ctypes.c_int

        self.yolo_api.inference.argtypes = []
        self.yolo_api.inference.restype = ObjectDetectResultList

        self.yolo_api.read_image_file.argtypes = [ctypes.c_char_p]
        self.yolo_api.read_image_file.restype = None

        self.model_path = model
        self.img_size = img_size
        self.yolo_api.init(self.model_path.encode("utf-8"), self.img_size)

    def __call__(self, image_path: str):
        self.yolo_api.read_image_file(image_path.encode("utf-8"))
        return self.yolo_api.inference()

    def od_to_yolo(self): pass
    def release(self): self.yolo_api.release()

if __name__ == "__main__":
    m = YOLO("/root/model/yolov8.rknn")

    result = m("/root/model/bus.jpg")
    for ind in range(result.count):
        res = result.results[ind]
        x1 = res.box.left
        y1 = res.box.top

        x2 = res.box.right
        y2 = res.box.bottom

        print(x1, y1, x2, y2)
    
    m.release()
