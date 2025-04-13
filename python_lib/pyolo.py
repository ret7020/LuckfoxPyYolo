import ctypes
import os
from od_structs import ObjectDetectResultList
from pathlib import Path

class suppress_stdout_stderr:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd_out = os.dup(1)
        self.save_fd_err = os.dup(2)

        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.save_fd_out, 1)
        os.dup2(self.save_fd_err, 2)
        os.close(self.null_fd)

class YOLO:
    def __init__(self, model: str, img_size: int = 640):
        self.so_lib = Path(__file__).absolute() / "libYoloBindings.so"
        self.yolo_api = ctypes.CDLL(self.so_lib)

        # Setup types
        self.yolo_api.init.argtypes = [ctypes.c_char_p]
        self.yolo_api.init.restype = None

        self.yolo_api.inference.argtypes = [ctypes.c_char_p]
        self.yolo_api.inference.restype = ObjectDetectResultList

        self.yolo_api.init()

    def __call__(self, image_path: str):
        return self.yolo_api.inference(image_path)

    def od_to_yolo(self): pass
    def release(self): self.yolo_api.release()

if __name__ == "__main__":
    m = YOLO("/root/model/yolov8.rknn")
    m("/root/model/bus.jpg")


# LIB_PATH = "/root/libHelloYolov8.so"
# MODEL_PATH = b"/root/model/yolov8.rknn"
# IMAGE_PATH = b"/root/model/bus.jpg"



# # Setup args



# # with suppress_stdout_stderr(): for debug enable this
# yolo_api.init(MODEL_PATH)

# result = yolo_api.inference(IMAGE_PATH)
# for ind in range(result.count):
#     res = result.results[ind]
#     x1 = res.box.left
#     y1 = res.box.top

#     x2 = res.box.right
#     y2 = res.box.bottom

#     print(x1, y1, x2, y2)


# yolo_api.release()
