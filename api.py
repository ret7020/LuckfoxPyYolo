import ctypes 
import os

OBJ_NUMB_MAX_SIZE = 128 # Same as in lib (defined in postprocess.h)

# object detection result struct
class ImageRect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_int),
        ("top", ctypes.c_int),
        ("right", ctypes.c_int),
        ("bottom", ctypes.c_int),
    ]

class ObjectDetectResult(ctypes.Structure):
    _fields_ = [
        ("box", ImageRect),
        ("prop", ctypes.c_float),
        ("cls_id", ctypes.c_int),
    ]

class ObjectDetectResultList(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("count", ctypes.c_int),
        ("results", ObjectDetectResult * OBJ_NUMB_MAX_SIZE),
    ]

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


LIB_PATH = "/root/libHelloYolov8.so"
MODEL_PATH = b"/root/model/yolov8.rknn"
IMAGE_PATH = b"/root/model/bus.jpg"

yolo_api = ctypes.CDLL(LIB_PATH)

# Setup args

yolo_api.init.argtypes = [ctypes.c_char_p]
yolo_api.init.restype = None

yolo_api.inference.argtypes = [ctypes.c_char_p]
yolo_api.inference.restype = ObjectDetectResultList

# with suppress_stdout_stderr(): for debug enable this
yolo_api.init(MODEL_PATH)

result = yolo_api.inference(IMAGE_PATH)
for ind in range(result.count):
    res = result.results[ind]
    x1 = res.box.left
    y1 = res.box.top

    x2 = res.box.right
    y2 = res.box.bottom

    print(x1, y1, x2, y2)


yolo_api.release()
