import ctypes

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