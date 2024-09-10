# from typing import Any
# import insightface

# import modules.globals
# from modules.typing import Frame

# FACE_ANALYSER = None


# def get_face_analyser() -> Any:
#     global FACE_ANALYSER

#     if FACE_ANALYSER is None:
#         FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
#         FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
#     return FACE_ANALYSER


# def get_one_face(frame: Frame) -> Any:
#     face = get_face_analyser().get(frame)
#     try:
#         return min(face, key=lambda x: x.bbox[0])
#     except ValueError:
#         return None


# def get_many_faces(frame: Frame) -> Any:
#     try:
#         return get_face_analyser().get(frame)
#     except IndexError:
#         return None


from typing import Any, List, Optional
import insightface
import modules.globals
from modules.typing import Frame,Face

FACE_ANALYSER = None

def initialize_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        print("PROVIDERS", modules.globals.execution_providers)
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers = modules.globals.execution_providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))

def set_to_gpu():
    global FACE_ANALYSER
    FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))

def get_face_analyser() -> Any:
    global FACE_ANALYSER
    # print("GET ONE FACE")
    if FACE_ANALYSER is None:
        initialize_face_analyser()
    # print("AFTER GET ONE FACE")
    return FACE_ANALYSER

def get_many_faces(frame: Frame) -> List[Face]:
    return FACE_ANALYSER.get(frame)

def get_one_face_left(frame: Frame) -> Optional[Face]:
    faces = FACE_ANALYSER.get(frame)
    return min(faces, key=lambda x: x.bbox[0]) if faces else None

def get_one_face_right(frame: Frame) -> Optional[Face]:
    faces = FACE_ANALYSER.get(frame)
    return max(faces, key=lambda x: x.bbox[0]) if faces else None

# def get_one_face(frame: Frame) -> Optional[Face]:
#     # faces = FACE_ANALYSER.get(frame, max_num=1)
#     print("BEFORE GET ONE FACE")
#     faces = get_face_analyser().get(frame, max_num = 1)
#     print("AFTER GET ONE FACE")
#     return faces[0] if faces else None

def get_one_face(frame: Frame) -> Any:
    print("herer before")
    print("type of frame", type(frame))
    print("frame size", frame.size)
    face = get_face_analyser().get(frame)
    print("herere after")
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

def get_two_faces(frame: Frame) -> List[Face]:
    faces = FACE_ANALYSER.get(frame, max_num=2)
    return sorted(faces, key=lambda x: x.bbox[0])
