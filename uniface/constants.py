# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from enum import Enum


# fmt: off
class SphereFaceWeights(str, Enum):
    """
    Trained on MS1M V2 dataset with 5.8 million images of 85k identities.
    https://github.com/yakhyo/face-recognition
    """
    SPHERE20      = "sphere20"
    SPHERE36      = "sphere36"

class MobileFaceWeights(str, Enum):
    """
    Trained on MS1M V2 dataset with 5.8 million images of 85k identities.
    https://github.com/yakhyo/face-recognition
    """
    MNET_025      = "mobilenetv1_025"
    MNET_V2       = "mobilenetv2"
    MNET_V3_SMALL = "mobilenetv3_small"
    MNET_V3_LARGE = "mobilenetv3_large"

class ArcFaceWeights(str, Enum):
    """
    Pretrained weights from ArcFace model (insightface).
    https://github.com/deepinsight/insightface
    """
    MNET   = "arcface_mnet"
    RESNET = "arcface_resnet"

class RetinaFaceWeights(str, Enum):
    """
    Trained on WIDER FACE dataset.
    https://github.com/yakhyo/retinaface-pytorch
    """
    MNET_025 =  "retinaface_mnet025"
    MNET_050 =  "retinaface_mnet050"
    MNET_V1  =  "retinaface_mnet_v1"
    MNET_V2  =  "retinaface_mnet_v2"
    RESNET18 =  "retinaface_r18"
    RESNET34 =  "retinaface_r34"


class SCRFDWeights(str, Enum):
    """
    Trained on WIDER FACE dataset.
    https://github.com/deepinsight/insightface
    """
    SCRFD_10G_KPS  = "scrfd_10g"
    SCRFD_500M_KPS = "scrfd_500m"


class YOLOv5FaceWeights(str, Enum):
    """
    Trained on WIDER FACE dataset.
    Original implementation: https://github.com/deepcam-cn/yolov5-face
    Exported to ONNX from: https://github.com/yakhyo/yolov5-face-onnx-inference

    Model Performance (WIDER FACE):
    - YOLOV5N: 11MB, 93.61% Easy / 91.52% Medium / 80.53% Hard
    - YOLOV5S: 28MB, 94.33% Easy / 92.61% Medium / 83.15% Hard
    - YOLOV5M: 82MB, 95.30% Easy / 93.76% Medium / 85.28% Hard
    """
    YOLOV5N = "yolov5n"
    YOLOV5S = "yolov5s"
    YOLOV5M = "yolov5m"


class DDAMFNWeights(str, Enum):
    """
    Trained on AffectNet dataset.
    https://github.com/SainingZhang/DDAMFN/tree/main/DDAMFN
    """
    AFFECNET7 = "affecnet7"
    AFFECNET8 = "affecnet8"


class AgeGenderWeights(str, Enum):
    """
    Trained on CelebA dataset.
    https://github.com/deepinsight/insightface
    """
    DEFAULT = "age_gender"


class FairFaceWeights(str, Enum):
    """
    FairFace attribute prediction (race, gender, age).
    Trained on FairFace dataset with balanced demographics.
    https://github.com/yakhyo/fairface-onnx
    """
    DEFAULT = "fairface"


class LandmarkWeights(str, Enum):
    """
    MobileNet 0.5 from Insightface
    https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg
    """
    DEFAULT = "2d_106"


class GazeWeights(str, Enum):
    """
    MobileGaze: Real-Time Gaze Estimation models.
    Trained on Gaze360 dataset.
    https://github.com/yakhyo/gaze-estimation
    """
    RESNET18     = "gaze_resnet18"
    RESNET34     = "gaze_resnet34"
    RESNET50     = "gaze_resnet50"
    MOBILENET_V2 = "gaze_mobilenetv2"
    MOBILEONE_S0 = "gaze_mobileone_s0"


class ParsingWeights(str, Enum):
    """
    Face Parsing: Semantic Segmentation of Facial Components.
    Trained on CelebAMask-HQ dataset.
    https://github.com/yakhyo/face-parsing
    """
    RESNET18 = "parsing_resnet18"
    RESNET34 = "parsing_resnet34"


class MiniFASNetWeights(str, Enum):
    """
    MiniFASNet: Lightweight Face Anti-Spoofing models.
    Trained on face anti-spoofing datasets.
    https://github.com/yakhyo/face-anti-spoofing

    Model Variants:
    - V1SE: Uses scale=4.0 for face crop (squeese-and-excitation version)
    - V2: Uses scale=2.7 for face crop (improved version)
    """
    V1SE = "minifasnet_v1se"
    V2   = "minifasnet_v2"


MODEL_URLS: dict[Enum, str] = {
    # RetinaFace
    RetinaFaceWeights.MNET_025:      'https://github.com/yakhyo/uniface/releases/download/weights/retinaface_mv1_0.25.onnx',
    RetinaFaceWeights.MNET_050:      'https://github.com/yakhyo/uniface/releases/download/weights/retinaface_mv1_0.50.onnx',
    RetinaFaceWeights.MNET_V1:       'https://github.com/yakhyo/uniface/releases/download/weights/retinaface_mv1.onnx',
    RetinaFaceWeights.MNET_V2:       'https://github.com/yakhyo/uniface/releases/download/weights/retinaface_mv2.onnx',
    RetinaFaceWeights.RESNET18:      'https://github.com/yakhyo/uniface/releases/download/weights/retinaface_r18.onnx',
    RetinaFaceWeights.RESNET34:      'https://github.com/yakhyo/uniface/releases/download/weights/retinaface_r34.onnx',
    # MobileFace
    MobileFaceWeights.MNET_025:      'https://github.com/yakhyo/uniface/releases/download/weights/mobilenetv1_0.25.onnx',
    MobileFaceWeights.MNET_V2:       'https://github.com/yakhyo/uniface/releases/download/weights/mobilenetv2.onnx',
    MobileFaceWeights.MNET_V3_SMALL: 'https://github.com/yakhyo/uniface/releases/download/weights/mobilenetv3_small.onnx',
    MobileFaceWeights.MNET_V3_LARGE: 'https://github.com/yakhyo/uniface/releases/download/weights/mobilenetv3_large.onnx',
    # SphereFace
    SphereFaceWeights.SPHERE20:      'https://github.com/yakhyo/uniface/releases/download/weights/sphere20.onnx',
    SphereFaceWeights.SPHERE36:      'https://github.com/yakhyo/uniface/releases/download/weights/sphere36.onnx',
    # ArcFace
    ArcFaceWeights.MNET:             'https://github.com/yakhyo/uniface/releases/download/weights/w600k_mbf.onnx',
    ArcFaceWeights.RESNET:           'https://github.com/yakhyo/uniface/releases/download/weights/w600k_r50.onnx',
    # SCRFD
    SCRFDWeights.SCRFD_10G_KPS:      'https://github.com/yakhyo/uniface/releases/download/weights/scrfd_10g_kps.onnx',
    SCRFDWeights.SCRFD_500M_KPS:     'https://github.com/yakhyo/uniface/releases/download/weights/scrfd_500m_kps.onnx',
    # YOLOv5-Face
    YOLOv5FaceWeights.YOLOV5N:       'https://github.com/yakhyo/yolov5-face-onnx-inference/releases/download/weights/yolov5n_face.onnx',
    YOLOv5FaceWeights.YOLOV5S:       'https://github.com/yakhyo/yolov5-face-onnx-inference/releases/download/weights/yolov5s_face.onnx',
    YOLOv5FaceWeights.YOLOV5M:       'https://github.com/yakhyo/yolov5-face-onnx-inference/releases/download/weights/yolov5m_face.onnx',
    # DDAFM
    DDAMFNWeights.AFFECNET7:         'https://github.com/yakhyo/uniface/releases/download/weights/affecnet7.script',
    DDAMFNWeights.AFFECNET8:         'https://github.com/yakhyo/uniface/releases/download/weights/affecnet8.script',
    # AgeGender
    AgeGenderWeights.DEFAULT:        'https://github.com/yakhyo/uniface/releases/download/weights/genderage.onnx',
    # FairFace
    FairFaceWeights.DEFAULT:         'https://github.com/yakhyo/fairface-onnx/releases/download/weights/fairface.onnx',
    # Landmarks
    LandmarkWeights.DEFAULT:         'https://github.com/yakhyo/uniface/releases/download/weights/2d106det.onnx',
    # Gaze (MobileGaze)
    GazeWeights.RESNET18:            'https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet18_gaze.onnx',
    GazeWeights.RESNET34:            'https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet34_gaze.onnx',
    GazeWeights.RESNET50:            'https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet50_gaze.onnx',
    GazeWeights.MOBILENET_V2:        'https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobilenetv2_gaze.onnx',
    GazeWeights.MOBILEONE_S0:        'https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx',
    # Parsing
    ParsingWeights.RESNET18:         'https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx',
    ParsingWeights.RESNET34:         'https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.onnx',
    # Anti-Spoofing (MiniFASNet)
    MiniFASNetWeights.V1SE:          'https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx',
    MiniFASNetWeights.V2:            'https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx',
}

MODEL_SHA256: dict[Enum, str] = {
    # RetinaFace
    RetinaFaceWeights.MNET_025:      'b7a7acab55e104dce6f32cdfff929bd83946da5cd869b9e2e9bdffafd1b7e4a5',
    RetinaFaceWeights.MNET_050:      'd8977186f6037999af5b4113d42ba77a84a6ab0c996b17c713cc3d53b88bfc37',
    RetinaFaceWeights.MNET_V1:       '75c961aaf0aff03d13c074e9ec656e5510e174454dd4964a161aab4fe5f04153',
    RetinaFaceWeights.MNET_V2:       '3ca44c045651cabeed1193a1fae8946ad1f3a55da8fa74b341feab5a8319f757',
    RetinaFaceWeights.RESNET18:      'e8b5ddd7d2c3c8f7c942f9f10cec09d8e319f78f09725d3f709631de34fb649d',
    RetinaFaceWeights.RESNET34:      'bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630',
    # MobileFace
    MobileFaceWeights.MNET_025:      'eeda7d23d9c2b40cf77fa8da8e895b5697465192648852216074679657f8ee8b',
    MobileFaceWeights.MNET_V2:       '38b148284dd48cc898d5d4453104252fbdcbacc105fe3f0b80e78954d9d20d89',
    MobileFaceWeights.MNET_V3_SMALL: 'd4acafa1039a82957aa8a9a1dac278a401c353a749c39df43de0e29cc1c127c3',
    MobileFaceWeights.MNET_V3_LARGE: '0e48f8e11f070211716d03e5c65a3db35a5e917cfb5bc30552358629775a142a',
    # SphereFace
    SphereFaceWeights.SPHERE20:      'c02878cf658eb1861f580b7e7144b0d27cc29c440bcaa6a99d466d2854f14c9d',
    SphereFaceWeights.SPHERE36:      '13b3890cd5d7dec2b63f7c36fd7ce07403e5a0bbb701d9647c0289e6cbe7bb20',
    # ArcFace
    ArcFaceWeights.MNET:             '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f',
    ArcFaceWeights.RESNET:           '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43',
    # SCRFD
    SCRFDWeights.SCRFD_10G_KPS:      '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91',
    SCRFDWeights.SCRFD_500M_KPS:     '5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a',
    # YOLOv5-Face
    YOLOv5FaceWeights.YOLOV5N:       'eb244a06e36999db732b317c2b30fa113cd6cfc1a397eaf738f2d6f33c01f640',
    YOLOv5FaceWeights.YOLOV5S:       'fc682801cd5880e1e296184a14aea0035486b5146ec1a1389d2e7149cb134bb2',
    YOLOv5FaceWeights.YOLOV5M:       '04302ce27a15bde3e20945691b688e2dd018a10e92dd8932146bede6a49207b2',
    # DDAFM
    DDAMFNWeights.AFFECNET7:         '10535bf8b6afe8e9d6ae26cea6c3add9a93036e9addb6adebfd4a972171d015d',
    DDAMFNWeights.AFFECNET8:         '8c66963bc71db42796a14dfcbfcd181b268b65a3fc16e87147d6a3a3d7e0f487',
    # AgeGender
    AgeGenderWeights.DEFAULT:        '4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb',
    # FairFace
    FairFaceWeights.DEFAULT:         '9c8c47d437cd310538d233f2465f9ed0524cb7fb51882a37f74e8bc22437fdbf',
    # Landmark
    LandmarkWeights.DEFAULT:         'f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf',
    # MobileGaze (trained on Gaze360)
    GazeWeights.RESNET18:            '23d5d7e4f6f40dce8c35274ce9d08b45b9e22cbaaf5af73182f473229d713d31',
    GazeWeights.RESNET34:            '4457ee5f7acd1a5ab02da4b61f02fc3a0b17adbf3844dd0ba3cd4288f2b5e1de',
    GazeWeights.RESNET50:            'e1eaf98f5ec7c89c6abe7cfe39f7be83e747163f98d1ff945c0603b3c521be22',
    GazeWeights.MOBILENET_V2:        'fdcdb84e3e6421b5a79e8f95139f249fc258d7f387eed5ddac2b80a9a15ce076',
    GazeWeights.MOBILEONE_S0:        'c0b5a4f4a0ffd24f76ab3c1452354bb2f60110899fd9a88b464c75bafec0fde8',
    # Face Parsing
    ParsingWeights.RESNET18:         '0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f',
    ParsingWeights.RESNET34:         '5b805bba7b5660ab7070b5a381dcf75e5b3e04199f1e9387232a77a00095102e',
    # Anti-Spoofing (MiniFASNet)
    MiniFASNetWeights.V1SE:          'ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676',
    MiniFASNetWeights.V2:            'b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907',
}

CHUNK_SIZE = 8192
