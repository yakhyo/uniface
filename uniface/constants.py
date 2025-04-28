# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from enum import Enum
from typing import Dict

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
    MNET = "arcface_mnet"
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


class LandmarkWeights(str, Enum):
    """
    MobileNet 0.5 from Insightface
    https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg
    """
    DEFAULT = "2d_106"

# fmt: on


MODEL_URLS: Dict[Enum, str] = {

    # RetinaFace
    RetinaFaceWeights.MNET_025: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1_0.25.onnx',
    RetinaFaceWeights.MNET_050: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1_0.50.onnx',
    RetinaFaceWeights.MNET_V1:  'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1.onnx',
    RetinaFaceWeights.MNET_V2:  'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv2.onnx',
    RetinaFaceWeights.RESNET18: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_r18.onnx',
    RetinaFaceWeights.RESNET34: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_r34.onnx',

    # MobileFace
    MobileFaceWeights.MNET_025:      'https://github.com/yakhyo/uniface/releases/download/v0.1.2/###',
    MobileFaceWeights.MNET_V2:       'https://github.com/yakhyo/uniface/releases/download/v0.1.2/###',
    MobileFaceWeights.MNET_V3_SMALL: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/###',
    MobileFaceWeights.MNET_V3_LARGE: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/###',

    # SphereFace
    SphereFaceWeights.SPHERE20: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/###',
    SphereFaceWeights.SPHERE36: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/###',


    # ArcFace
    ArcFaceWeights.MNET:   'https://github.com/yakhyo/uniface/releases/download/v0.1.2/w600k_mbf.onnx',
    ArcFaceWeights.RESNET: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/w600k_r50.onnx',

    # SCRFD
    SCRFDWeights.SCRFD_10G_KPS:  'https://github.com/yakhyo/uniface/releases/download/v0.1.2/scrfd_10g_kps.onnx',
    SCRFDWeights.SCRFD_500M_KPS: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/scrfd_500m_kps.onnx',


    # DDAFM
    DDAMFNWeights.AFFECNET7: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/affecnet7.script',
    DDAMFNWeights.AFFECNET8: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/affecnet8.script',

    # AgeGender
    AgeGenderWeights.DEFAULT: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/genderage.onnx',

    # Landmarks
    LandmarkWeights.DEFAULT: 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/2d106det.onnx',
}

MODEL_SHA256: Dict[Enum, str] = {
    # RetinaFace
    RetinaFaceWeights.MNET_025: 'b7a7acab55e104dce6f32cdfff929bd83946da5cd869b9e2e9bdffafd1b7e4a5',
    RetinaFaceWeights.MNET_050: 'd8977186f6037999af5b4113d42ba77a84a6ab0c996b17c713cc3d53b88bfc37',
    RetinaFaceWeights.MNET_V1:  '75c961aaf0aff03d13c074e9ec656e5510e174454dd4964a161aab4fe5f04153',
    RetinaFaceWeights.MNET_V2:  '3ca44c045651cabeed1193a1fae8946ad1f3a55da8fa74b341feab5a8319f757',
    RetinaFaceWeights.RESNET18: 'e8b5ddd7d2c3c8f7c942f9f10cec09d8e319f78f09725d3f709631de34fb649d',
    RetinaFaceWeights.RESNET34: 'bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630',

    # MobileFace
    MobileFaceWeights.MNET_025:      '#',
    MobileFaceWeights.MNET_V2:       '#',
    MobileFaceWeights.MNET_V3_SMALL: '#',
    MobileFaceWeights.MNET_V3_LARGE: '#',

    # SphereFace
    SphereFaceWeights.SPHERE20: '#',
    SphereFaceWeights.SPHERE36: '#',


    # ArcFace
    ArcFaceWeights.MNET:   '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f',
    ArcFaceWeights.RESNET: '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43',

    # SCRFD
    SCRFDWeights.SCRFD_10G_KPS:  '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91',
    SCRFDWeights.SCRFD_500M_KPS: '5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a',

    # DDAFM
    DDAMFNWeights.AFFECNET7: '10535bf8b6afe8e9d6ae26cea6c3add9a93036e9addb6adebfd4a972171d015d',
    DDAMFNWeights.AFFECNET8: '8c66963bc71db42796a14dfcbfcd181b268b65a3fc16e87147d6a3a3d7e0f487',

    # AgeGender
    AgeGenderWeights.DEFAULT: '4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb',

    # Landmark
    LandmarkWeights.DEFAULT: 'f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf',
}

CHUNK_SIZE = 8192
