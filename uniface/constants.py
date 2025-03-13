# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

from typing import Dict


MODEL_URLS: Dict[str, str] = {
    'retinaface_mnet025': 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1_0.25.onnx',
    'retinaface_mnet050': 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1_0.50.onnx',
    'retinaface_mnet_v1': 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv1.onnx',
    'retinaface_mnet_v2': 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_mv2.onnx',
    'retinaface_r18': 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_r18.onnx',
    'retinaface_r34': 'https://github.com/yakhyo/uniface/releases/download/v0.1.2/retinaface_r34.onnx'
}

MODEL_SHA256: Dict[str, str] = {
    'retinaface_mnet025': 'b7a7acab55e104dce6f32cdfff929bd83946da5cd869b9e2e9bdffafd1b7e4a5',
    'retinaface_mnet050': 'd8977186f6037999af5b4113d42ba77a84a6ab0c996b17c713cc3d53b88bfc37',
    'retinaface_mnet_v1': '75c961aaf0aff03d13c074e9ec656e5510e174454dd4964a161aab4fe5f04153',
    'retinaface_mnet_v2': '3ca44c045651cabeed1193a1fae8946ad1f3a55da8fa74b341feab5a8319f757',
    'retinaface_r18': 'e8b5ddd7d2c3c8f7c942f9f10cec09d8e319f78f09725d3f709631de34fb649d',
    'retinaface_r34': 'bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630'
}

CHUNK_SIZE = 8192
