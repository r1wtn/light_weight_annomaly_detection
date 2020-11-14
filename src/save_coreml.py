from coremltools.proto import FeatureTypes_pb2 as _FeatureTypes_pb2
from coremltools.models.utils import save_spec
from coremltools.models import MLModel
import coremltools as ct
import numpy as np
from PIL import Image
import argparse
import inspect

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--image_path', type=str)
parser.add_argument('-o', '--onnx_file', type=str)
args = parser.parse_args()
image_path = args.image_path
onnx_file = args.onnx_file

example_image = Image.open("../example/000001.png")
example_image = example_image.resize((128, 128))
coreml_file = onnx_file + ".mlmodel"

scale = 1.0 / (0.5 * 255.0)
preprocessing_args = dict(
    is_bgr=True,
    red_bias=-(0.5 * 255.0) * scale,
    green_bias=-(0.5 * 255.0) * scale,
    blue_bias=-(0.5 * 255.0) * scale,
    image_scale=scale
)

model = ct.converters.onnx.convert(
    model=onnx_file, image_input_names=["input_1"], preprocessing_args=preprocessing_args)
model.save(coreml_file)

out_dict = model.predict({"input_1": example_image})
print(out_dict)
