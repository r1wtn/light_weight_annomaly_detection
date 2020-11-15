from coremltools.proto import FeatureTypes_pb2 as _FeatureTypes_pb2
from coremltools.models.utils import save_spec
from coremltools.models import MLModel
import coremltools as ct
import numpy as np
from PIL import Image
import argparse
import inspect

parser = argparse.ArgumentParser(description='')
parser.add_argument('-o', '--onnx_file', type=str)
parser.add_argument('-f', '--saved_feature_path', type=str)
args = parser.parse_args()
onnx_file = args.onnx_file
saved_feature_path = args.saved_feature_path

saved_feature = np.loadtxt(saved_feature_path).reshape(-1)

example_image01 = Image.open("../example/000001.png")
example_image01 = example_image01.resize((128, 128))
example_image02 = Image.open("../example/000002.jpg")
example_image02 = example_image02.resize((128, 128))
example_image03 = Image.open("../example/000003.png")
example_image03 = example_image03.resize((128, 128))
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

out_dict01 = model.predict({"input_1": example_image01})
embedding01 = out_dict01["output_1"].reshape(-1)
out_dict02 = model.predict({"input_1": example_image02})
embedding02 = out_dict02["output_1"].reshape(-1)
out_dict03 = model.predict({"input_1": example_image03})
embedding03 = out_dict03["output_1"].reshape(-1)

# cosine similality
cos_sim = np.dot(saved_feature, embedding01) / \
    (np.linalg.norm(saved_feature, ord=2) * np.linalg.norm(embedding01, ord=2))
print(f"Image01: {cos_sim}")
cos_sim = np.dot(saved_feature, embedding02) / \
    (np.linalg.norm(saved_feature, ord=2) * np.linalg.norm(embedding02, ord=2))
print(f"Image02: {cos_sim}")
cos_sim = np.dot(saved_feature, embedding03) / \
    (np.linalg.norm(saved_feature, ord=2) * np.linalg.norm(embedding03, ord=2))
print(f"Image03: {cos_sim}")

print(embedding03)
