import argparse
import base64
import io
import pickle
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch
from transformers import SamModel, SamProcessor, pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

# Reference: https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
# For automatic mask generation
generator = pipeline(
    "mask-generation",
    model=model, 
    image_processor=processor.image_processor,
    device=device
)

def decode_base64(base64_image: str):
    image_data = base64.b64decode(base64_image)
    image_data = io.BytesIO(image_data)
    image = Image.open(image_data).convert("RGB")
    return image

def process_masks_for_response(masks: list):
    """Process masks for response. Compress masks as numpy array."""
    result = []
    for mask in masks:
        item = {}
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        item["segmentation"] = base64.b64encode(pickle.dumps(mask)).decode('utf-8')
        result.append(item)

    return result

def take_best_masks(masks: list):
    """For each prediction, keep only the best mask."""
    result = [mask[0] for mask in masks]  # The first dimension is the number of mask generated per prediction
    return result

    
@app.route('/sam_auto_mask_generation', methods=['POST'])
def sam_auto_mask_generation():
    data = request.json
    base64_image = data["image"]
    image = decode_base64(base64_image)

    # Call SAM to generate masks
    outputs = generator(image, points_per_batch=128)
    masks = outputs["masks"]

    results = process_masks_for_response(masks)
    return jsonify({'result': results, 'type': 'automatic mask generation'})


@app.route('/sam_mask_by_point_set', methods=['POST'])
def sam_mask_by_point_set():
    data = request.json
    base64_image = data["image"]
    image = decode_base64(base64_image)
    points = data['points']  # [[[550, 600], [2100, 1000]]]
    labels = data['labels']
    return_best = data['return_best']

    # One image per batch, thus the added dimension.
    inputs = processor(image, input_points=[points], input_labels=[labels], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            multimask_output=not return_best,
            **inputs
        )

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )[0]  # Only allow one image
    if return_best:
        masks = take_best_masks(masks)
    scores = outputs.iou_scores[0]

    results = process_masks_for_response(masks)
    return jsonify({'result': results, 'type': 'mask by point set'})


@app.route('/sam_mask_by_bbox', methods=['POST'])
def sam_mask_by_bbox():
    data = request.json
    base64_image = data["image"]
    image = decode_base64(base64_image)
    bboxes = data['bboxes']  # [[[650, 900, 1000, 1250]]]
    return_best = data['return_best']

    # One image per batch, thus the added dimension.
    inputs = processor(image, input_boxes=[bboxes], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            multimask_output=not return_best,
            **inputs
        )

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )[0]  # Only allow one image
    if return_best:
        masks = take_best_masks(masks)
    scores = outputs.iou_scores[0]

    results = process_masks_for_response(masks)
    return jsonify({'result': results, 'type': 'mask by bounding box'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAM Server')
    parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
    parser.add_argument('--port', default=55563, type=int, help='Port number to run the app on')
    args = parser.parse_args()

    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)