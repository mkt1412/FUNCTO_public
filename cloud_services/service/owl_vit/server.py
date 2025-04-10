import argparse
import base64
import io
import os
import json

import numpy as np
import torch

from transformers import pipeline
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from PIL import Image

from flask import Flask, request, jsonify

# Load checkpoints
# ckpt_name = "google/owlv2-base-patch16-ensemble"
ckpt_name = "google/owlv2-large-patch14-ensemble"

processor = Owlv2Processor.from_pretrained(ckpt_name)
model = Owlv2ForObjectDetection.from_pretrained(ckpt_name).to("cuda")


# Flask app
app = Flask(__name__)


def decode_base64(base64_image: str):
    image_data = base64.b64decode(base64_image)
    image_data = io.BytesIO(image_data)
    image = Image.open(image_data).convert("RGB")
    return image


def predict_boxes(image: Image.Image, text_queries: list[str], bbox_conf: float=0.2):
    inputs = processor(
        text=[text_queries],  # 2D list needed
        images=image, 
        return_tensors="pt"
    ).to("cuda")
    # Forward model
    with torch.no_grad():
        outputs = model(**inputs)
        
    padded_image_size = inputs.pixel_values.shape[2:]  # (batch, channel, w?, h?)
    target_sizes=torch.Tensor([padded_image_size]).cuda()  # 2D list needed
    results = processor.post_process_object_detection(
        outputs=outputs, 
        target_sizes=target_sizes,
        threshold=bbox_conf
    )

    results = results[0]  # Retrieve predictions for the first image for the corresponding text queries
    
    bboxes, scores, labels = results["boxes"], results["scores"], results["labels"]

    # Restore bbox names
    box_names = [text_queries[i] for i in labels]
    # Find the correct bbox locations. The image was both padded and scaled.
    padded_width, padded_height = padded_image_size
    assert padded_width == padded_height, "This is strange. OWLViT should pad images to squares."
    width, height = image.size
    longest_edge = max(width, height)
    scale_ratio = longest_edge / padded_width
    # Rescale and normalize bbox positions
    for bbox in bboxes:
        bbox[0::2] = bbox[0::2] * scale_ratio / width
        bbox[1::2] = bbox[1::2] * scale_ratio / height

    return box_names, bboxes.tolist(), scores.tolist()


def match_by_image(image: Image.Image, query_image: Image.Image, match_threshold: float=0.2, nms_threshold=1.0):
    inputs = processor(images=image, query_images=query_image, return_tensors="pt").to("cuda")
    # Get predictions
    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)
    outputs.logits = outputs.logits.cpu()
    outputs.target_pred_boxes = outputs.target_pred_boxes.cpu()
    padded_image_size = inputs.pixel_values.shape[2:]  # (batch, channel, w?, h?)
    target_sizes=torch.Tensor([padded_image_size]).cuda()  # 2D list needed
    results = processor.post_process_image_guided_detection(outputs=outputs, threshold=match_threshold, nms_threshold=nms_threshold, target_sizes=target_sizes)
    bboxes, scores = results[0]["boxes"], results[0]["scores"]

    # Find the correct bbox locations. The image was both padded and scaled.
    padded_width, padded_height = padded_image_size
    assert padded_width == padded_height, "This is strange. OWLViT should pad images to squares."
    width, height = image.size
    longest_edge = max(width, height)
    scale_ratio = longest_edge / padded_width

    # Rescale and normalize bbox positions
    for bbox in bboxes:
        bbox[0::2] = bbox[0::2] * scale_ratio / width
        bbox[1::2] = bbox[1::2] * scale_ratio / height

    return bboxes.tolist(), scores.tolist()


@app.route('/owl_detect', methods=['POST'])
def api_detect_objects():
    # Parse JSON data
    query_data = request.get_json()
    text_queries = query_data["text_queries"]
    base64_image = query_data["image"]

    # Decode the base64 image
    image = decode_base64(base64_image)

    box_names, bboxes, scores = predict_boxes(
        image, 
        text_queries,
        bbox_conf=query_data["bbox_conf_threshold"]
    )
    
    # Return results as JSON
    return jsonify({
        'scores': scores, 
        'bboxes': bboxes, 
        'box_names': box_names,
    })

@app.route('/owl_match_by_image', methods=['POST'])
def api_match_by_image():
    # Parse JSON data
    query_data = request.get_json()
    base64_image = query_data["image"]
    base64_query_image = query_data["query_image"]

    # Decode the base64 image
    image = decode_base64(base64_image)
    query_image = decode_base64(base64_query_image)

    bboxes, scores = match_by_image(
        image, 
        query_image,
        match_threshold=query_data["match_threshold"],
        nms_threshold=query_data["nms_threshold"]
    )
    
    # Return results as JSON
    return jsonify({
        'scores': scores, 
        'bboxes': bboxes, 
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OWL-ViT Server')
    parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
    parser.add_argument('--port', default=55570, type=int, help='Port number to run the app on')
    args = parser.parse_args()

    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)
