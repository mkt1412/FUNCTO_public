import json
from PIL import Image
from typing import List as list
import requests

import io
import random
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import base64
from io import BytesIO
from typing import List as list


def convert_pil_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

COMMON_OBJECTS = [
    "refrigerator",
    "oven",
    "microwave",
    "toaster",
    "blender",
    "coffee maker",
    "dishwasher",
    "pot",
    "pan",
    "cutting board",
    "knife",
    "spoon",
    "fork",
    "plate",
    "bowl",
    "cup",
    "coaster",
    "glass",
    "kettle",
    "paper towel holder",
    "trash can",
    "food storage container",
    "sofa",
    "coffee table",
    "television",
    "bookshelf",
    "armchair",
    "floor lamp",
    "rug",
    "picture frame",
    "curtain",
    "blanket",
    "vase",
    "indoor plant",
    "remote control",
    "candle",
    "wall art",
    "clock",
    "magazine rack",
    "phone",
    "pen",
    "marker",
    "laptop",
    "tape",
    "keyboard",
    "block",
]

# def convert_cxcywh_to_x1y1x2y2(cxcywh_box):
#     """
#     Converts bounding box coordinates from cXcYWH format to X1Y1X2Y2 format.

#     Parameters:
#     xywh_box (list or tuple): Bounding box in XYWH format [cx, cy, width, height].

#     Returns:
#     list: Bounding box in X1Y1X2Y2 format [x1, y1, x2, y2].
#     """
#     cx, cy, w, h = cxcywh_box
#     return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

class Detector():
    pass

class OWLViT(Detector):
    # def __init__(self, server_url="http://phoenix0.d2.comp.nus.edu.sg:55570"):
    def __init__(self, server_url="http://crane5.d2.comp.nus.edu.sg:4000"):
    # def __init__(self, server_url="http://crane6.d2.comp.nus.edu.sg:55570"):
        self.server_url = server_url

    def detect_objects(self, image: Image.Image, text_queries: list[str], bbox_score_top_k=20, bbox_conf_threshold=0.5):
        """
        Function to call an object detection API and return the response.

        Parameters:
        - image (Image): The image to detect.
        - text_queries (list of str): Text queries for object detection.
        - image_file_path (str): File path to the image to be analyzed.
        - bbox_score_top_k (int, optional): Number of top scoring bounding boxes to return. Defaults to 20.
        - bbox_conf_threshold (float, optional): Confidence threshold for bounding boxes. Defaults to 0.5.
        
        Returns:
        - tuple: Parsed response data from the API, containing scores, boxes, box_names, and objectnesses.
         Example result:
        [
            {'score': 0.3141017258167267,
            'bbox': [0.212062269449234,
            0.3956533372402191,
            0.29010745882987976,
            0.08735490590333939],
            'box_name': 'roof',
            'objectness': 0.09425540268421173
            }, ...
        ]
        """

        # Convert image to base64
        base64_image = convert_pil_image_to_base64(image.convert('RGB'))
        # Constructing the POST request
        payload = {
            "text_queries": text_queries,
            "image": base64_image,
            "bbox_score_top_k": bbox_score_top_k,
            "bbox_conf_threshold": bbox_conf_threshold
        }
        response = requests.post(
            self.server_url + "/owl_detect", 
            json=payload,
        )
        
        # Check for request failure
        if response.status_code != 200:
            raise ConnectionError(f"Request failed with status code {response.status_code}")

        resp_data = json.loads(response.text)
        
        # Retrieve the relevant data
        scores = resp_data['scores']
        bboxes = resp_data['bboxes']
        box_names = resp_data['box_names']
        
        
        # Assert that all lists have the same length
        assert len({len(scores), len(bboxes), len(box_names)}) == 1, "Server returned data with different lengths. Something is wrong, most probably on the server side."

        dict_data = [{'score': score, 'bbox': bbox, 'box_name': box_name} 
                 for score, bbox, box_name in zip(scores, bboxes, box_names)]

        return dict_data

    def match_by_image(self, image: Image.Image, query_image: Image.Image, match_threshold: float=0.8, nms_threshold: float=1.0):
        """
        Calls an object detection API to match objects within the provided 'image' against a 'query_image'.

        Parameters:
        - image (Image.Image): The target image in which to detect and match objects.
        - query_image (Image.Image): The query image containing the object(s) to match against the target image.
        - match_threshold (float, optional): The threshold for considering a detection a match. Defaults to 0.8.
        - nms_threshold (float, optional): The threshold for non-maximum suppression. Defaults to 1.0 - no suppression.

        Returns:
        - list of dicts: Each dict in the list corresponds to a matched object and contains 'score', 'bbox' (bounding box coordinates), 'box_name' (name of the detected object), and 'objectness' (a measure of how likely it is that the box contains any object).
        Example result format:
        [
            {
                'score': 0.3141,
                'bbox': [0.212, 0.396, 0.290, 0.087]
            }, 
            ...
        ]
        """
        # Convert image to base64
        base64_image = convert_pil_image_to_base64(image.convert('RGB'))
        base64_query_image = convert_pil_image_to_base64(query_image.convert('RGB'))
        # Constructing the POST request
        payload = {
            "image": base64_image,
            "query_image": base64_query_image,
            "match_threshold": match_threshold,
            "nms_threshold": nms_threshold
        }
        response = requests.post(
            self.server_url + "/owl_match_by_image", 
            json=payload,
        )
        
        # Check for request failure
        if response.status_code != 200:
            raise ConnectionError(f"Request failed with status code {response.status_code}")

        resp_data = json.loads(response.text)
        
        # Retrieve the relevant data
        scores = resp_data['scores']
        bboxes = resp_data['bboxes']        
        
        # Assert that all lists have the same length
        assert len({len(scores), len(bboxes)}) == 1, "Server returned data with different lengths. Something is wrong, most probably on the server side."

        dict_data = [{'score': score, 'bbox': bbox} for score, bbox in zip(scores, bboxes)]

        return dict_data

def visualize_image(image, masks=None, bboxes=None, points=None, show=True, return_img=False):
    img_height, img_width = np.array(image).shape[:2]
    plt.tight_layout()
    plt.imshow(image)
    plt.axis('off')
    plot = plt.gcf()

    # Overlay mask if provided
    if masks is not None:
        for mask in masks:
            colored_mask = np.zeros((*mask.shape, 4))
            random_color = [0.5 + 0.5 * random.random() for _ in range(3)] + [0.8]  # RGBA format
            colored_mask[mask > 0] = random_color
            plt.imshow(colored_mask) 

    # Draw bounding boxes if provided
    if bboxes is not None:
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1 *= img_width
            y1 *= img_height
            x2 *= img_width
            y2 *= img_height
            
            width = x2 - x1
            height = y2 - y1
            # Create a Rectangle patch
            rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='blue', facecolor='none')
            plt.gca().add_patch(rect)
            
    # Plot points if provided
    if points is not None:
        points = np.array(points)
        points[:, 0] = points[:, 0] * img_width
        points[:, 1] = points[:, 1] * img_height
        plt.scatter(points[:, 0], points[:, 1], c='red', s=50)  # larger circle
        plt.scatter(points[:, 0], points[:, 1], c='yellow', s=30)  # smaller circle inside

    if return_img:
        buffer = io.BytesIO()
        plot.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        img = Image.open(buffer)

    if show:
        plt.show(plot)

    plt.close(plot)

    if return_img:
        return img