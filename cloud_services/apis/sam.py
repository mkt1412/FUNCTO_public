import requests
import pickle
import base64
import numpy as np
from PIL import Image

import io
import random
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import base64
from io import BytesIO

def convert_pil_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

class Segmentor():
    pass

class SAM(Segmentor):
    def __init__(self, server_url="http://crane5.d2.comp.nus.edu.sg:4001"):
    # def __init__(self, server_url="http://crane6.d2.comp.nus.edu.sg:55563"):
        self.server_url = server_url  

    def _send_request(self, endpoint: str, image: Image, additional_data: dict = None):
        """
        Send a request to the server with the specified image and additional data.
        
        :param endpoint: The endpoint for the specific segmentation method.
        :param image: The image to be segmented.
        :param additional_data: Additional data required by the specific method.
        :return: The response from the server.
        """
        image_base64 = convert_pil_image_to_base64(image)
        payload = {"image": image_base64}
        if additional_data:
            payload.update(additional_data)

        # Convert numpy arrays to lists
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()

        response = requests.post(f"{self.server_url}/{endpoint}", json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()

    def segment_auto_mask(self, image: Image.Image):
        """
        Automatically generate masks for the image.
        
        :param image: The image to be segmented.
        :return: Segmentation results.
        """
        response = self._send_request('sam_auto_mask_generation', image)
        return self._process_response(response)

    def segment_by_point_set(self, image: Image.Image, points: list, point_labels: list):
        """
        Generate masks for the image based on the provided point set (in 0-1 range).
        
        :param image: The image to be segmented.
        :param points: The points for segmentation (in 0-1 range). Shape: (nb_predictions, nb_points_per_mask, 2).
        :return: Segmentation results.
        """
        scaled_points = self._scale_points_to_image_size(points, image.size)
        # Request points should be in shape (nb_predictions, nb_points_per_mask, 2)

        response = self._send_request(
            endpoint='sam_mask_by_point_set',
            image=image, 
            additional_data={
                'points': scaled_points, 
                'labels': point_labels,
                'return_best': True
            }
        )
        return self._process_response(response)

    def segment_by_bboxes(self, image: Image.Image, bboxes: list):
        """
        Generate masks for the image based on the provided bounding box (in 0-1 range).
        
        :param image: The image to be segmented.
        :param bbox: The bounding box for segmentation (in 0-1 range).
        :return: Segmentation results.
        """
        # Correct the dimension of the bboxes: [bbox, bbox, ...] -> [[bbox], [bbox], ...]
        bboxes = [[bbox] for bbox in bboxes]
        
        scaled_bboxes = self._scale_bboxes_to_image_size(bboxes, image.size)
        response = self._send_request(
            endpoint='sam_mask_by_bbox',
            image=image, 
            additional_data={
                'bboxes': scaled_bboxes,
                'return_best': True
            }
        )
        return self._process_response(response)

    def _scale_points_to_image_size(self, points, image_size):
        """
        Scale points from 0-1 range to image size for a 3D array.

        :param points: 3D list of points in 0-1 range (nb_predictions, nb_points_per_mask, 2).
        :param image_size: Size of the image (width, height).
        :return: 3D list of points scaled to the image size.
        """
        width, height = image_size
        scaled_points = []

        for points_set in points:
            scaled_set = [[int(x * width), int(y * height)] for x, y in points_set]
            scaled_points.append(scaled_set)

        return scaled_points

    def _scale_bboxes_to_image_size(self, bboxes, image_size):
        """
        Scale bounding boxes from 0-1 range to image size for a 3D array.

        :param bboxes: 3D list of bounding boxes in 0-1 range (nb_predictions, nb_bboxes_per_mask, 4).
        :param image_size: Size of the image (width, height).
        :return: 3D list of bounding boxes scaled to the image size.
        """
        width, height = image_size
        scaled_bboxes = []

        for bbox_set in bboxes:
            assert len(bbox_set) == 1, "Only one bounding box allowed for each prediction."
            bbox = bbox_set[0]
            # Need to normailize the bboxes.
            scaled_set = [[int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)]]
            scaled_bboxes.append(scaled_set)

        return scaled_bboxes

    def _process_response(self, response):
        """
        Process the response from the server.
        
        :param response: The response from the server.
        :return: Processed segmentation results.
        """
        results = []
        for item in response["result"]:
            tmp_dict = {}
            # Decode the base64 string and then unpickle it
            tmp_dict["segmentation"] = pickle.loads(base64.b64decode(item["segmentation"]))
            del item["segmentation"]
            
            # Convert lists back to numpy arrays if necessary
            for key, value in item.items():
                if isinstance(value, list):
                    tmp_dict[key] = np.array(value)
                else:
                    tmp_dict[key] = value
            results.append(tmp_dict)
        return results
    

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