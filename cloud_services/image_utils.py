from functools import partial
import io
import pickle
import random
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import base64
from io import BytesIO
from typing import List as list
from typing import Tuple as tuple

from PIL import Image
from torchvision import transforms

from cloud_services.visualizer import Visualizer

def normalized_bbox_to_pixel_scale(bbox, image) -> list[int]:
    width, height = image.size
    return int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)


def resize_image(image: Image.Image, size) -> Image:
    transform = transforms.Compose([transforms.Resize(int(size), interpolation=Image.BICUBIC)])
    processed_image = transform(image)
    return processed_image


def convert_pil_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# def visualize_bboxes(image: Image.Image, bboxes: list, alpha=0.9) -> Image.Image:
#     if len(bboxes) == 0:
#         return image
    
#     visualizer = Visualizer(image, metadata=None)

#     for i, bbox in enumerate(bboxes):
#         label = i + 1
#         # color_mask = np.random.random((1, 3)).tolist()[0]
#         demo = visualizer.draw_bbox_with_number(normalized_bbox_to_pixel_scale(bbox, image), text=str(label), alpha=alpha)
#     return Image.fromarray(demo.get_image())


def annotate_masks(
        image: Image.Image, 
        masks: list, 
        label_mode='1', 
        alpha=0.5, 
        draw_mask=False, 
        draw_mark=True, 
        draw_box=False,
        mark_position='center'
    ) -> Image.Image:
    if len(masks) == 0:
        return image

    visualizer = Visualizer(image, metadata=None)
    width, height = image.size
    # Construct annotation (drawing) mode
    anno_mode = []
    if draw_mask:
        anno_mode.append("Mask")
    if draw_mark:
        anno_mode.append("Mark")
    if draw_box:
        anno_mode.append("Box")

    mask_map = np.zeros((height, width), dtype=np.uint8)    
    for i, mask in enumerate(masks):
        label = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        # color_mask = [int(c*255) for c in color_mask]
        demo = visualizer.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, mark_position=mark_position)
        # assign the mask to the mask_map
        mask_map[mask == 1] = label
    return Image.fromarray(demo.get_image())


def annotate_positions_in_image(image: Image.Image, positions: list[tuple[int]], font_size=25) -> Image.Image:
    visualizer = Visualizer(image, metadata=None)
    width, height = image.size
    pixel_positions = [(x * width, y * height) for (x, y) in positions]  # Not accurate for drawing at the center of the text, but enough
    for idx, position in enumerate(pixel_positions, start=1):
        visualizer.draw_text(text=str(idx), position=position, font_size=font_size)
    return Image.fromarray(visualizer.get_output().get_image())


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


get_visualized_image = partial(visualize_image, show=False, return_img=True)
