import io
import pickle
import numpy as np
import requests
import base64
import json

from PIL import Image


def encode_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_mask(image: Image, img_size, label_mode, alpha, anno_mode):
    # transform = transforms.Compose([transforms.Resize(int(img_size), interpolation=Image.BICUBIC)])
    # image_ori = transform(image)
    image_base64 = encode_image_to_base64(image)
    payload = {
        "image": image_base64,
    }

    response = requests.post(server_url, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors

    results = []
    for item in response.json()["result"]:
        tmp_dict = {}
        # Decode the base64 string and then unpickle it
        tmp_dict["segmentation"] = pickle.loads(base64.b64decode(item["segmentation"]))
        
        # Convert lists back to numpy arrays if necessary
        for key, value in item.items():
            if isinstance(value, list):
                tmp_dict[key] = np.array(value)
        results.append(tmp_dict)
    return results




if __name__ == "__main__":
    test_image_path = "../../tests/assets/images/test.png" 
    server_url = "http://phoenix0.d2.comp.nus.edu.sg:55563/sam_auto_mask_generation"
    # Load the image from the file path
    test_image = Image.open(test_image_path)

    # Example parameters, adjust as needed
    img_size = 640
    label_mode = '1'
    alpha = 0.1
    anno_mode = ['Mask']

    results = get_mask(test_image, img_size, label_mode, alpha, anno_mode)
    print(results)