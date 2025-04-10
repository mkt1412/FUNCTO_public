import argparse
import base64
from io import BytesIO
import requests

from flask import Flask, request, jsonify
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from transformers import BitsAndBytesConfig
import requests
import copy

pretrained = "lmms-lab/llava-next-110b"
model_name = "llava_qwen"
conv_template = "qwen_1_5" # Make sure you use correct chat template for different models

device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, load_4bit=True, device_map=device_map) # Add any other thing you want to pass in llava_model_args
model.eval()
model.tie_weights()

# LLaVA model setup
# model_id = "llava-hf/llava-v1.6-34b-hf"
# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# model_id = "lmms-lab/llama3-llava-next-8b"

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )
# pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
# pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
# pipe = pipeline("image-to-text", model=model_id)

# Flask app
app = Flask(__name__)


def convert_base64_to_pil_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    return image


@app.route('/llava_chat', methods=['POST'])
def llava_chat():
    # Parse JSON data
    query_data = request.get_json()
    base64_image = query_data["image"]
    prompt = query_data["prompt"]
    temperature = query_data["temperature"]
    max_new_tokens = query_data.get("max_new_tokens", 1024)

    # Convert base64 to PIL Image
    image = convert_base64_to_pil_image(base64_image)

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    question = DEFAULT_IMAGE_TOKEN + f"\n{prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]


    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

    # Return results as JSON
    return jsonify({
        'text': text_outputs
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLaVA Server')
    parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
    parser.add_argument('--port', default=55576, type=int, help='Port number to run the app on')
    args = parser.parse_args()

    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)
