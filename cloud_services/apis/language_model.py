import datetime

from PIL import Image
import os
import requests

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import base64
from io import BytesIO

from openai import OpenAI
import google.generativeai as genai

# from utils.image_utils import convert_pil_image_to_base64

def convert_pil_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

class LanguageModel():
    def __init__(self, support_vision):
        self._support_vision = support_vision

    def support_vision(self)-> bool:
        return self._support_vision

class GPT4V(LanguageModel):
    def __init__(
        self, 
        model="gpt-4o-2024-05-13",
        temperature=0.0
    ):
        self.model = model
        
        self.temperature = temperature
        self.last_input = None
        super().__init__(
            support_vision=True
        )

    def chat(self, prompt, image, meta_prompt=""):
        base64_image = convert_pil_image_to_base64(image)

        # Get OpenAI API Key from environment variable
        # api_key = os.environ["OPENAI_API_KEY"]
        client = OpenAI(
        )
        self.last_input = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": meta_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ]
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": meta_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        ret = response.choices[0].message.content
        return ret

    def continue_chat(self, prompt, image, response):
        base64_image = convert_pil_image_to_base64(image)

        client = OpenAI(
        )
        message = self.last_input.append(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                        "type": "text",
                        "text": response
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=message,
            temperature=self.temperature,
            max_tokens=1024,
        )
        ret = response.choices[0].message.content
        return ret


class GPT4(LanguageModel):
    def __init__(
        self,
        model="gpt-4",
        temperature=0.0
    ):
        self.model = model
        
        self.temperature = temperature

        super().__init__(
            support_vision=True
        )

    def chat(self, prompt, meta_prompt=""):
        # Get OpenAI API Key from environment variable
        api_key = os.environ["OPENAI_API_KEY"]


        client = OpenAI(
            api_key=api_key,
        )
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": meta_prompt
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content


class GEMINI_PRO(LanguageModel):
    def __init__(self, temperature=0.0):
        # Configure the Google Generative AI with API key
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel('gemini-pro')

        self.temperature = temperature

    def chat(self, prompt, meta_prompt=""):
        response = self.client.generate_content(
            meta_prompt + prompt,
            generation_config=genai.types.GenerationConfig(temperature=self.temperature)
        )
        return response.text


class GEMINI_PRO_VISION(LanguageModel):
    def __init__(self, temperature=0.0):
        # Configure the Google Generative AI with API key
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel('gemini-pro-vision')

        self.temperature = temperature

    def chat(self, prompt, image, meta_prompt=""):
        messages = [meta_prompt, image, prompt]  # Gemini directly takes in PIL images
        response = self.client.generate_content(
            messages,
            generation_config=genai.types.GenerationConfig(temperature=self.temperature)
        )
        return response.text


class LLaVA(LanguageModel):
    def __init__(self, server_url="http://crane5.d2.comp.nus.edu.sg:55576/llava_chat", temperature=0.0):
        self.server_url = server_url

        self.temperature = temperature

    def chat(self, prompt, image, meta_prompt=""):
        base64_image = convert_pil_image_to_base64(image)
        payload = {
            "prompt": meta_prompt + '\n' + prompt,
            "image": base64_image,
            "max_new_tokens": 2048,
            "temperature": self.temperature
        }
        response = requests.post(
            self.server_url, 
            json=payload,
        ).json()
        return response["text"]
    