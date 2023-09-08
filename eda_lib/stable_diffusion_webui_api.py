import base64
import os
import random

import requests


def img2img(api, text, steps, denoising_strength, info):
    new_filename = info['file_name']
    api_url = f"{api}/sdapi/v1/img2img"
    with open(info['file_path'], 'rb') as file:
        image_data = file.read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    payload = {
        "init_images": [encoded_image],
        'prompt': text,
        "steps": steps,
        "denoising_strength": denoising_strength,
        "width": info['width'],
        "height": info['height'],
    }
    response = requests.post(api_url, json=payload)
    for i in range(random.randint(15, 25)):
        new_filename += random.choice('QAZXfrSWEDCVFRTqazxswgbnhyujmkiolpGBNHYUJedcvtMKIOLP')
    print(new_filename)
    if response.status_code == 200:
        response_data = response.json()
        encoded_result = response_data["images"][0]
        result_data = base64.b64decode(encoded_result)
        if not os.path.exists("diffimages"):
            os.makedirs("diffimages")
        output_path = f"diffimages/{new_filename}.jpg"

        with open(output_path, 'wb') as file:
            file.write(result_data)

        return { 'file_name': info['file_name'],
                'source': info['source'],
                'hotel_id': int(info['hotel_id']),
                'chain_id': int(info['chain_id']),
                'file_path': output_path,
                'width': info['width'],
                'height': info['height']
        }
    else:
        print("Ошибка при выполнении запроса:", response.text)
        return None