import numpy as np
import tensorflow
from PIL import Image


from transformers import CLIPProcessor, TFCLIPModel, AutoTokenizer


class ClipModel:


    def __init__(self):
        self.model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.template = "a photo of a {label}"

    def load_hotel_tags(self, hotel_tags):
        self.hotel_tags = np.array(hotel_tags)
        self.hotel_class = [self.template.format(label=tag) for tag in hotel_tags]
        inputs = self.tokenizer(self.hotel_class, padding=True, return_tensors="tf")
        self.text_features = self.model.get_text_features(**inputs)

    def get_tags(self, image_path):
        image = Image.open(image_path)

        inputs = self.processor(images=image, return_tensors="tf")

        image_features = self.model.get_image_features(**inputs)

        # Pick the top 5 most similar labels for the image
        similarity = tensorflow.matmul(image_features, self.text_features, transpose_b=True)
        similarity /= tensorflow.norm(image_features, ord=2)
        similarity /= tensorflow.norm(self.text_features, ord=2, axis=1)
        values, indices = tensorflow.nn.top_k(similarity[0], k=10)

        return list(zip(values.numpy(), self.hotel_tags[indices]))








