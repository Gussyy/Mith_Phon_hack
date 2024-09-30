import os
import torch
from torchvision import transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors

class ImageRetrieval:
    def __init__(self, data_path, model=None):
        self.data_path = data_path
        if model is None:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.model = model
        self.model.eval()  # Set the model to evaluation mode
        self.preprocess = transforms.Compose([
            transforms.Resize(224),  # Resize images to match DINOv2 input
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_classes, self.image_features = self.process_data()
        self.knn = NearestNeighbors(n_neighbors=1)  # Find the single nearest neighbor
        self.knn.fit(self.image_features)

    def process_data(self):
        image_classes = []
        image_features = []

        for class_name in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, class_name)
            if os.path.isdir(class_path):
                #image_classes.append(class_name)
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    img = Image.open(image_path).convert('RGB')
                    img_tensor = self.preprocess(img).unsqueeze(0)  # Add batch dimension
                    with torch.no_grad():
                        features = self.model(img_tensor)
                    image_classes.append(class_name)
                    image_features.append(features.squeeze().numpy())  # Remove batch dimension and convert to numpy

        return image_classes, image_features

    def retrieve_similar_class(self, query_image_path):
        query_img = Image.open(query_image_path).convert('RGB')
        query_tensor = self.preprocess(query_img).unsqueeze(0)
        with torch.no_grad():
            query_features = self.model(query_tensor)
        _, indices = self.knn.kneighbors(query_features.squeeze().numpy().reshape(1, -1))
        similar_class_index = indices[0][0]
        return self.image_classes[similar_class_index]
    
    def __call__(self, query_image_path):
        return self.retrieve_similar_class(query_image_path)