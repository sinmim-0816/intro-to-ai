from constant import DEVICE, SEVERITY_CLASSES
import torch
import torch.nn.functional as F
from PIL import Image
import os
from model_definitions import CNNModel, TransferVGG, TransferEfficient
from torchvision import transforms
from typing import Dict,Tuple
from captum.attr import LayerGradCam
import cv2
import numpy as np

# INCIDENT CLASSIFIER CLASS (ML Wrapper)
class IncidentClassifier:
    """
    Handles loading the two trained models (Accident/Severity) 
    and performing two-stage image classification based on the model_choice.
    """
    def __init__(self, accident_choice: str, severity_choice: str):
        print(f"Loading Incident Classifier models ({accident_choice}) and Severity Model ({severity_choice}) on {DEVICE}...")

        MODEL_DIR="models"
        
        # --- Define Normalization Constants ---
        TRANSFER_NORMALIZATION = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        # Assuming these are the correct mean/std used during Custom CNN training
        CUSTOM_NORMALIZATION=transforms.Normalize(
            mean=[0.46021583676338196, 0.4511672258377075, 0.45412397384643555],
            std=[0.24809545278549194, 0.2457689493894577, 0.2474203258752823]
        )
        
        # Define Model Configuration Map
        MODEL_CONFIGS = {
            'EfficientNet': {
                'CLASS': TransferEfficient, 'IMG_SIZE': 224, 'NORM': TRANSFER_NORMALIZATION,
                'PATH_A': os.path.join(MODEL_DIR, 'accident_transferefficient_unfreeze_cnn_pytorch.pth'),
                'PATH_B': os.path.join(MODEL_DIR,'severity_transferefficient_unfreeze_cnn_pytorch.pth')
            },
            'VGG16': {
                'CLASS': TransferVGG, 'IMG_SIZE': 128, 'NORM': TRANSFER_NORMALIZATION,
                'PATH_A': os.path.join(MODEL_DIR,'accident_transfervgg_unfreeze_cnn_pytorch.pth'),
                'PATH_B': os.path.join(MODEL_DIR,'severity_transfervgg_unfreeze_cnn_pytorch.pth')
            },
            'Custom CNN': {
                'CLASS': CNNModel, 'IMG_SIZE': 128, 'NORM': CUSTOM_NORMALIZATION, 
                'PATH_A': os.path.join(MODEL_DIR,'accident_custom_final_cnn_pytorch.pth'),
                'PATH_B': os.path.join(MODEL_DIR,'severity_custom_final_cnn_pytorch.pth')
            }
        }
        # Validate Choices
        if accident_choice not in MODEL_CONFIGS:
            raise ValueError(f"Unknown accident model: {accident_choice}")
        if severity_choice not in MODEL_CONFIGS:
            raise ValueError(f"Unknown severity model: {severity_choice}")
        
        self.config_a = MODEL_CONFIGS[accident_choice]
        self.config_b = MODEL_CONFIGS[severity_choice]
        # --- Define Model-Specific Transforms ---
        
        # 1. Transform for Accident Model (A)
        self.transform_a = transforms.Compose([
            transforms.Resize((self.config_a['IMG_SIZE'], self.config_a['IMG_SIZE'])),
            transforms.ToTensor(),
            self.config_a['NORM']
        ])
        # 2. Transform for Severity Model (B)
        self.transform_b = transforms.Compose([
            transforms.Resize((self.config_b['IMG_SIZE'], self.config_b['IMG_SIZE'])),
            transforms.ToTensor(),
            self.config_b['NORM']
        ])
        # --- Load Model A (Accident/No Accident) ---
        Class_A = self.config_a['CLASS']
        path_a = self.config_a['PATH_A']
        self.model_a = Class_A(num_classes=2) 
        self._load_model_weights(self.model_a, path_a)
        
        # --- Load Model B (Severity: 3 Classes) ---
        Class_B = self.config_b['CLASS']
        path_b = self.config_b['PATH_B']
        self.model_b = Class_B(num_classes=3)
        self._load_model_weights(self.model_b, path_b)

    def _load_model_weights(self, model, path):
        """Helper function to load state dictionary safely."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model weights file not found: {path}. Ensure it is in the same directory.")
            
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set to evaluation mode for inference

    def _preprocess(self, image_path: str, model_type: str = 'A') -> torch.Tensor:
        """Loads and preprocesses an image for model input."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        # self.transform is now correctly assigned in __init__ using the right IMG_SIZE
        if model_type == 'A':
            transform = self.transform_a
        elif model_type == 'B':
            transform = self.transform_b
        else:
            raise ValueError("model_type must be 'A' (Accident) or 'B' (Severity).")
        
        return transform(image).unsqueeze(0).to(DEVICE)

    def predict_severity(self, image_path: str) -> Tuple[str, Dict[str, float]]:
        """
        Performs the two-stage prediction, using the correct preprocessing for each stage.
        Returns the resulting class and a dictionary of probabilities.
        """
        
        with torch.no_grad():
            # --- Stage 1: Binary Prediction (using Model A's transform) ---
            input_tensor_a = self._preprocess(image_path, model_type='A')
            output_a = self.model_a(input_tensor_a)
            prob_a = F.softmax(output_a, dim=1).cpu().numpy()[0]
            
            # Binary classification mapping
            ACCIDENT_CLASSES = ['no_accident', 'accident']
            accident_probs = dict(zip(ACCIDENT_CLASSES, prob_a.tolist()))
            
            # Check the index of the maximum probability
            predicted_a_index = prob_a.argmax() 
            
            # If Model A predicts "No Accident" (index 0)
            if ACCIDENT_CLASSES[predicted_a_index] == 'no_accident':
                # Return 'no_accident' and the binary probabilities
                return 'no_accident', accident_probs
            
            # --- Stage 2: Severity Prediction (using Model B's transform) ---
            input_tensor_b = self._preprocess(image_path, model_type='B')
            output_b = self.model_b(input_tensor_b)
            prob_b = F.softmax(output_b, dim=1).cpu().numpy()[0]
            
            # Severity classification mapping
            severity_probs = dict(zip(SEVERITY_CLASSES, prob_b.tolist()))
            severity_index = prob_b.argmax()
            severity_class = SEVERITY_CLASSES[severity_index]
            
            # Return the predicted severity class and its probabilities
            return severity_class, severity_probs
        
    def generate_grad_cam(self, image_path:str, model_type:str='B', target_class: int = None)-> Image.Image:
        # Preprocess
        input_tensor=self._preprocess(image_path,model_type).to(DEVICE)
        
        # Select Model
        if model_type == 'A':
            model = self.model_a
        elif model_type == 'B':
            model = self.model_b
        else:
            raise ValueError("model_type must be 'A' or 'B'")
        # Forward pass to get prediction if target_class not provided
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            predicted_class = probs.argmax(dim=1).item()
        
        if target_class is None:
            target_class = predicted_class

        # Use last conv layer for Grad-CAM
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'features'):
            target_layer = model.backbone.features[-1]
        else:
            raise ValueError("Cannot find a convolutional layer for Grad-CAM.")

        # Create Grad-CAM object
        grad_cam = LayerGradCam(model, target_layer)

        # Compute attribution
        attribution = grad_cam.attribute(input_tensor, target=target_class)

        # Convert to numpy heatmap
        heatmap = attribution.squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max() + 1e-8

        # Overlay heatmap on original image
        orig_image = Image.open(image_path).convert('RGB').resize((input_tensor.size(3), input_tensor.size(2)))
        
        heatmap_cv = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_cv = cv2.cvtColor(heatmap_cv, cv2.COLOR_BGR2RGB)
        heatmap_cv = cv2.resize(heatmap_cv, (orig_image.width, orig_image.height))

        heatmap_pil = Image.fromarray(heatmap_cv).convert('RGBA')
        orig_image = orig_image.convert('RGBA')

        overlayed = Image.blend(orig_image, heatmap_pil, alpha=0.5)

        return overlayed