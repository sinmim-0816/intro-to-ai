from constant import DEVICE, SEVERITY_CLASSES, torch
import torch.nn.functional as F
from PIL import Image
import os
from model_definitions import CNNModel, TransferVGG, TransferEfficient

# INCIDENT CLASSIFIER CLASS (ML Wrapper)
class IncidentClassifier:
    """
    Handles loading the two trained models (Accident/Severity) 
    and performing two-stage image classification based on the model_choice.
    """
    def __init__(self, model_choice: str):
        print(f"Loading Incident Classifier models ({model_choice}) on {DEVICE}...")

        self.model_choice = model_choice
        
        # Define Standard Image Transformations
        # self.transform = transforms.Compose([
        #     transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        MODEL_DIR = "models"
        # Determine File Paths and Model Class based on Frontend selection 
        if model_choice == 'EfficientNet':
            path_a = os.path.join(MODEL_DIR, 'accident_transferefficient_unfreeze_cnn_pytorch.pth')
            path_b = os.path.join(MODEL_DIR,'severity_transferefficient_unfreeze_cnn_pytorch.pth')
            Current_ML_Class = TransferEfficient
        elif model_choice == 'VGG16':
            path_a = os.path.join(MODEL_DIR,'accident_transfervgg16_cnn_pytorch.pth')
            path_b = os.path.join(MODEL_DIR,'severity_transfervgg16_cnn_pytorch.pth')
            Current_ML_Class = TransferVGG
        elif model_choice == 'Custom CNN':
            path_a = os.path.join(MODEL_DIR,'accident_custom_final_cnn_pytorch.pth')
            path_b = os.path.join(MODEL_DIR,'severity_custom_final_cnn_pytorch.pth')
            Current_ML_Class = CNNModel
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")

        # Load Model A (Binary: Accident/No Accident)
        self.model_a = Current_ML_Class(num_classes=2) 
        # Load the trained weights (.pth file)
        self._load_model_weights(self.model_a, path_a)
        
        # Load Model B (Severity: 3 Classes) 
        # Instantiate the model architecture
        self.model_b = Current_ML_Class(num_classes=3)
        # Load the trained weights (.pth file)
        self._load_model_weights(self.model_b, path_b)
        
        print(f"Models for {model_choice} loaded successfully.")

    def _load_model_weights(self, model, path):
        """Helper function to load state dictionary safely."""
        if not os.path.exists(path):
            # Crucial for debugging file issues
            raise FileNotFoundError(f"Model weights file not found: {path}. Ensure it is in the same directory.")
            
        # The line that loads the trained weights into the instantiated model:
        # map_location ensures it loads correctly regardless of the training environment (GPU/CPU)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set to evaluation mode for inference

    def _preprocess(self, image_path: str) -> torch.Tensor:
        """Loads and preprocesses an image for model input."""
        if not os.path.exists(image_path):
             raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0).to(DEVICE)


    def predict_severity(self, image_path: str) -> str:
        """
        Performs the two-stage prediction:
        Stage 1: Predict if an accident occurred (Model A).
        Stage 2: If yes, predict the severity (Model B).
        """
        input_tensor = self._preprocess(image_path)
        
        with torch.no_grad():
            # Stage 1: Binary Prediction
            output_a = self.model_a(input_tensor)
            prob_a = F.softmax(output_a, dim=1)
            _, predicted_a = torch.max(prob_a, 1)

            # If Model A predicts "No Accident"
            if predicted_a.item() == 0: 
                return 'no_accident'
            
            # Stage 2: Severity Prediction
            output_b = self.model_b(input_tensor)
            prob_b = F.softmax(output_b, dim=1)
            _, predicted_b_index = torch.max(prob_b, 1)
            
            severity_class = SEVERITY_CLASSES[predicted_b_index.item()]
            return severity_class