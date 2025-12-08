import torch

# CONSTANTS AND UTILITY SETUP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224 

# Define severity weights
SEVERITY_PENALTIES = {
    'no_accident': 1.0,   
    'minor': 1.25,        # 25% penalty
    'intermediate': 1.75, # 75% penalty
    'major': 2.5          # 150% penalty
}

SEVERITY_CLASSES = ['intermediate', 'major', 'minor'] 