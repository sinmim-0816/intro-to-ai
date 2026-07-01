import torch

# CONSTANTS AND UTILITY SETUP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define severity weights
ACCIDENT_TYPE = { 
    'minor': 1.0,        
    'intermediate': 2.0, 
    'major': 3.0
}

SEVERITY_CLASSES = ['Intermediate', 'Major', 'Minor']

SEVERITY_DISPLAY_ORDER = ['Major', 'Intermediate', 'Minor']