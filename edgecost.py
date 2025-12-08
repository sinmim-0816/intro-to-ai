from constant import SEVERITY_PENALTIES
from classifier import IncidentClassifier

# Pathfinding Integration (The Core Cost Function)
def calculate_edge_cost(base_travel_time: float, image_path: str, ml_classifier: IncidentClassifier) -> tuple[float, str]:
    """
    Calculates the predicted travel time (cost) for an edge.
    Returns the predicted cost and the severity string for logging/visualization.
    """
    predicted_severity = ml_classifier.predict_severity(image_path)
    penalty_factor = SEVERITY_PENALTIES.get(predicted_severity, 1.0)
    predicted_cost = base_travel_time * penalty_factor
    
    # Return both the cost (for pathfinding) and the severity (for visualization/logging)
    return predicted_cost, predicted_severity