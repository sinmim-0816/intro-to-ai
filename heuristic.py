import math

def heuristic(node, goals, coords):
    if not goals:
        return 0 

    min_h = float('inf')
    (x1, y1) = coords[node]
    
    # compare distance to each goal and take the minimum
    if goals in coords:
        (x2, y2) = coords[goals]
        h = math.hypot(x1 - x2, y1 - y2)  # Euclidean distance
        if h < min_h:
            min_h = h
    
    return min_h