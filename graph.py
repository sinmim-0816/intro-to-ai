from typing import Tuple, Dict, List, Union,Set,Callable,Any
import os
from collections import deque
from heuristic import heuristic
import heapq
from constant import ACCIDENT_TYPE

class Graph:
    """
    Represents the simplified road network graph extracted from OSM data 
    and populated with base travel times from the traffic file.
    """
    def __init__(self, node_coords: Dict[str, Tuple[float,float]], graph_connections: Dict[str, Dict[str,float]], edge_info):
        """
        Initializes the graph with coordinates and connectivity (adjacency list).
        Connections store {Node_U: {Node_V: Base_Time}}
        """
        self.node_names=list(node_coords.keys())
        self.coords=node_coords
        self.connections=graph_connections
        self.edge_info=edge_info

    def get_edge_info(self, u, v):
        """Get dictionary {way_id, base_time}."""
        return self.edge_info.get(u, {}).get(v, None)

    def get_base_time(self, u, v):
        edge = self.edge_info.get(u, {}).get(v)
        if not edge:
            return float('inf')
        return edge["base_time"]

    def get_way_id(self, u, v):
        edge = self.edge_info.get(u, {}).get(v)
        if not edge:
            return None
        return edge["way_id"]

    def get_all_possible_files()->List[str]:
        """
        Attempts to list all .txt files in the 'test_case' folder for selection. 
        """
        test_case_dir='test_case'
        try:
            files=[f for f in os.listdir(test_case_dir) if f.endswith('.txt')]
        except FileNotFoundError:
            files=[]
        except Exception as e:
            print(f"Error while listing files: {e}")
            files=[]
        return files

GraphConnections = Dict[str, Dict[str, float]]
Path = List[str]
PathMetrics = List[Dict[str, Union[Path, str, None]]] 
InternalGraph=Dict[str, List[Tuple[str,float]]]

def reconstruct_path(start_node:str, goal_node:str, parent_dict:Dict[str,str])->Path:
    """Helper function to reconstruct the path from the parent dictionary."""
    path = []
    current:Union[str,None] = goal_node
    while current is not None:
        path.append(current)
        current = parent_dict.get(current)
    path.reverse()
    
    # Simple check to ensure path starts at the origin
    if path and path[0] == start_node:
        return path
    return []

def path_cost(path: list[str], graph: Graph, accident_segments: dict, accident_multiplier: int) -> float:
    """
    Computes total cost of a path, applying accident multipliers if present.
    """
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]

        # Use edge_info to get both way_id and base_time
        edge = graph.get_edge_info(u, v)
        if not edge:
            return float('inf')

        base_time = edge["base_time"]
        way_id = edge["way_id"]

        if way_id in accident_segments:
            severity = accident_segments[way_id]["severity"]
            multiplier=accident_multiplier*ACCIDENT_TYPE.get(severity.lower(),1.0)
            total += base_time * multiplier
        else:
            total += base_time
    return total

# BFS
def bfs(graph: 'Graph', start: str, goal: str, accident_multiplier: int, max_paths: int = 5, accident_segments: Dict[str, Dict[str, Any]]= None) -> 'PathMetrics':
    """
    A helper function for BFS that returns up to max_paths shortest paths.
    
    NOTE: This BFS finds shortest paths by the number of steps (edges). 
    It is modified to return multiple paths, even if they are slightly longer 
    in steps, up to the max_paths limit.
    """
    # 1. Prepare the graph for easy traversal (though the original graph object 
    #    could also be used directly if it's performant)
    graph_list_format: Dict[str, List[Tuple[str, float]]] = {
        u: [(v, graph.get_base_time(u, v)) for v in graph.connections.get(u, {})]
        for u in graph.connections
    }
    
    queue = deque([[start]])
    found_paths: List[List[str]] = []
    
    # continue until the queue is empty or we have found enough paths
    while queue and len(found_paths) < max_paths:
        # get the next path from the queue (FIFO)
        path = queue.popleft()
        node = path[-1] 
        
        # 2. Check if the goal is reached
        if node == goal:
            found_paths.append(path)
            continue 
        
        # 3. Explore neighbors
        if node in graph_list_format:
            # Explore neighbors in sorted order (as in the original)
            for neighbour, _ in sorted(graph_list_format.get(node, [])):
                # avoid cycles within the same path
                if neighbour not in path: 
                    # append new path including the neighbour
                    new_path = path + [neighbour]
                    queue.append(new_path)

    # 4. Calculate metrics and sort by cost
    path_metrics = [
        {
            "path": p,
            "goal_node": goal,
            "meet_node": None,
            "total_cost": path_cost(p, graph, accident_segments, accident_multiplier),
            "step_count": len(p) - 1
        }
        for p in found_paths
    ]
    
    path_metrics.sort(key=lambda x: x["total_cost"])
    
    return path_metrics

# DFS
def dfs(graph: Graph, start: str, goal: str, accident_multiplier:int, max_paths: int = 5,accident_segments: dict[str, dict[str, Any]] = None) -> PathMetrics:
    """
    A helper function for DFS that returns up to max_paths shortest paths.
    """
    graph_list_format: InternalGraph = {
        u: [(v, graph.get_base_time(u, v)) for v in graph.connections.get(u, {})]
        for u in graph.connections
    }
    # initialize stack with a path staring from the start node
    stack = [[start]] 
    # track parent relationships for GUI and final path reconstruction
    parent = {start: None}
    found_paths=[]

    # continue until all possible paths have been explored
    while stack and len(found_paths)<max_paths:
        # take the most recently added path (LIFO)
        path = stack.pop()
        node = path[-1]
        
        # collect current frontier nodes
        frontier_nodes = [p[-1] for p in stack]

        # check if current node is a goal
        if node == goal:
            found_paths.append(path)
            continue

        # retrieve and sort all neighbors for deterministic expansion order
        neighbours = [n for n, _ in graph_list_format.get(node, [])]
        neighbours = sorted(neighbours)

        # push unvisited neighbors to the stack in reverse order (leftmost expand first)
        for neighbour in reversed(neighbours):
            # prevent cycles by skipping nodes already in the same path
            if neighbour not in path: 
                # record parent for GUI 
                parent[neighbour] = node 

                # create new path and push it to the stack
                new_path = path + [neighbour]
                stack.append(new_path)

    path_metrics=[
        {
            "path": p, 
            "goal_node": goal, 
            "meet_node": None,
            "total_cost": path_cost(p, graph,   accident_segments, accident_multiplier),
            "step_count": len(p) - 1
        } 
        for p in found_paths
    ]
    path_metrics.sort(key=lambda x: x["total_cost"])
    return path_metrics

# Bidirectional Search (BDS)
def path_to_tuple(path:Path)->Tuple[str, ...]:
    """Converts a list path to a tuple for set storage (hashing)."""
    return tuple(path)

def is_path_simple_and_unique(path:Path, found_paths_set:Set[Tuple[str, ...]])->bool:
    """
    Checks if a path is simple (no cycles) and has not been previously found.
    """
    if len(path) != len(set(path)):
        return False
    # Check if the path is unique among those already found
    return path_to_tuple(path) not in found_paths_set

def reconstruct_bidirectional_path(meet_node:str, parent_start: Dict[str,str], parent_goal:Dict[str,str])->Path:
    """
    Reconstructs the full path from start to goal via the meet_node
    using the two parent dictionaries.
    """
    path_start:Path = []
    n: Union[str, None]=meet_node
    while n is not None:
        path_start.append(n)
        n = parent_start.get(n)
    path_start.reverse()

    # trace the path from the meeting node to the goal node
    path_goal_segment:Path = []
    # Start tracing back from the neighbor of the meeting node in the goal search
    n = parent_goal.get(meet_node) 
    while n is not None:
        path_goal_segment.append(n)
        n = parent_goal.get(n)

    return path_start + path_goal_segment


def bds(graph:Graph, start:str, goal:str,accident_multiplier:int, max_path:int=5,accident_segments: dict[str, dict[str, Any]] = None) -> PathMetrics:
    """
    A helper function for BDS that returns up to max_paths shortest paths.
    """
    graph_list_format: InternalGraph = {
        u: [(v, graph.get_base_time(u, v)) for v in graph.connections.get(u, {})]
        for u in graph.connections
    }
    found_paths_set:Set[Tuple[str,...]] = set()
    all_path_metrics:PathMetrics = []
    
    # Start node is a goal check
    if start == goal:
        return [{"path": [start], "goal_node": start, "meet_node": start}]

    # 2. Reverse graph creation (remains unchanged)
    reverse_graph:InternalGraph = {n: [] for n in graph_list_format.keys()}
    for u in graph_list_format:
        for v, cost in graph_list_format[u]:
            reverse_graph.setdefault(v, []).append((u, cost))
    
    reverse_graph.setdefault(goal, [])

    # Initialize frontiers and parent dictionaries (remains unchanged)
    front_start:deque[Tuple[str,Path]] = deque([(start, [start])])
    front_goal:deque[Tuple[str,Path]] = deque([(goal,[goal])])
    parent_start:Dict[str,Union[str,None]] = {start: None}
    parent_goal:Dict[str,Union[str,None]] = {goal:None}

    # 4. Bidirectional search loop
    while front_start and front_goal:
        # Check if max_path limit reached to break the loop early
        if len(all_path_metrics) >= max_path:
            break
            
        # Expand the smaller frontier first
        if len(front_start) <= len(front_goal):
            # FORWARD EXPANSION
            node_s, path_s = front_start.popleft()

            neighbor_s=sorted(graph_list_format.get(node_s,[]))

            for neighbor, cost in neighbor_s:
                # avoid cycles in the same branch
                if neighbor in path_s:
                    continue
                
                # add unvisited neighbor to the frontier
                new_path = path_s + [neighbor]
                
                # Check if max_path limit reached before pushing (optimisation)
                if len(all_path_metrics) >= max_path: 
                    break

                if neighbor not in parent_start:
                    parent_start[neighbor] = node_s
                    front_start.append((neighbor, new_path))

                    # check if frontiers meet
                    if neighbor in parent_goal:
                        path_to_check = reconstruct_bidirectional_path(neighbor, parent_start, parent_goal)
                        # Use the helper function to validate path
                        if path_to_check and is_path_simple_and_unique(path_to_check, found_paths_set):
                            
                            # COLLECT PATH
                            found_paths_set.add(path_to_tuple(path_to_check))
                            all_path_metrics.append({
                                "path": path_to_check,
                                "goal_node": path_to_check[-1],
                                "meet_node": neighbor
                            })
                            
                            if len(all_path_metrics) >= max_path:
                                break 
            
            if len(all_path_metrics) >= max_path:
                continue 

        else:
            # BACKWARD EXPANSION (symmetric to forward)
            node_g, path_g = front_goal.popleft()

            neighbor_g=sorted(reverse_graph.get(node_g,[]))

            for neighbor, cost in neighbor_g:
                if neighbor in path_g:
                    continue
                
                new_path = path_g + [neighbor]
                
                # Check if max_path limit reached before pushing (optimisation)
                if len(all_path_metrics) >= max_path: 
                    break

                if neighbor not in parent_goal:
                    parent_goal[neighbor] = node_g
                    front_goal.append((neighbor, new_path))
                    
                    # check if frontiers meet
                    if neighbor in parent_start:
                        path_to_check = reconstruct_bidirectional_path(neighbor, parent_start, parent_goal)
                        
                        if path_to_check and is_path_simple_and_unique(path_to_check, found_paths_set):
                            
                            # COLLECT PATH
                            found_paths_set.add(path_to_tuple(path_to_check))
                            all_path_metrics.append({
                                "path": path_to_check,
                                "goal_node": path_to_check[-1],
                                "meet_node": neighbor
                            })
                            
                            if len(all_path_metrics) >= max_path:
                                break 
            
            if len(all_path_metrics) >= max_path:
                continue 


    # 5. Final results
    if all_path_metrics:
        # Sort paths by total cost instead of length
        all_path_metrics.sort(
            key=lambda x: path_cost(x["path"], graph, accident_segments,accident_multiplier)
        )

        for pm in all_path_metrics:
            pm["total_cost"] = path_cost(pm["path"], graph, accident_segments,accident_multiplier)
        
    return all_path_metrics

# GBFS
FrontierEntry = Tuple[float, Path]
def gbfs(graph: 'Graph', start: str, goal: str, coords: Dict[str, Tuple[float, float]], accident_multiplier: int, 
         accident_segments: Dict[str, Dict[str, Any]] = None, max_paths: int = 5) -> 'PathMetrics':
    """
    MODIFIED Greedy Best-First Search (GBFS) to return up to max_paths unique paths.
    
    WARNING: GBFS only minimizes heuristic cost (h), not total path cost (g+h), 
    meaning paths found are not guaranteed to be the cost-shortest.
    """
    
    # graph_list_format remains the same
    graph_list_format: Dict[str, List[Tuple[str, float]]] = {
        u: [(v, graph.get_base_time(u, v)) for v in graph.connections.get(u, {})]
        for u in graph.connections
    }
    
    found_paths: List[Path] = []
    found_paths_set: Set[Tuple[str, ...]] = set()
    
    # 1. Initialize frontier with (h_cost, [start_node]) - Storing the full path
    # Removed chronological_count, visited set, and parent dictionary.
    frontier: List[FrontierEntry] = [(heuristic(start, goal, coords), [start])]
    
    # continue search while nodes remain in the frontier
    while frontier and len(found_paths) < max_paths:
        # get path with smallest heuristic value
        h, path = heapq.heappop(frontier) 
        node = path[-1]
        
        # 2. Check if goal is reached
        if node == goal:
            path_tuple = tuple(path)
            # Only add if the exact path hasn't been found already
            if path_tuple not in found_paths_set:
                found_paths.append(path)
                found_paths_set.add(path_tuple)
            
            # Continue exploring other paths if max_paths isn't reached
            continue

        # 3. Explore neighbors
        # No need to check max_path limit here, we continue until all paths in 
        # the heap are explored or max_paths is hit in the while loop condition.
        for neighbour, _ in sorted(graph_list_format.get(node, [])):
            
            # Prevent cycles within the current path only
            if neighbour not in path:
                new_path = path + [neighbour]
                
                # Calculate heuristic for the neighbor
                h_new = heuristic(neighbour, goal, coords)
                
                # push new path to frontier
                heapq.heappush(frontier, (h_new, new_path))

    # 4. Calculate metrics and sort by cost (Final logic remains the same)
    path_metrics = [
        {
            "path": p,
            "goal_node": goal,
            "meet_node": None,
            "total_cost": path_cost(p, graph, accident_segments, accident_multiplier),
            "step_count": len(p) - 1
        }
        for p in found_paths
    ]
    path_metrics.sort(key=lambda x: x["total_cost"])
    
    return path_metrics

# A Star
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

# A star
def a_star(graph: 'Graph', start: str, goal: str, coords: Dict[str, Tuple[float, float]],
                      accident_multiplier: int, accident_segments: Optional[Dict[str, Dict[str, Any]]] = None, 
                      max_path: int = 5) -> 'PathMetrics':
    """
    MODIFIED A* Search to find up to max_path cost-shortest unique paths 
    by removing the single-path (best_g) pruning.
    """

    graph_list_format: Dict[str, List[Tuple[str, float]]] = {
        u: [(v, graph.get_base_time(u, v)) for v in graph.connections.get(u, {})]
        for u in graph.connections
    }
    
    found_paths: List[List[str]] = []
    found_paths_set: Set[Tuple[str, ...]] = set()
    
    chronological_count: int = 0
    expansion_order = {} 
    expand_counter: int = 0
    frontier: List[Tuple[float, int, str, float, Dict[str, str]]] = [
        (heuristic(start, goal, coords), chronological_count, start, 0, {start: None})
    ]

    chronological_count += 1

    while frontier and len(found_paths) < max_path:
        f, chronological, node, g, parent = heapq.heappop(frontier)

        expansion_order[node] = expand_counter
        expand_counter += 1

        # check if a goal is found
        if node == goal:
            current_path = reconstruct_path(start, node, parent)
            path_tuple = tuple(current_path)
            
            if current_path and path_tuple not in found_paths_set:
                found_paths.append(current_path)
                found_paths_set.add(path_tuple)
            continue 

        # generate children
        for neighbour, _ in sorted(graph_list_format.get(node, [])):
            
            if neighbour in parent:
                continue
            
            segment_full_cost = path_cost([node, neighbour], graph, accident_segments, accident_multiplier)
            
            new_g = g + segment_full_cost

            new_f = new_g + heuristic(neighbour, goal, coords)

            new_parent = dict(parent)
            new_parent[neighbour] = node
            heapq.heappush(frontier, (new_f, chronological_count, neighbour, new_g, new_parent))
            chronological_count += 1

    if found_paths:
        found_paths.sort(key=lambda p: path_cost(p, graph, accident_segments, accident_multiplier))
        
        path_metrics = []
        for p in found_paths[:max_path]: # Only process up to max_path elements
            final_total_cost = path_cost(p, graph, accident_segments, accident_multiplier)
            path_metrics.append({
                "path": p,
                "goal_node": p[-1],
                "meet_node": None,
                "total_cost": final_total_cost, # <-- CORRECTED
                "step_count": len(p) - 1
            })
        return path_metrics
    else:
        return []
    
# IDA Star
def ida_star(
    graph: Graph,
    start: str,
    goal: str,
    coords: Dict[str, Tuple[float, float]],
    accident_multiplier: int,
    accident_segments: dict[str, dict[str, Any]] = None,
    max_paths: int = 5
) -> PathMetrics:
    """
    IDA* search returning up to max_paths shortest paths.
    Considers base travel times and accident multipliers.
    """
    path_metrics: List[Dict[str, Any]] = []
    
    # Storage for discovered paths
    found_paths: List[Path] = []
    found_paths_set: Set[Tuple[str, ...]] = set()

    # --------------------------
    # Inner DFS: depth-limited search
    # --------------------------
    def dfs_limited(node: str, g: float, bound: float, path: List[str]):
        nonlocal min_exceed, found_paths
    
        f = g + heuristic(node, goal, coords)

        # If f-cost exceeds limit, we update min_exceed
        if f > bound:
            min_exceed = min(min_exceed, f)
            return

        # If goal is reached → record path
        if node == goal:
            path_tuple = tuple(path)
            if path_tuple not in found_paths_set:
                found_paths.append(list(path))
                found_paths_set.add(path_tuple)
            return

        # Explore neighbors
        for neighbour in graph.connections.get(node, {}):
            if neighbour in path:  # avoid cycles
                continue

            # Base cost only; accident cost handled later in path_cost()
            segment_full_cost = path_cost([node,neighbour], graph, accident_segments, accident_multiplier)
            dfs_limited(neighbour, g + segment_full_cost, bound, path + [neighbour])
            
    # --------------------------
    # Iterative deepening loop
    # --------------------------
    bound = heuristic(start, goal, coords)

    while len(found_paths) < max_paths:

        min_exceed = float('inf')

        dfs_limited(start, 0, bound, [start])

        # If no better bound found, stop
        if min_exceed == float('inf'):
            break

        bound = min_exceed  # Increase search limit

    # --------------------------
    # Convert found paths to metrics
    # --------------------------
    if found_paths:
        found_paths.sort(key=lambda p: path_cost(p, graph, accident_segments, accident_multiplier))
        for p in found_paths[:max_paths]:
            final_total_cost = path_cost(p, graph, accident_segments, accident_multiplier)
            path_metrics.append({
                "path": p,
                "goal_node": goal, 
                "meet_node": None,
                "total_cost": final_total_cost, 
                "step_count": len(p) - 1
            })
        return path_metrics

    return []

def find_path_algorithm(
    graph: Graph, 
    start: str, 
    end: str, 
    algorithm_choice: str, 
    accident_segments: dict[str, dict[str, Any]],
    accident_multiplier: int
) -> List[Dict[str, Any]]:
    """
    Executes the chosen pathfinding algorithm and returns all found paths
    with total cost and step count for each.
    """
    cleaned_algorithm_choice = algorithm_choice.strip()
    coords = graph.coords
    path_metrics: PathMetrics = []

    # Select algorithm
    if cleaned_algorithm_choice == "A*":
        path_metrics = a_star(graph, start, end, coords,accident_multiplier=accident_multiplier,accident_segments=accident_segments)
    elif cleaned_algorithm_choice == "IDA*":
        path_metrics = ida_star(graph, start, end, coords,accident_multiplier=accident_multiplier, accident_segments=accident_segments)
    elif cleaned_algorithm_choice == "GBFS":
        path_metrics = gbfs(graph, start, end,coords,accident_multiplier=accident_multiplier,accident_segments=accident_segments)
    elif cleaned_algorithm_choice == "BFS":
        path_metrics = bfs(graph, start, end,accident_multiplier=accident_multiplier,accident_segments=accident_segments)
    elif cleaned_algorithm_choice == "DFS":
        path_metrics = dfs(graph, start, end,accident_multiplier=accident_multiplier,accident_segments=accident_segments)
    elif cleaned_algorithm_choice == "BDS":
        path_metrics = bds(graph, start, end,accident_multiplier=accident_multiplier,accident_segments=accident_segments)
    else:
        print(f"Error: Unknown algorithm choice: {algorithm_choice}")
        return []

    # Prepare all paths info
    all_paths_info = []
    for pm in path_metrics:
        path = pm.get("path", [])
        if not path:
            continue

        # Use cost returned from algorithm instead of recalculating
        total_cost = pm.get("total_cost", 0)
        steps_count = pm.get("step_count", len(path) - 1)

        all_paths_info.append({
            "path": path,
            "total_cost": total_cost,
            "steps_count": steps_count,
            "isBest": False
        })

    # Mark the best path by total cost
    if all_paths_info:
        best_index = min(range(len(all_paths_info)), key=lambda i: all_paths_info[i]["total_cost"])
        all_paths_info[best_index]["isBest"] = True

    return all_paths_info