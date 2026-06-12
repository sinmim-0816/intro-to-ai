from typing import Tuple, Dict, List, Union,Set,Callable
import os
from collections import deque
from heuristic import heuristic
import heapq

class Graph:
    """
    Represents the simplified road network graph extracted from OSM data 
    and populated with base travel times from the traffic file.
    """
    def __init__(self, node_coords: Dict[str, Tuple[float,float]], graph_connections: Dict[str, Dict[str,float]]):
        """
        Initializes the graph with coordinates and connectivity (adjacency list).
        Connections store {Node_U: {Node_V: Base_Time}}
        """
        self.node_names=list(node_coords.keys())
        self.coords=node_coords
        self.connections=graph_connections
        
    def get_all_nodes(self) -> List[str]:
        """Returns a list of all named nodes."""
        return self.node_names
    
    def get_base_time(self, u: str, v: str) -> float:
        """Retrieves the base travel time cost from node u to node v."""
        return self.connections.get(u, {}).get(v, float('inf'))
    
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

# BFS
def bfs(graph_connections: GraphConnections, start: str, goal: str, max_paths: int = 5) -> PathMetrics:
    """
    A helper function for BFS that returns up to max_paths shortest paths.
    """
    graph_list_format: InternalGraph = {u: [(v, cost) for v, cost in connections.items()] for u, connections in graph_connections.items()}
    
    queue = deque([[start]])
    found_paths=[]
    min_length_found=float('inf')
    # record parent relationships for GUI
    parent = {start: None}

    # continue until there are no more nodes to explore
    while queue and len(found_paths)<max_paths:
        # get the next path from the queue (FIFO)
        path = queue.popleft()
        node = path[-1] 
        
        current_length=len(path)

        if current_length > min_length_found and min_length_found != float('inf'):
            break
        
        # check if the goal is reached
        if node == goal:
            if min_length_found == float('inf'):
                min_length_found=current_length
            if current_length == min_length_found:
                found_paths.append(path)
        
        # explore all unvisited neighbors
        if len(found_paths)<max_paths or min_length_found==float('inf'):
            if node in graph_list_format:
                for neighbour, _ in sorted(graph_list_format.get(node, [])):
                    # avoid cycles within the same path
                    if neighbour not in path: 
                        parent[neighbour] = node
                        # append new path including the neighbour
                        new_path = path + [neighbour]
                        queue.append(new_path)
                        

    return [
        {
            "path": p, 
            "goal_node": goal, 
            "meet_node": None
        } 
        for p in found_paths
    ]

# DFS
def dfs(graph_connections: GraphConnections, start: str, goal: str, max_paths: int = 5) -> PathMetrics:
    """
    A helper function for DFS that returns up to max_paths shortest paths.
    """
    graph_list_format: InternalGraph = {u: [(v, cost) for v, cost in connections.items()] for u, connections in graph_connections.items()}
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

    return [
        {
            "path": p, 
            "goal_node": goal, 
            "meet_node": None # DFS is unidirectional, no meeting node
        } 
        for p in found_paths
    ]
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

def bds(graph_connections:GraphConnections, start:str, goal:str, max_path:int=5) -> PathMetrics:
    """
    A helper function for BDS that returns up to max_paths shortest paths.
    """
    graph_list_format: InternalGraph = {u: [(v, cost) for v, cost in connections.items()] for u, connections in graph_connections.items()}
    
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
                                break # Break the inner loop (neighbors)
            
            if len(all_path_metrics) >= max_path:
                continue 


    # 5. Final results
    if all_path_metrics:
        # Sort paths by length (number of edges) to return the k *shortest* paths
        all_path_metrics.sort(key=lambda x: len(x["path"]))
        
    return all_path_metrics

# GBFS
def gbfs(graph_connections: GraphConnections, start: str, goal: str, coords: Dict[str, Tuple[float, float]], max_paths: int = 5) -> PathMetrics:
    """
    A helper function for GBFS that returns up to max_paths shortest paths.
    """
    graph_list_format: InternalGraph = {u: [(v, cost) for v, cost in connections.items()] for u, connections in graph_connections.items()}
    
    found_paths:List[Path]=[]
    chronological_count:int = 0
    found_paths_set:Set[Tuple[str,...]] = set()
    
    visited:Set[str] = set()
    
    # Frontier: Store (h_cost, node, chronological)
    # The full path list is removed for efficiency.
    frontier:List[Tuple[float,str]] = [(heuristic(start, goal, coords), start, chronological_count)]
    parent = {start: None}
    
    chronological_count += 1

    # continue search while nodes remain in the frontier
    while frontier and len(found_paths) < max_paths:
        # get node with smallest heuristic value
        h, node, chronological = heapq.heappop(frontier) 
        
        # Skip if already visited/expanded
        if node in visited:
            continue
            
        # Mark node as visited/expanded
        visited.add(node)

        # check if goal is reached
        if node == goal:
            
            # The path is reconstructed using the parent dictionary
            full_path = reconstruct_path(start, node, parent) 
            
            # Check for path uniqueness
            path_tuple = tuple(full_path)
            if full_path and path_tuple not in found_paths_set:
                
                path_cost = len(full_path) - 1
                
                # Store the path and update the set
                found_paths.append(full_path)
                found_paths_set.add(path_tuple)
                
                if len(found_paths) >= max_paths:
                    break

        # explore neighbors in sorted order
        for neighbour, _ in sorted(graph_list_format.get(node, [])):
            
            # Skip neighbors already visited
            if neighbour in visited:
                continue
            
            # Check for max_path limit BEFORE pushing new nodes
            if len(found_paths) >= max_paths:
                break
            
            # record parent for path reconstruction
            parent[neighbour] = node
            
            # push neighbor to frontier based on heuristic value
            heapq.heappush(frontier, (heuristic(neighbour, goal, coords), neighbour, chronological_count))
            chronological_count += 1

    return [
        {
            "path": p, 
            "goal_node": p[-1], 
            "meet_node": None
        } 
        for p in found_paths
    ]
# A Star
def a_star(graph_connections: GraphConnections, start: str, goal: str, coords: Dict[str, Tuple[float, float]], max_path: int = 5) -> PathMetrics:
    """
    A helper function for A star that returns up to max_paths shortest paths.
    """
    graph_list_format: InternalGraph = {u: [(v, cost) for v, cost in connections.items()] for u, connections in graph_connections.items()}
    
    found_paths:List[Path]=[]
    found_paths_set: Set[Tuple[str,...]] = set()
    # Initialize counters and tracking structures
    chronological_count:int = 0  
    expansion_order = {} 
    expand_counter:int = 0

    # dictionary to store the lowest g cost found so far for each node
    best_g:Dict[str,float] = {start: 0}

    # (f, parent_expansion_order, chronological, node, g, parent)
    frontier:List[Tuple[float,float,str,Dict[str,str]]] = [(heuristic(start, goal, coords), -1, chronological_count, start, 0, {start: None})]

    # increment for the next node generated
    chronological_count += 1

    while frontier and len(found_paths)<max_path:
        # pop the node with the lowest f value
        f, parent_exp, chronological, node, g, parent = heapq.heappop(frontier)

        # recode expansion order
        expansion_order[node] = expand_counter
        expand_counter += 1

        # check if a goal is found
        if node == goal:
            current_path = reconstruct_path(start, node, parent)
            path_tuple=tuple(current_path)
            if current_path and path_tuple not in found_paths_set:
                found_paths.append(current_path)
                found_paths_set.add(path_tuple)
                if len(found_paths) >= max_path:
                    break
                

        # generate children after expansion order assigned
        for neighbour, cost in sorted(graph_list_format.get(node, [])):
            # skip if neighbour is already in the path
            if neighbour in parent:
                continue
            
            # count new path cost
            new_g = g + cost

            # skip if already found a cheaper path to neighbour
            if new_g >= best_g.get(neighbour, float('inf')):
                continue

            # record best g cost and count total estimated cost (f = g + h)
            new_f = new_g + heuristic(neighbour, goal, coords)

            # create new parent dictionary
            new_parent = dict(parent)
            new_parent[neighbour] = node

            # push neighbour to frontier
            heapq.heappush(frontier, (new_f, expansion_order[node], chronological_count, neighbour, new_g, new_parent))
            chronological_count += 1

    # yield failure if no path found
    if found_paths:
        found_paths.sort(key=lambda p: len(p)) 
        return [
            {
                "path": p, 
                "goal_node": p[-1], 
                "meet_node": None
            } 
            for p in found_paths
        ]
    else:
        return []
# IDA Star
def ida_star(graph_connections: GraphConnections, start: str, goal: str, coords: Dict[str, Tuple[float, float]], max_path: int = 5) -> PathMetrics:
    """
    A helper function for IDA star that returns up to max_paths shortest paths.
    """
    graph_list_format: InternalGraph = {u: [(v, cost) for v, cost in connections.items()] for u, connections in graph_connections.items()}
    
    def dfs_limited(node, g, bound, path, found_paths, found_paths_set, max_path, graph_list_format):
        """
        Recursive depth-limited DFS used by IDA*.
        Returns the smallest f-cost that exceeded the current bound, 
        or a special string signal.
        """
        
        f = g + heuristic(node, goal, coords)
        
        # recursive depth-limited DFS used by IDA*
        if f > bound:
            return f
        
        # goal check
        if node == goal:
            full_path = path
            path_tuple = tuple(full_path)
            
            # Check for path uniqueness
            if path_tuple not in found_paths_set:
                # Path Cost (g is the path cost in A* algorithms)
                path_cost = g 
                
                found_paths.append((path_cost, full_path))
                found_paths_set.add(path_tuple)
                
                # Check if we hit the limit
                if len(found_paths) >= max_path:
                    return "MAX_FOUND" 
        
        # generate and sort neighbors
        neighbors_data = []
        for neighbour, cost in graph_list_format.get(node, []):
            if neighbour not in path: 
                g_new = g + cost
                h_new = heuristic(neighbour, goal, coords)
                f_new = g_new + h_new

                # store multiple tie-breaking values for consistent ordering
                neighbors_data.append((f_new, h_new, neighbour, g_new, cost)) 
                
        # sort neighbors by (f, h, node) for tie-breaking
        neighbors_data.sort(key=lambda x: (x[0], x[1], x[2])) 
        
        min_bound = float('inf')
        
        # recursive exploration of neighbors
        for f_new, h_new, neighbour, g_new, cost in neighbors_data:
            # 6. Pass all tracking variables recursively
            result = dfs_limited(
                neighbour, g_new, bound, path + [neighbour], found_paths, found_paths_set, max_path,graph_list_format
            )

            # Handle termination and next bound calculation
            if result == "MAX_FOUND":
                return "MAX_FOUND"
            elif isinstance(result, (int, float)) and result < min_bound:
                # record smallest f value exceeding the bound
                min_bound = result 
                
        # Return the smallest exceeding bound (min_bound) or MAX_FOUND/None
        return min_bound

    # Outer loop for Iterative Deepening
    bound = heuristic(start, goal, coords) 
    parent = {start: None}
    
    # K-Path Tracking Initialization
    found_paths:List[Tuple[float,Path]] = []
    found_paths_set = set()

    while True:
        # Check limit before starting a new iteration
        if len(found_paths) >= max_path:
            break
            
        # 1. Call dfs_limited with all tracking data
        result = dfs_limited(
            start, 0, bound, [start], found_paths, found_paths_set, max_path,graph_list_format
        )
        
        # 2. Handle termination conditions
        if result == "MAX_FOUND":
            break
            
        # no more nodes to explore
        if result == float('inf'):
            break
            
        # 3. increase bound for next iteration
        bound = result 
        
    # Final success/failure yield
    if found_paths:
        found_paths.sort(key=lambda x:x[0])
        return [
            {
                "path": path, 
                "goal_node": path[-1], 
                "meet_node": None 
            } 
            for path in found_paths
        ]
    else:
        return []

def calculate_path_cost(path: List[str], cost_func: Callable[[str, str], float]) -> float:
    """Calculates the total cost of a path using the provided cost function."""
    total_cost=0.0
    for i in range(len(path) -1):
        u=path[i]
        v=path[i+1]
        cost=cost_func(u,v)
        # Path segment is not connected (should not happen for a valid path)
        if cost==float('inf'):
            return float('inf')
        
        total_cost+=cost
    return total_cost

def find_path_algorithm(graph: Graph, start: str, end: str, algorithm_choice: str, cost_func: Callable[[str, str], float]) -> Tuple[Union[List[str], None], float, int]:
    """Executes the chosen pathfinding algorithm and formats the result.
    """
    graph_connections=graph.connections
    coords=graph.coords
    path_metrics:PathMetrics=[]
    
    if algorithm_choice in {"AS", "A*"}:
        path_metrics = a_star(graph_connections, start, end, coords)
    elif algorithm_choice in {"IDAS", "IDA*"}:
        path_metrics = ida_star(graph_connections, start, end, coords)
    elif algorithm_choice == 'GBFS':
        path_metrics = gbfs(graph_connections, start, end, coords)
    elif algorithm_choice == 'BFS':
        path_metrics = bfs(graph_connections, start, end)
    elif algorithm_choice == 'DFS':
        path_metrics = dfs(graph_connections, start, end)
    elif algorithm_choice == 'BDS':
        path_metrics = bds(graph_connections, start, end)
    else:
        print(f"Error: Unknown algorithm choice: {algorithm_choice}")
        return None, 0.0, 0
    
    if path_metrics:
        best_result=path_metrics[0]
        path:Path=best_result["path"]
        
        total_cost=calculate_path_cost(path,cost_func)
        steps_count:int = best_result.get("step_count", len(path)-1)
        return path, total_cost,steps_count
    else:
        return None, 0.0,0
    
 
