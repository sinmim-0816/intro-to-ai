import streamlit as st
import time
import pandas as pd
import pydeck as pdk
import xml.etree.ElementTree as ET
import os
from typing import List, Tuple, Optional, Dict, Set, Union
from backend import IncidentClassifier, calculate_edge_cost
from graph import Graph, find_path_algorithm
# Read Osm File
TRAFFIC_BASE_TIMES: Dict[Tuple[str,str],float]={}
# Store the nodes defined in the .txt file
DEFINITIVE_NODE_NAMES:Set[str]=set()

def read_osm_file(filename: str='map.osm')->Optional[str]:
    """
    Reads the content of the OSM file from the local disk.
    """
    try:
        if not os.path.exists(filename):
            st.error(f"FATAL ERROR: OSM file '{filename}' not found. Please ensure it is exists inside the directory.")
            return None
        
        with open(filename,"r",encoding='utf-8') as f:
            return f.read()
    
    except Exception as e:
        st.error(f"Error reading the local OSM file: {e}")
        return None

# Read the text file 
def read_traffic_data(filename: str='tc1.txt')->Tuple[Dict[Tuple[str, str], float], Set[str], Dict[str, str]]:
    """
    Reads the custom traffic data file to extract base travel times.
    """
    filename = os.path.join("test_case", filename)
    global TRAFFIC_BASE_TIMES
    global DEFINITIVE_NODE_NAMES
    traffic_data: Dict[Tuple[str, str], float] = {}
    node_id_to_name: Dict[str, str] = {}
    metadata: Dict[str, Union[str, List[str]]] = {'START': '', 'GOAL': [], 'ACCIDENT_MULTIPLIER': 1}
    
    TRAFFIC_BASE_TIMES.clear()
    DEFINITIVE_NODE_NAMES.clear()
    
    # Check if file exists, if not, print the warning you identified and return empty data.
    if not os.path.exists(filename):
        st.warning(f"Traffic data file '{filename}' not found. Using the mock times for all connections.")
        # Ensure the function returns the full expected tuple structure when file is missing
        return traffic_data, DEFINITIVE_NODE_NAMES, node_id_to_name,metadata

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines=f.readlines()
            
        mode=None
        for line in lines:
            line=line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line == '[NODES]':
                mode='NODES'
                continue
            
            if line.startswith('[') and line.endswith(']'):
                mode='TRAFFIC'
                continue
            
            # Parse the node id and node name
            if mode=='NODES':
                parts=line.split(',')
                if len(parts) >=4:
                    node_id=parts[0].strip()
                    node_name=parts[3].strip()
                    node_id_to_name[node_id]=node_name
                    DEFINITIVE_NODE_NAMES.add(node_name)
            
            # Parse Traffic Times
            elif mode == 'WAYS':
                parts=[p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    try:
                        id_u=parts[1]
                        id_v=parts[2]
                        time_cost=float(parts[5])
                        name_u=node_id_to_name.get(id_u)
                        name_v=node_id_to_name.get(id_v)
                        
                        if name_u and name_v:
                            traffic_data[(name_u,name_v)]=time_cost
                    
                    except (ValueError, IndexError) as e:
                        pass
            elif mode == 'META':
                parts=[p.strip() for p in line.split(',')]
                key=parts[0]
                if key=='START' and len(parts)>=2:
                    start_id=parts[1]
                    metadata['START']=node_id_to_name.get(start_id,start_id)
                elif key=='GOAL' and len(parts)>=2:
                    goal_id=parts[1]
                    metadata['GOAL']=node_id_to_name.get(goal_id,goal_id)
                elif key=='ACCIDENT_MULTIPLIER' and len(parts)>=2:
                    try:
                        metadata['ACCIDENT_MULTIPLIER']=float(parts[1])
                    except ValueError:
                        pass
                    
    except Exception as e:
        st.error(f"Error parsing traffic data file: {e}")
        # Return empty data structure on parsing failure
        return traffic_data, DEFINITIVE_NODE_NAMES, node_id_to_name

    TRAFFIC_BASE_TIMES.update(traffic_data) 

    return traffic_data, DEFINITIVE_NODE_NAMES,node_id_to_name,metadata

# OSM Parsing Logic
def parse_osm_data(osm_content:str)->Tuple[Dict[str,Tuple[float,float]], Dict[str,Dict[str,float]]]:
    """
    Parses OSM XML content to extract named nodes and simplified graph connections.
    Includes logic to find coordinates for named landmarks defined as <way> elements.
    Returns: (named_node_coords, graph_connections)
    """
    root=ET.fromstring(osm_content)
    node_coords_by_id={}
    
    # 1. First Pass: Get all node coordinates by their OSM ID
    for node in root.findall('node'):
        node_id=node.get('id')
        lat_str = node.get('lat')
        lon_str = node.get('lon')
        if lat_str and lon_str:
            try:
                lat=float(lat_str)
                lon=float(lon_str)
                node_coords_by_id[node_id]=(lat, lon)
            except ValueError:
                continue

    # 2. Second Pass: Find coordinates for DEFINITIVE_NODE_NAMES
    named_node_coords={}
    
    # 2a. Check for landmarks defined as named <node> elements
    for node in root.findall('node'):
        name_tag=node.find("tag/[@k='name']")
        if name_tag is not None:
            name=name_tag.get('v')
            if name in DEFINITIVE_NODE_NAMES and name not in named_node_coords:
                node_id=node.get('id')
                if node_id in node_coords_by_id:
                    named_node_coords[name]=node_coords_by_id[node_id]

    
    # 2b. Check for landmarks defined as named <way> elements (like Masjid Bandaraya Kuching)
    # This associates the landmark name with the FIRST node in the way boundary.
    for way in root.findall('way'):
        name_tag=way.find("tag/[@k='name']")
        if name_tag is not None:
            name=name_tag.get('v')
            # If the name is one of our definitive points and we haven't found it yet
            if name in DEFINITIVE_NODE_NAMES and name not in named_node_coords:
                node_refs=way.findall('nd')
                if node_refs:
                    ref_id = node_refs[0].get('ref')
                    if ref_id in node_coords_by_id:
                        named_node_coords[name]=node_coords_by_id[ref_id]


    loaded_nodes_count = len(named_node_coords)
    if loaded_nodes_count == 0:
        st.error(f"Parsing Error: 0 definitive nodes loaded from OSM file. Check if {DEFINITIVE_NODE_NAMES} exist as named <node> or <way> elements.")
        return named_node_coords, {}

    st.success(f"Successfully loaded {loaded_nodes_count} named nodes from OSM file.")
    graph_connections: Dict[str, Dict[str, float]] = {name: {} for name in named_node_coords.keys()}
    
    valid_node_names = set(named_node_coords.keys())
    
    for (name_u, name_v), cost in TRAFFIC_BASE_TIMES.items():
        # Only add the edge if BOTH the start and end nodes were successfully mapped 
        # to coordinates from the OSM file.
        if name_u in valid_node_names and name_v in valid_node_names:
            if cost != float('inf'):
                # Add the connection and its base cost
                graph_connections[name_u][name_v] = cost
    
    return named_node_coords, graph_connections

# Graph Constant Variables
PARSED_NODE_COORDS={}
PARSED_GRAPH_CONNECTIONS={}

# STREAMLIT CACHING 
@st.cache_resource
def load_backend_components(model_choice, graph_file_path):
    """Loads the ML classifier and the Graph structure."""
    global PARSED_NODE_COORDS,PARSED_GRAPH_CONNECTIONS
    with st.spinner(f"Loading OSM data and initializing backend:"):
        _, _, node_id_map,metadata = read_traffic_data(graph_file_path)
        osm_content=read_osm_file()
        if osm_content is None: 
            PARSED_NODE_COORDS={}
            PARSED_GRAPH_CONNECTIONS={}
        else:
            PARSED_NODE_COORDS, PARSED_GRAPH_CONNECTIONS=parse_osm_data(osm_content)
            
        node_list=list(PARSED_NODE_COORDS.keys())
        ml_classifier = IncidentClassifier(model_choice=model_choice)
        traffic_graph = Graph(PARSED_NODE_COORDS, PARSED_GRAPH_CONNECTIONS)
        
        if not node_list:
            st.warning("No named nodes were successfully loaded from OSM file. Check file integrity.")
        
    return ml_classifier, traffic_graph, node_list,node_id_map,metadata

# ----------------------------------------------------
# PAGE SETUP
# ----------------------------------------------------

st.set_page_config(
    page_title="Traffic ICS Route Finder",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Traffic Incident Classification System (ICS)")
st.caption("COS30019 Assignment 2B: ML Integration with Pathfinding")

# ----------------------------------------------------
# SIDEBAR - PARAMETER SETTINGS (Input & Control)
# ----------------------------------------------------
st.sidebar.header("Data & Pathfinding Parameters")

# Get list of possible files 
all_files = Graph.get_all_possible_files()

# 1. File (Case) Selection
file_choice = st.sidebar.selectbox(
    "1. Select Traffic Data Case:", 
    options=all_files,
    index=0 
)

# ML Model Selection (For Report Testing/Comparison)
model_options = ['Custom CNN', 'VGG16', 'EfficientNet']
model_choice = st.sidebar.selectbox(
    "2. ML Model Configuration:", 
    options=model_options,
    index=0 
)

# Load the backend components once (cached). This returns the node list based on the file choice.
ML_CLASSIFIER, TRAFFIC_GRAPH, NODE_LIST, DEFINITIVE_NODES_FROM_FILE,METADATA = load_backend_components(model_choice, file_choice)


# Default indices for the selectbox to ensure they are valid
if NODE_LIST:
    file_start_node=METADATA.get('START')
    default_start_node=file_start_node if file_start_node in NODE_LIST else NODE_LIST[0]
    file_end_node=METADATA.get('GOAL')
    default_end_node=file_end_node if file_end_node in NODE_LIST else NODE_LIST[1]
    default_start_index=NODE_LIST.index(default_start_node) 
    default_end_index = NODE_LIST.index(default_end_node)

if not NODE_LIST:
    st.sidebar.error("No nodes loaded. Check 'map.osm' file in the root directory.")
    start_node = ""
    end_node = ""
else:
    start_node = st.sidebar.selectbox(
        "3. Origin Node:", 
        options=NODE_LIST,
        index=default_start_index if default_start_index is not None else 0
    )
    end_node = st.sidebar.selectbox(
        "4. Destination Node:", 
        options=NODE_LIST,
        index=default_end_index if default_end_index is not None else 0
    )
    
# Pathfinding Algorithm Selection
algorithms = ['BFS','DFS','BDS','GBFS','A*','IDA*']
algorithm_choice = st.sidebar.selectbox(
    "5. Select Algorithm:", 
    options=algorithms,
    index=0 
)


# Find Route Button
if st.sidebar.button("Find Optimized Route", type="primary"):
    st.session_state['run_search'] = True
else:
    # Initialize run_search state
    if 'run_search' not in st.session_state:
         st.session_state['run_search'] = False

# ----------------------------------------------------
# MAIN AREA - ROUTE SEARCH EXECUTION
# ----------------------------------------------------

def draw_map_visualization(path: List[str], graph: Graph, start_node: str, end_node: str):
    """
    Renders the map visualization using PyDeck, plotting the route and key nodes.
    """
    st.subheader("Visual Path Recommendation")
    
    if not graph.coords:
        st.warning("Cannot draw map: No geographical coordinates loaded from the data.")
        return

    # 1. Prepare data for PyDeck
    route_data = []
    points_data = []
    
    # Process path into line segments and node points
    path_set = set(path)
    
    # Add all named nodes for context, marking path nodes specifically
    for node_name, coords in graph.coords.items():
        lat, lon = coords
        
        is_on_path = node_name in path_set
        is_start_end = node_name == start_node or node_name == end_node

        if is_on_path:
             # Add to points data for markers
            points_data.append({
                'name': node_name,
                'lat': lat,
                'lon': lon,
                # Color code: Red for Start/End, Blue for other Path Nodes, Grey for general nodes (if plotting all)
                'color': [255, 0, 0, 255] if is_start_end else [0, 150, 200, 255]
            })

    # Prepare line segments based on the calculated path
    for i in range(len(path) - 1):
        node_u = path[i]
        node_v = path[i+1]
        
        coords_u = graph.get_node_coords(node_u)
        coords_v = graph.get_node_coords(node_v)
        
        if coords_u and coords_v:
            # LineLayer needs [longitude, latitude] format
            route_data.append({
                'source': [coords_u[1], coords_u[0]], # [lon, lat]
                'target': [coords_v[1], coords_v[0]], # [lon, lat]
                'color_r': 255, # Mock: use red for all route segments
                'color_g': 0, 
                'color_b': 0, 
                'stroke_width': 10,
                'segment': f"{node_u} -> {node_v}"
            })

    # 2. Define center for the map
    df_points = pd.DataFrame(points_data)
    
    # Center the map on the calculated path
    if not df_points.empty:
        center_lat = df_points['lat'].mean()
        center_lon = df_points['lon'].mean()
    else:
        # Fallback center coordinates (Kuching)
        center_lat = 1.558
        center_lon = 110.344

    # --- Define Layers ---
    
    # 1. Route Line Layer
    line_layer = pdk.Layer(
        'LineLayer',
        data=route_data,
        get_source_position='source',
        get_target_position='target',
        get_color='[color_r, color_g, color_b, 200]', # Use defined color for segment severity
        get_width=5,
        pickable=True
    )
    
    # 2. Node Point Layer (Markers)
    point_layer = pdk.Layer(
        'ScatterplotLayer',
        data=df_points,
        get_position='[lon, lat]', # PyDeck requires [lon, lat] for position
        get_fill_color='color',
        get_radius=10,
        pickable=True,
        auto_highlight=True
    )

    # --- Define View State and Deck ---
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=14,
        pitch=45,
    )

    r = pdk.Deck(
        layers=[line_layer, point_layer], 
        initial_view_state=view_state,
        tooltip={"text": "{name}\n{segment}"}
    )
    
    st.pydeck_chart(r)
    st.caption(f"Map centered at ({center_lat:.4f}, {center_lon:.4f}).")
    st.info("NOTE: Line segments are currently mock-colored (Red). In your final submission, this color must reflect the ML-predicted severity/cost of the road segment.")


def find_route():
    """Executes the pathfinding algorithm and displays results."""
    
    # Input validation moved here just before execution
    if not start_node or not end_node or start_node == end_node:
        st.warning("Please select valid and different Origin and Destination nodes.")
        return
    
    st.subheader(f"Route Search: {algorithm_choice} (Data: {file_choice}, Model: {model_choice})")

    # Console area for logging ML predictions per edge
    with st.expander("Show Detailed Prediction Log", expanded=False):
        log_container = st.container()

    # --- 1. Dynamic Cost Function Definition ---
    def dynamic_cost_function(node_u, node_v):
        """Retrieves base time and image path from graph, then uses ML to calculate cost."""
        try:
            # 1. Get base data from the graph
            base_time = TRAFFIC_GRAPH.get_base_time(node_u, node_v)
            if base_time == float('inf'):
                # Handle unconnected nodes during pathfinding attempts
                log_container.code(f"Edge {node_u}->{node_v}: No direct connection found in mock data.")
                return float('inf')
                
            image_path = TRAFFIC_GRAPH.get_image_path(node_u, node_v)
            
            # 2. The key ML integration call (using the mock function)
            predicted_cost = calculate_edge_cost(base_time, image_path, ML_CLASSIFIER)
            
            # For this simple frontend, we'll log the base data access:
            log_container.code(f"Edge {node_u}->{node_v}: Base Time={base_time:.2f}, Predicted Cost={predicted_cost:.2f}, Image={image_path}")

            return predicted_cost
        
        except Exception as e:
            st.error(f"Error calculating cost for edge {node_u}->{node_v}. Error: {e}")
            return float('inf') 

    # Call Pathfinding Algorithm
    start_time = time.time()
    try:
        path, predicted_time, total_nodes_explored = find_path_algorithm(
            TRAFFIC_GRAPH, 
            start_node, 
            end_node, 
            algorithm_choice, 
            cost_func=dynamic_cost_function
        )
    except Exception as e:
        st.error(f"Pathfinding Algorithm Execution Error: An unexpected error occurred: {e}")
        return

    end_time = time.time()
    search_duration = end_time - start_time

    #  Display Results
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Predicted Travel Time", f"{predicted_time:.2f} units")
    col2.metric("Algorithm Used", algorithm_choice)
    col3.metric("Search Duration", f"{search_duration:.4f} seconds")
    col4.metric("Nodes Explored", total_nodes_explored)

    if path and predicted_time < float('inf'):
        st.success(f"✅ Route Found: {start_node} to {end_node}")
        st.code(f"{' -> '.join(path)}", language="text")

        # ------------------ 4. Map Visualization ------------------
        draw_map_visualization(path, TRAFFIC_GRAPH, start_node, end_node)

    else:
        st.error(f"❌ No path found from {start_node} to {end_node}. Check connectivity or start/end nodes.")


if st.session_state.get('run_search', False):
    find_route()
    st.session_state['run_search'] = False 