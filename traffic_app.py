import streamlit as st
import os
import folium
from streamlit_folium import st_folium
from classifier import IncidentClassifier
from constant import SEVERITY_DISPLAY_ORDER, ACCIDENT_TYPE, SEVERITY_CLASSES
from typing import Optional, Tuple, Dict, List
import xml.etree.ElementTree as ET
import base64
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString
from graph import Graph,find_path_algorithm

st.set_page_config(
    page_title="Traffic App", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

@st.cache_data(show_spinner="Loading and processing OSM graph data (via OSMnx)...")
def load_osm_graph(filepath: str) -> Optional[nx.MultiDiGraph]:
    try:
        # 1. Load the graph from the local XML file
        G = ox.graph_from_xml(filepath, simplify=False)
        
        # 2. Define the highway tags we want to KEEP in the network
        allowed_highways = {
            "motorway", "trunk", "primary", "secondary", "tertiary","unclassified", "residential", "service"
        }
        
        edges_to_remove = []
        
        # 3. Iterate over all edges and identify non-road features for removal
        for u, v, k, data in G.edges(keys=True, data=True):
            highway_tag = data.get('highway')
            
            # Normalize the highway tag (it can be a string or a list of strings)
            if isinstance(highway_tag, (list, tuple)):
                current_highways = set(highway_tag)
            elif isinstance(highway_tag, str):
                current_highways = {highway_tag}
            else:
                # If no highway tag is present, it is marked as non-road
                current_highways = set() 
                
            # Check if there is any intersection between the edge's tags and the allowed tags
            if not current_highways.intersection(allowed_highways):
                edges_to_remove.append((u, v, k))

        # 4. Remove the non-road edges
        G.remove_edges_from(edges_to_remove)
        
        # 5. Clean up isolated nodes (by keeping only the largest connected component)
        if len(G.nodes) > 0:
            G_components = list(nx.weakly_connected_components(G))
            
            # Identify the largest component by node count
            largest_component_nodes = max(G_components, key=len) 
            G = G.subgraph(largest_component_nodes).copy()
        
        return G
        
    except Exception as e:
        st.error(f"OSMnx Error loading graph from '{filepath}': {e}")
        st.info("Please ensure your 'map.osm' file is a valid OSM XML file.")
        return None

def get_way_data_from_graph(G: nx.MultiDiGraph, from_lat_lon: Tuple[float, float], to_lat_lon: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], str]:
    coords = []
    road_name = "Unnamed Road"
    
    try:
        u_osm_node = ox.nearest_nodes(G, from_lat_lon[1], from_lat_lon[0])
        v_osm_node = ox.nearest_nodes(G, to_lat_lon[1], to_lat_lon[0])
        
        route = ox.shortest_path(G, u_osm_node, v_osm_node, weight='length')
        
        if route is None:
            # If no route is found, just connect the custom start/end nodes
            return [(from_lat_lon[0], from_lat_lon[1]), (to_lat_lon[0], to_lat_lon[1])], "Route Not Found"

        # 3. Extract detailed geometry and name from the route's edges
        all_coords = []
        name_list = set()
        
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i+1]
            
            edge_data = G.get_edge_data(u, v)
            
            if edge_data:
                data = edge_data[list(edge_data.keys())[0]] 
                
                if 'geometry' in data and isinstance(data['geometry'], LineString):
                    geom_coords = [(y, x) for x, y in data['geometry'].coords]
                    all_coords.extend(geom_coords)
                else:
                    # Fallback: use node coordinates
                    all_coords.append((G.nodes[u]['y'], G.nodes[u]['x']))
                
                # Extract road name
                if 'name' in data:
                    name = data['name']

                    if isinstance(name, list):
                        name_list.update(name)
                    else:
                        name_list.add(name)

        if all_coords:
            final_coords = [(from_lat_lon[0], from_lat_lon[1])]
            final_coords.extend(all_coords) 
            final_coords.append((to_lat_lon[0], to_lat_lon[1]))
            coords = list(dict.fromkeys(final_coords))
            
        else:
             # Fallback: connect the two nearest OSM nodes
             coords = [(G.nodes[u_osm_node]['y'], G.nodes[u_osm_node]['x']), 
                       (G.nodes[v_osm_node]['y'], G.nodes[v_osm_node]['x'])]
        
        # Format road name
        if name_list:
            road_name = ", ".join(sorted(list(name_list)))
        elif len(coords) > 2:
            road_name = "Segmented Road"
        
    except Exception as e:
        coords = [(from_lat_lon[0], from_lat_lon[1]), (to_lat_lon[0], to_lat_lon[1])]
        road_name = "Fallback Line"
        
    return coords, road_name

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
# Read OSM file from local disk
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

# Clear accident severity
def clear_accident(way_id, graph):
    st.session_state.manual_cleared.add(way_id)
    if way_id in st.session_state.accident_severity:
        del st.session_state.accident_severity[way_id]
    st.session_state.road_choice=None

    if st.session_state.get("algorithm_choice", "-- Select an algorithm --") != "-- Select an algorithm --":
        compute_path(graph)

# Clear algorithm selection
def clear_algorithm():
    st.session_state.algorithm_choice="-- Select an algorithm --"
    st.session_state.all_paths_to_draw = {}

# Retrieve test case data
def parse_map_data(text):
    nodes = {}
    ways = []
    cameras = {}
    meta = {}

    section = None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Detect section headers
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue

        # Parse lines inside each section
        if section == "NODES":
            # format: id,lat,lon,name
            parts = line.split(",", 3)
            node_id = int(parts[0])
            lat = float(parts[1])
            lon = float(parts[2])
            name = parts[3]
            nodes[node_id] = {"lat": lat, "lon": lon, "name": name}

        elif section == "WAYS":
            # We need to split by commas but ignore commas inside brackets []
            parts = line.split(",", 5)

            way_id = int(parts[0])
            from_node = int(parts[1])
            to_node = int(parts[2])
            road = parts[3].strip()
            highway_type = parts[4]
            travel_time = int(parts[5])

            ways.append({
                "way_id": way_id,
                "from": from_node,
                "to": to_node,
                "road": road,
                "type": highway_type,
                "time": travel_time
            })

        elif section == "CAMERAS":
            # format: way_id,image_path
            way_id, img = line.split(",", 1)
            cameras[int(way_id)] = img

        elif section == "META":
            # format: KEY,VALUE
            key, value = line.split(",", 1)
            # convert numeric entries
            if value.isdigit():
                value = int(value)
            meta[key] = value

    return nodes, ways, cameras, meta

# Logic to execute the algorithm        
def compute_path(graph:Graph):
    if "algorithm_choice" in st.session_state and st.session_state.algorithm_choice != "-- Select an algorithm --":
        all_paths = find_path_algorithm(
            graph=graph,
            start=str(st.session_state.start_node),
            end=str(st.session_state.end_node), 
            algorithm_choice=st.session_state.algorithm_choice,  
            accident_segments=st.session_state.accident_severity,
            accident_multiplier=float(meta['ACCIDENT_MULTIPLIER'])
        )

    st.session_state.selected_path_index = 0
    st.session_state.all_paths_to_draw = all_paths 
    st.session_state.all_path_sequences = [p["path"] for p in all_paths]

    st.session_state.all_path_costs = [p["total_cost"] for p in all_paths]
    st.session_state.all_path_steps = [p["steps_count"] for p in all_paths]

def draw_map(nodes, ways, accident_severity):
    # Create map
    m = folium.Map(location=[1.559, 110.345], zoom_start=16, tiles="OpenStreetMap")

    # Nodes (start, end, normal)
    start_node = st.session_state.get('start_node', -1)
    end_node = st.session_state.get('end_node', -1)

    for nid, info in nodes.items():
        if nid == start_node:
            color = "green"
        elif nid == end_node:
            color = "red"
        else:
            color = "blue"

        folium.Marker(
            [info["lat"], info["lon"]],
            tooltip=f"{nid}: {info['name']}",
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)

    # Recommended paths to draw
    calculated_paths = st.session_state.get('all_paths_to_draw', [])
    selected_index = st.session_state.get('selected_path_index', 0)

    if calculated_paths:
        best_path_info = calculated_paths[selected_index]
        best_path_nodes = [int(n) for n in best_path_info['path']] 
        best_path_cost = best_path_info['total_cost']
        other_paths_info = [
            p for i, p in enumerate(calculated_paths) if i != selected_index
        ]
    else:
        best_path_nodes = []
        other_paths_info = []

    # Severity colours
    severity_colors = {
        "Major": "red",
        "Intermediate": "orange",
        "Minor": "yellow",
        "no_accident": "green"
    }

    # Read toggle mode from Streamlit
    mode = st.query_params.get("mode", "real")
    accident_ways = []
    normal_ways = []

    # Draw ways (two modes)
    # Custom icon for accident marker
    icon_image = "traffic-icon.png"

    icon = folium.CustomIcon(
        icon_image,
        icon_size=(40, 40),
        icon_anchor=(20, 42),
        popup_anchor=(-3, -40),
    )
    
    for w in ways:
        start = nodes[w["from"]]
        end = nodes[w["to"]]

        info = accident_severity.get(w["way_id"], {"severity": "no_accident", "image": None})
        w["severity"] = info["severity"]
        w["image_url"] = info["image"]

        start = nodes[w["from"]]
        end = nodes[w["to"]]

        # SIMPLIFIED MODE
        if mode == "simplified":
            w["coords"] = [
                (start["lat"], start["lon"]),
                (end["lat"], end["lon"])
            ]
        else:
            w["coords"] = w["geometry"]

        if w["severity"] != "no_accident":
            accident_ways.append(w)
        else:
            normal_ways.append(w)

    # Draw Normal Roads (Bottom Layer)
    for w in normal_ways:
        road_color = severity_colors[w["severity"]]
        road_name = w['road']

        folium.PolyLine(
            locations=w["coords"],
            tooltip=f"{w['way_id']}: {road_name}",
            color=road_color,
            weight=4,
            opacity=0.8
        ).add_to(m)
    
    # Draw Alternative Paths (if any)
    for path_info in other_paths_info:
        path_nodes = [int(n) for n in path_info['path']]
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i+1]
            
            segment = next((w for w in ways if w["from"] == u and w["to"] == v), None)

            if segment:
                road_color = '#ADD8E6' # Light Blue
                folium.PolyLine(
                    locations=segment["coords"],
                    color=road_color,
                    weight=8, 
                    opacity=0.5
                ).add_to(m)

    # Draw Best Path
    if best_path_nodes:
        total_time_str = f"Time: {best_path_cost:.0f} min"

        all_best_path_coords = []

        for i in range(len(best_path_nodes) - 1):
            u = best_path_nodes[i]
            v = best_path_nodes[i+1]
            
            # Find the corresponding way segment from your 'ways' list
            segment = next((w for w in ways if w["from"] == u and w["to"] == v), None)
            
            if segment:
                # Store name for tooltip/popup
                all_best_path_coords.extend(segment["coords"])
                
                road_color = '#00008B' # Dark Blue
                folium.PolyLine(
                    locations=segment["coords"],
                    color=road_color,
                    weight=7, 
                    opacity=0.8 
                ).add_to(m)

        if all_best_path_coords:
            folium.PolyLine(
                locations=list(dict.fromkeys(all_best_path_coords)), 
                tooltip=folium.Tooltip(total_time_str, permanent=True), 
                color='rgba(0,0,0,0)', 
                weight=10, 
                opacity=0.0
            ).add_to(m)
    
    # Draw Accident Roads and Markers (Top Layer) 
    for w in accident_ways:
        start = nodes[w["from"]]
        end = nodes[w["to"]]
        
        road_color = severity_colors[w["severity"]]
        road_name = w['road']
        
        # 1. Draw the Road Segment (on top)
        folium.PolyLine(
            locations=w["coords"],
            tooltip=f"{w['way_id']}: {road_name}",
            color=road_color,
            weight=6, 
            opacity=0.6
        ).add_to(m)

        # 2. Add the Accident Marker
        # Midpoint calculation 
        coords = w["coords"]
        if len(coords) > 2:
            mid_index = len(coords) // 2
            mid_lat, mid_lon = coords[mid_index]
        else:
            mid_lat = (start["lat"] + end["lat"]) / 2
            mid_lon = (start["lon"] + end["lon"]) / 2

        base_time = w["time"] 
        adjusted_time = base_time * ACCIDENT_TYPE.get(w["severity"].lower(), 1.0) * float(meta['ACCIDENT_MULTIPLIER'])

        # Build popup HTML
        image_url = w["image_url"]
        if image_url:
            b64 = image_to_base64(image_url)
            img_html = f"<img src='data:image/jpeg;base64,{b64}' style='width:100%; border-radius:5px;'>"
        else:
            img_html = "<i>No image available</i>"

        popup_html = f"""
        <div style="width:220px">
            <h5 style="margin:0; font-size:18px;">{road_name}</h5>
            {img_html}
            <p style="margin:0; padding-top:10px;">
                Severity: {w['severity']}<br>
                Base Time: {base_time} min<br>
                Adj. Time: <span style='color:red;'>{adjusted_time:.0f} min</span>
            </p>
        </div>
        """

        # Add marker (on top of the PolyLine)
        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=icon,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)

    st.markdown("""
        <style>
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            margin-top: 40px !important;
        }
        .css-1d391kg {padding:0 !important; margin:0 !important;}
        </style>
    """, unsafe_allow_html=True)

    # Legend
    legend_html = """
    <div style="
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border-radius: 8px;
        box-shadow: 0 0 6px rgba(0,0,0,0.3);
        font-size: 14px;
        color: black;
    ">
        <b>Legend</b><br>
        <span style="color:green;">●</span> Start Node<br>
        <span style="color:red;">●</span> End Node<br>
        <span style="color:blue;">●</span> Normal Node<br>
        <div style="display:flex; align-items:center; gap:5px;"> 
            <span style="display:inline-block; width:25px; height:4px; background-color:red;"></span> Major Accident Road<br> 
        </div> 
        <div style="display:flex; align-items:center; gap:5px;"> 
            <span style="display:inline-block; width:25px; height:4px; background-color:orange;"></span> Intermediate Accident Road<br> 
        </div> 
        <div style="display:flex; align-items:center; gap:5px;"> 
            <span style="display:inline-block; width:25px; height:4px; background-color:yellow;"></span> Minor Accident Road<br> 
        </div>
        <div style="display:flex; align-items:center; gap:5px;"> 
            <span style="display:inline-block; width:25px; height:4px; background-color:#00008B;"></span> Selected Path<br> 
        </div>
        <div style="display:flex; align-items:center; gap:5px;"> 
            <span style="display:inline-block; width:25px; height:4px; background-color:#ADD8E6;"></span> Alternative Path<br> 
        </div>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))

    # 7. Render map
    st_folium(m, width="100%", height=560, key="traffic_map_display")

    return m

def set_selected_path_index(index):
    st.session_state.selected_path_index = index

# PAGE SETUP
SIDEBAR_WIDTH = "500px"

st.markdown(
    f"""
    <style>
        [data-testid="stSidebar"] {{
            width: {SIDEBAR_WIDTH} !important;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            width: {SIDEBAR_WIDTH} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# SIDEBAR 
st.sidebar.header("🚦Traffic ICS Route Finder")

# Sessions
if "accident_severity" not in st.session_state:
    st.session_state.accident_severity = {}
if "algorithm_choice" not in st.session_state:
    st.session_state.algorithm_choice="-- Select a algorithm --"
if "all_paths_to_draw" not in st.session_state:
    st.session_state.all_paths_to_draw = {}
if 'prev_file_choice' not in st.session_state:
    st.session_state.prev_file_choice = None
if "selected_path_index" not in st.session_state:
    st.session_state.selected_path_index = 0
if "start_node" not in st.session_state:
    st.session_state.start_node = None
if "end_node" not in st.session_state:
    st.session_state.end_node = None
if "manual_cleared" not in st.session_state:
    st.session_state.manual_cleared = set()
if "road_choice" not in st.session_state:
    st.session_state.road_choice = None

# Test Case Selection
all_files = [f for f in os.listdir("test_case") if f.endswith(".txt")] # Get all .txt files inside the folder
with st.sidebar.expander("1. Traffic Case Selection", expanded=True):
    file_choice=st.session_state.file_choice = st.selectbox(
        "Select Traffic Data Case:", 
        options=all_files,
        index=all_files.index(st.session_state.prev_file_choice) if st.session_state.prev_file_choice in all_files else 0
    )

if st.session_state.file_choice != st.session_state.prev_file_choice:
    # Reset the algorithm choice
    st.session_state.algorithm_choice = "-- Select an algorithm --"
    # Update prev_file_choice
    st.session_state.prev_file_choice = st.session_state.file_choice
    st.session_state.all_paths_to_draw = []
    st.session_state.accident_severity = {}
    st.session_state.road_choice = None

    if 'origin_select' in st.session_state:
        del st.session_state.origin_select
    if 'goal_select' in st.session_state:
        del st.session_state.goal_select

# 2. Machine Learning Model Configuration
model_options = ['Custom CNN', 'VGG16', 'EfficientNet']
with st.sidebar.expander("2. Machine Learning Model Configuration", expanded=True):
    # Accident Model Selection
    accident_model_choice = st.selectbox(
        "Accident Classifier Model (Accident/No Accident):", 
        options=model_options,
        key='accident_model_choice'
    )

    # Severity Model Selection
    severity_model_choice = st.selectbox(
        "Severity Classifier Model:", 
        options=model_options,
        key='severity_model_choice'
    )

osm_content = read_osm_file()

G = None

if osm_content:
    G = load_osm_graph('map.osm')
else:
    osm_name_map = {}

if file_choice:
    file_path = os.path.join("test_case", file_choice)

    with open(file_path, "r") as f:
        text = f.read()

    # Parse map data
    nodes, ways, cameras, meta = parse_map_data(text)
    if st.session_state.file_choice != st.session_state.prev_file_choice:
        nodes, ways, cameras, meta = parse_map_data(text)
    graph_connections = {}
    edge_info = {}
    for way in ways:
        u = str(way["from"])
        v = str(way["to"])
        graph_connections.setdefault(u, {})[v] = {"base_time": way["time"], "way_id": way["way_id"]}
        edge_info.setdefault(u, {})[v] = {"base_time": way["time"], "way_id": way["way_id"]}

    # Coordinates
    coords = {str(nid): (data["lat"], data["lon"]) for nid, data in nodes.items()}
    graph = Graph(node_coords=coords, graph_connections=graph_connections, edge_info=edge_info)
    if G is not None:
        for w in ways:
            u_id = w["from"]
            v_id = w["to"]

            start_coords = (nodes[u_id]["lat"], nodes[u_id]["lon"])
            end_coords = (nodes[v_id]["lat"], nodes[v_id]["lon"])
           
            w["geometry"], w['real_road_name'] = get_way_data_from_graph(
                G, start_coords, end_coords
            )

    # Rebuild classifier when model choices change
    selected_models = (accident_model_choice, severity_model_choice)

    if ("classifier" not in st.session_state or
        "model_choices" not in st.session_state or
        st.session_state.model_choices != selected_models):

        st.session_state.classifier = IncidentClassifier(
            accident_model_choice,
            severity_model_choice
        )

        st.session_state.model_choices = selected_models

    classifier = st.session_state.classifier

    # ------------------------------------------
    # CAMERA SEVERITY PREDICTION (AUTO)
    # ------------------------------------------

    # initialise classifier once
    if "classifier" not in st.session_state:
        st.session_state.classifier = IncidentClassifier(accident_model_choice, severity_model_choice)

    classifier = st.session_state.classifier

    # process every camera image
    for way_id, img_path in cameras.items():
        if way_id not in st.session_state.manual_cleared:
            severity, _ = classifier.predict_severity(os.path.join("test_case", img_path))

            st.session_state.accident_severity[way_id] = {
                "severity": severity,
                "image": os.path.join("test_case", img_path)
            }

# 3. Location Selection
with st.sidebar.expander("3. Location Selection", expanded=True):
    # Defaults from META
    default_start = meta.get("START")
    default_end   = meta.get("GOAL")

    # Build mapping "label → id"
    node_choices = {f"{nid} – {info['name']}": nid for nid, info in nodes.items()}

    # ---- Start Node Dropdown ----
    start_label = st.selectbox(
        "Origin Node:",
        options=list(node_choices.keys()),
        index=list(node_choices.values()).index(default_start)
              if default_start in node_choices.values() else 0,
        key='origin_select',
        on_change=clear_algorithm
    )
    start_node = node_choices[start_label]
    st.session_state['start_node'] = start_node 

    # ---- End Node Dropdown ----
    end_choices = {label: nid for label, nid in node_choices.items() if nid != start_node}

    end_label = st.selectbox(
        "Goal Node:",
        options=list(end_choices.keys()),
        index=list(end_choices.values()).index(default_end)
              if default_end in end_choices.values() else 0,
        key='goal_select',
        on_change=clear_algorithm
    )
    end_node = end_choices[end_label]
    st.session_state['end_node'] = end_node

# 4. Pathfinding Algorithm Selection
with st.sidebar.expander("4. Pathfinding Algorithm Selection", expanded=True):
    algorithm_options = ["-- Select an algorithm --"] + ['BFS','DFS','BDS','GBFS','A*','IDA*']
    algorithm_choice = st.selectbox(
        "Select Pathfinding Algorithm:", 
        options=algorithm_options,
        key='algorithm_choice',
        on_change=lambda: compute_path(graph)
    )

    # Check if we have paths stored
    all_paths = st.session_state.get('all_paths_to_draw', {})
    if all_paths and st.session_state.algorithm_choice != "-- Select an algorithm --":
        path_data = []
        radio_options = {}

        for i, path_info in enumerate(all_paths[:5]): # Limit to first 5 paths
            path_nodes = path_info["path"]
            path_names=[nodes[int(n)]["name"] for n in path_nodes]
            total_cost = path_info["total_cost"]
            steps = path_info["steps_count"]

            display_label = f"Route {i+1} (Time: {total_cost:.0f} min)"
            radio_options[display_label] = i 

            path_data.append({
                "Route": f"Route {i+1}",
                "Time (min)": f"{total_cost:.0f}",
                "Steps": steps,
                "Node Sequence": ' → '.join(path_names)
            })

        # Default index should be the currently selected path
        current_index = st.session_state.get('selected_path_index', 0)
        
        # Ensure index is valid for the options list
        if current_index >= len(list(radio_options.values())):
            current_index = 0
            st.session_state.selected_path_index = 0 # Reset to 0 if out of range

        # 1. Use st.radio for the selection
        selected_label = st.radio(
            "**Select Route to Highlight:**",
            options=list(radio_options.keys()),
            index=current_index,
            key='route_selection_radio',
            on_change=lambda: set_selected_path_index(radio_options[st.session_state.route_selection_radio])
        )

        active_index = radio_options.get(selected_label, 0)
        
        # 3. Display details for ONLY the currently active route (no tabs needed)
        if 0 <= active_index < len(path_data):
            details = path_data[active_index]

            st.subheader(f"{details['Route']} Details")
            
            st.markdown(f"**Total Cost (Time):** {details['Time (min)']} min")
            st.markdown(f"**Steps Count:** {details['Steps']}")
            st.markdown(f"**Full Path:**")
            st.markdown(f"{details['Node Sequence']}")
        else:
            st.info("Select a route above to see details.")

# 5. Accident Upload 
with st.sidebar.expander("5. Upload Accident Image", expanded=True):
    display_options_map = {}
    
    for w in ways:
        way_id = w['way_id']
        from_node_id = w['from']
        to_node_id = w['to']
        
        # Look up the names in the nodes dictionary
        start_name = nodes.get(from_node_id, {}).get('name', f"Node {from_node_id}")
        end_name = nodes.get(to_node_id, {}).get('name', f"Node {to_node_id}")
        display_string = f"{way_id}: {start_name} -> {end_name}"
        display_options_map[display_string] = str(way_id)

    way_display_options = list(display_options_map.keys())

    # Create the selectbox with the descriptive labels
    road_choice = st.selectbox(
        "Select Road for Accident Location:", 
        options=way_display_options,
        key='road_choice'
    )
    
    # Check if a choice was made before proceeding
    if road_choice is not None:
        selected_way_id_str = display_options_map.get(road_choice)
        selected_way_id = int(selected_way_id_str) if selected_way_id_str else None

        uploaded_image = st.file_uploader(
            "Upload an image of the accident location:",
            type=["png", "jpg", "jpeg"],
            key='uploaded_file'
        )

        if uploaded_image is not None and selected_way_id is not None:
            # DISPLAY UPLOADED IMAGE
            st.image(uploaded_image, caption=f"Uploaded Image for Way: {road_choice}") 

            IMAGES_DIR = "uploads"
            os.makedirs(IMAGES_DIR, exist_ok=True)

            # Use the main way_id for file naming to prevent conflicts
            file_name = f"accident_{selected_way_id}.jpg"
            save_path = os.path.join(IMAGES_DIR, file_name)

            with open(save_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            try:
                severity_class, severity_probs = classifier.predict_severity(save_path)

                # Store in session state
                st.session_state.accident_severity[selected_way_id] = {
                    "severity": severity_class,
                    "image": save_path
                }
                if st.session_state.algorithm_choice != "-- Select an algorithm --":
                    compute_path(graph)
                if severity_class == "no_accident":
                    st.info("No accident detected in this image.")

                    for key, val in severity_probs.items():
                        st.progress(int(val * 100), text=f"{key}: {val * 100:.2f}%")
                else:
                    st.warning(f"Severity Detected for Way {selected_way_id}: {severity_class}")

                    for key in SEVERITY_DISPLAY_ORDER:
                        if key in severity_probs:
                            val = severity_probs[key]
                            st.progress(int(val * 100), text=f"{key}: {val * 100:.2f}%")

            except Exception as e:
                st.error(f"Error during accident detection: {e}")

    if st.session_state.get("accident_severity"):
        # Show the accident for user
        way_id_to_nodes = {
            w["way_id"]: (w["from"], w["to"])
            for w in ways
        }

        st.subheader("Current Detected Accident Roads")
        
        for way_id, info in st.session_state.accident_severity.items():
            severity = info["severity"]
            image_path = info["image"]

            if severity != "no_accident":
                # Convert to readable node names
                from_node, to_node = way_id_to_nodes.get(way_id, (None, None))
                from_name = nodes.get(from_node, {}).get("name", f"Node {from_node}")
                to_name   = nodes.get(to_node,   {}).get("name", f"Node {to_node}")
                
                # Title
                st.markdown("---")
                st.markdown(f"### {way_id}: {from_name} → {to_name}")

                target_class_index = SEVERITY_CLASSES.index(severity)
                gradcam_image = classifier.generate_grad_cam(image_path=image_path, target_class=target_class_index)

                # Thumbnail
                st.image(gradcam_image, width=300)

                st.write(f"Severity: {severity}")
                # Add the Clear Button
                st.button(
                    label=f"Clear Accident",
                    key=f"clear_btn_{way_id}",
                    on_click=clear_accident, 
                    args=(way_id,graph,),
                    type="primary"
                )

# Toggle Switch for Map Mode
col1, col2 = st.columns([0.8, 0.2])

with col2:
    show_simplified = st.toggle(
        "Simplified Map Mode", 
        value=st.query_params.get("mode") == "simplified",
        key='map_mode_toggle'
    )
    
    if show_simplified:
        st.query_params['mode'] = 'simplified'
    else:
        # Deleting the param also causes a rerun
        if 'mode' in st.query_params:
            del st.query_params['mode']
        
draw_map(nodes, ways, st.session_state.accident_severity)