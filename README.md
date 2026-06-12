# Traffic Incident Classification System

A Streamlit app that combines image-based incident classification with graph search to recommend a route through a road network.

## What It Does

- Loads a road network from `map.osm`
- Loads traffic test cases from `test_case/*.txt`
- Uses a two-stage PyTorch classifier to estimate whether an incident occurred and, if so, its severity
- Converts severity into a travel-time penalty
- Runs a pathfinding algorithm to choose the best route
- Displays the result in Streamlit with a PyDeck map visualization

## Project Structure

- `traffic_app.py` - Streamlit UI and end-to-end routing workflow
- `classifier.py` - Wrapper that loads the trained accident and severity models
- `model_definitions.py` - CNN, VGG16, and EfficientNet model definitions
- `graph.py` - Graph container plus BFS, DFS, bidirectional search, greedy best-first search, A*, and IDA*
- `edgecost.py` - Converts predicted severity into an edge cost
- `constant.py` - Shared constants, severity labels, and penalty weights
- `heuristic.py` - Heuristic function used by informed search methods
- `map.osm` - OpenStreetMap data used to place named nodes on the map
- `accident_*.ipynb` / `severity_*.ipynb` - Training and experiment notebooks for the classifiers
- `tool.yml` - Conda environment specification

## Requirements

This project targets Python 3.11 and uses:

- `streamlit`
- `torch`, `torchvision`, `torchaudio`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pydeck`
- `opencv-python-headless`
- `joblib`

## Setup

1. Create the environment:

```bash
conda env create -f tool.yml
conda activate cos30019_pytorch
```

2. Make sure these local assets exist:

- `map.osm`
- `test_case/*.txt`
- `models/accident_custom_final_cnn_pytorch.pth`
- `models/severity_custom_final_cnn_pytorch.pth`
- `models/accident_transfervgg16_cnn_pytorch.pth`
- `models/severity_transfervgg16_cnn_pytorch.pth`
- `models/accident_transferefficient_unfreeze_cnn_pytorch.pth`
- `models/severity_transferefficient_unfreeze_cnn_pytorch.pth`

3. Run the app:

```bash
streamlit run traffic_app.py
```

## How It Works

1. The sidebar lets you choose a traffic case, model family, origin, destination, and routing algorithm.
2. The app reads the selected traffic file and matches its node names to locations in `map.osm`.
3. For each edge, the classifier predicts incident severity from the associated image.
4. Severity is translated into a multiplier that increases the base travel time.
5. The selected search algorithm finds a route using those dynamic edge costs.
6. The final path and node markers are shown on the map.

## Supported Algorithms

- BFS
- DFS
- BDS
- GBFS
- A*
- IDA*

## Notes

- The app expects the `models/` directory to contain the trained `.pth` files listed above.
- If `test_case/` is missing, the route selector will not have any traffic cases to load.
- The notebooks in this repo look like separate training experiments for accident detection and severity classification, with custom CNN, VGG16, and EfficientNet variants.
