🦴 TABULATOR: Bone Topology Analysis and Segmentation
🚀 Introduction
This project introduces TABULATOR, a powerful tool for the detailed analysis of bone topology through advanced image segmentation. The TABULATOR notebook is engineered to process bone scan images, meticulously identifying and quantifying key structural features. By calculating essential metrics such as bone length, volume, and connectivity, it provides a robust framework for in-depth analysis. This tool is invaluable for researchers, medical professionals, and data scientists who require precise, quantitative insights into bone architecture from medical imaging data.

✨ Features
🖼️ Image Preprocessing: Resizes and converts images into a high-contrast binary format, preparing them for detailed analysis.

💀 Skeletonization: Implements the skeletonize function from scikit-image to reduce bone structures to their essential topological skeleton.

📍 Node and Endpoint Detection: Pinpoints critical structural points, including junctions (find_nodes) and terminations (find_endpoints), within the bone's skeleton.

🔗 Topological Analysis: Goes beyond simple measurements by calculating Euclidean distances between nodes and intelligently matching endpoints to their nearest nodes.

📐 Geometric Measurements: Computes the volume of individual bone fragments and fits ellipses to provide an estimation of bone thickness and geometry.

📊 Data Export: Systematically saves all extracted data into CSV files, ready for statistical analysis and further research.

🎨 Advanced Visualization: Generates richly annotated images that clearly display the results, including skeletons, nodes, endpoints, and fitted circles and ellipses.

⚙️ How It Works
The notebook follows a sequential pipeline to analyze bone topology:

Image Loading and Preprocessing: The process begins by loading a TIFF image and resizing it to a standard dimension. It is then converted to a binary image using a threshold, which separates the bone from the background.

Skeletonization: The binary image is skeletonized to create a one-pixel-wide representation of the bone structure. This is crucial for accurately identifying nodes and endpoints.

Feature Detection:

Nodes: Junction points in the skeleton, where three or more branches meet, are identified using convolution to count neighboring pixels.

Endpoints: The ends of bone branches are detected by finding pixels with only one neighbor.

Topological and Geometric Analysis:

The notebook calculates the Euclidean distance between connected nodes.

It also computes the geodesic radius for each node, which helps in understanding the local thickness of the bone.

Ellipses are fitted to endpoint pairs to model the shape and orientation of bone segments.

Data Aggregation and Export: All the calculated data—including node coordinates, endpoint locations, distances, and ellipse parameters—is compiled and saved to CSV files for easy access and analysis.

💻 Code Highlights
Here are a few key functions from the notebook that drive the analysis:

Node Detection
Nodes are identified as pixels with three or more neighbors in the skeletonized image. This is achieved using convolution:

def find_nodes(skel: np.ndarray) -> List[Tuple[int, int]]:
    deg = neighbour_count(skel)
    node_mask = (skel == 1) & (deg >= 3)
    lbl, n_comp = label(node_mask)
    centroids = center_of_mass(node_mask, lbl, range(1, n_comp + 1))
    return [(int(round(r)), int(round(c))) for r, c in centroids]

Endpoint Detection
Endpoints are pixels with exactly one neighbor, marking the end of a bone branch:

def find_endpoints(skel: np.ndarray) -> List[Tuple[int, int]]:
    deg = neighbour_count(skel)
    return [tuple(p) for p in np.argwhere((skel == 1) & (deg == 1))]

Geodesic Radius Calculation
The geodesic radius at a node is determined using the distance transform on the binary image:

def geodesic_radius(binary_img, node):
    mask = (binary_img > 0).astype(np.uint8)
    dist = distance_transform_edt(mask)
    y, x = node
    return dist[y, x]

📋 Dependencies
This project relies on the following Python libraries:

numpy

opencv-python

scipy

pandas

scikit-image

seaborn

matplotlib

📦 Installation
To set up the project, clone this repository and install the required dependencies:

git clone <your-repository-url>
cd <your-repository-name>
pip install numpy opencv-python scipy pandas scikit-image seaborn matplotlib

🚀 Usage
Set Input Image Path: In the notebook, update the input_image variable with the path to your TIFF image file.

Execute the Notebook: Run the cells in the Jupyter Notebook in order.

Save Intermediate Images (Optional): You can choose to save the resized and binary images when prompted.

Review the Outputs: All generated files, including annotated images and CSV data, will be saved in the Single_image/ directory.

📂 Output Files
The analysis produces a comprehensive set of output files in the Single_image/ directory:

resize_image.png: The resized input image.

binary_image.png: The binary version of the image.

skeleton_image.png: The skeletonized representation of the bone structures.

node_image.png: Image with annotated nodes and endpoints.

node_image.csv: Data on detected nodes, endpoints, and their connections.

circle_image.png: Image with circles fitted to the detected nodes.

circle_image.csv: Data on the fitted circles.

ellipse_image.png: Image with ellipses fitted to bone structures.

ellipse_image.csv: Data on the fitted ellipses.

volume.csv: The volume of each detected bone shape.

summary.csv: A summary of the analysis results.

🤝 Contributing
Contributions are highly encouraged! Please submit a pull request or open an issue to discuss potential improvements.

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
