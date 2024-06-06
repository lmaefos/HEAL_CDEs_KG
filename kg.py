import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter
from pyvis.network import Network
from IPython.core.display import display, HTML

# Read the new CSV file
descriptors_data = pd.read_csv("HEAL CDE Team_CoreMeasures.Study.PI_as of 2024.05-13_forKG.csv")

# Convert DataFrame to a list of lists with all entries as strings
project_data = descriptors_data.astype(str).values.tolist()

# Define a color mapping for each column header
color_map = {
    'Core CDE Measures': '#4b0082',  # Dark Purple
    'Domain': '#dda0dd',            # Light Purple
    'Questionnaire': '#ff1493',     # Dark Pink
    'Study Name': '#1f77b4',        # Blue (if needed)
    'PI Name': '#2ca02c',           # Green (if needed)
    'HEAL Research Program': '#ffb6c1' # Light Pink
}

shape_map = {
    'Core CDE Measures': 'dot',      # Circle
    'Domain': 'ellipse',             # Ellipse
    'Questionnaire': 'square',       # Square
    'HEAL Research Program': 'triangle',  # Triangle
    'Study Name': 'text',            # Text
    'PI Name': 'text'                # Text
}

# Create a Network graph object
net = Network(notebook=True, width="1000px", height="600px", cdn_resources='remote', font_color='white', bgcolor="black", select_menu=True, filter_menu=True)
st.title('Interactive Knowledge Network of HEAL Core CDEs')

# Add description before presenting the knowledge graph
st.markdown("""
This dynamic tool presents a knowledge graph and is designed to help researchers understand [common data elements (CDEs), particularly those that pertain to HEAL Pain Research](https://heal.nih.gov/data/common-data-elements). The tool allows users to explore relationships among CDEs and identify patterns in their use across studies.
            
## Understanding Nodes and Edges

- **Nodes** (aka vertices) are the fundamental units in a graph. They represent entities or objects. Nodes can have properties like size, color, label, etc., which help in identifying or categorizing them. Each node can have various properties like size, color, label, etc.
- **Edges** (aka links) represent the connections or relationships between nodes. They define how nodes are related, such as a core CDE measure being associated with a particular domain or questionnaire.

## Understanding Pathways
A **Pathway** is a sequence of nodes connected by edges, representing a specific route or series of connections in the graph. Pathways can illustrate different patterns or sequences of relationships. There can be multiple pathways showing various possible routes or sequences between nodes.
A pathway might show how a "Core CDE Measure" leads to a "Domain," which then connects to a "Questionnaire," forming a sequence.

- "<span style="color:#4b0082;">Core CDE Measure</span>" --> "<span style="color:#dda0dd;">Domain</span>" --> "<span style="color:#ff1493;">Questionnaire</span>"

## Graph Key 

### Node Colors and Shapes:
- `Core CDE Measures`:
  - <span style="color:#4b0082;">&#x25CF;</span> Dark Purple (Circle)
  - **Size**: Proportional to their frequency in the reported CDE use; bigger circles indicate greater usage across studies.
  - **Purpose**: These nodes represent the core measures and are central to the graph.

- `Domain`:
  - <span style="color:#dda0dd;">⬮</span> Light Purple (Ellipse)
  - **Size**: Fixed size.
  - **Purpose**: These nodes categorize different domains within the data.

- `Questionnaire`:
  - <span style="color:#ff1493;">&#x25A0;</span> Dark Pink (Square)
  - **Size**: Fixed size.
  - **Purpose**: These nodes represent various questionnaires.

- `HEAL Research Program`:
  - <span style="color:#ffb6c1;">&#x25B2;</span> Light Pink (Triangle)
  - **Size**: Fixed size.
  - **Purpose**: These nodes represent different programs under the HEAL (Helping to End Addiction Long-term) initiative.

- `Study Name` and `PI Name`:
  - Text (no specific shape)
  - **Size**: Fixed size.
  - **Purpose**: These nodes represent the names of different studies and principal investigators.

### Edges Colors
  - Edges inherit the color of the nodes they connect. For example, an edge connecting a "Core CDE Measures" node (<span style="color:#4b0082;">&#x25CF;</span>) to a "Domain" node (<span style="color:#dda0dd;">⬮</span>) will inherit the color properties **_from_** the connected nodes.
    - For example, if you select the questionnaire (<span style="color:#ff1493;">&#x25A0;</span>), 'PROMIS', you will see:
      - <span style="color: #4b0082;">Dark purple</span> edges: These connect to 'Core CDE Measures' nodes related to 'PROMIS'.
      - <span style="color: #dda0dd;">Light purple</span> edges: These connect 'PROMIS' to nodes categorized under 'Domain'.
      - <span style="color: #ff1493;">Dark pink</span> edges: These connect 'PROMIS' to other nodes categorized under 'Questionnaire'.
  - Study Name and PI Name are represented as text without associated shapes or colors, thus they have the default <span style="color: #0000ff;">blue</span> edges.

## Explore!
This interactive knowledge graph is designed to let researchers highlight and explore individual nodes and their connections. Users can search and navigate through the graph using simple properties like color, shape, and size, easing identification of patterns, relationships, and focal points of interest.
The graph is continually being refined and updated.

For more information on using the filter feature, [explanation below](#selecting-a-node).

            
""", unsafe_allow_html=True)

def create_knowledge_graph(data, columns):
    all_descriptors = [descriptor for entry in data for descriptor in entry if descriptor not in ['nan', '']]
    descriptor_frequency = Counter(all_descriptors)
    max_frequency = max(descriptor_frequency.values())

    added_nodes = set()  # Track added nodes to avoid duplicates and ensure node existence before adding edges
    min_size = 15 # Minimum size for nodes

    for entry in data:
        core_cde_measure = entry[columns.get_loc('Core CDE Measures')]
        if core_cde_measure not in ['nan', '']:
            if core_cde_measure not in added_nodes:
                net.add_node(core_cde_measure, label=core_cde_measure, color=color_map['Core CDE Measures'],
                             size=30 * (descriptor_frequency[core_cde_measure] / max_frequency), shape=shape_map['Core CDE Measures'])
                added_nodes.add(core_cde_measure)

        # Create and connect nodes for each category
        entries_dict = {}
        for key in ['Domain', 'Questionnaire', 'HEAL Research Program', 'Study Name', 'PI Name']:
            entries = entry[columns.get_loc(key)]
            entries_dict[key] = process_entries(entries, key, added_nodes)

        # Create edges between Core CDE Measure and other nodes, and between all other node pairs
        for key, nodes in entries_dict.items():
            for node in nodes:
                net.add_edge(core_cde_measure, node)  # Link Core CDE Measure to each node
                # Link each node to every other node
                for other_key, other_nodes in entries_dict.items():
                    if other_key != key:
                        for other_node in other_nodes:
                            net.add_edge(node, other_node)

    # Configure physics options
    net.set_options("""
    var options = {
        "nodes": {
            "borderWidth": 1,
            "borderWidthSelected": 2,
            "font": {
                "size": 14,
                "color": "white"
            }
        },
        "edges": {
            "color": {
                "inherit": true
            },
            "smooth": {
                "type": "continuous"
            }
        },
        "physics": {
        "enabled": true,
        "stabilization": {
          "enabled": true,
          "iterations": 2000,
          "updateInterval": 100
        },
        "barnesHut": {
          "gravitationalConstant": -155000,
          "centralGravity": 0.007,
          "springLength": 1500,
          "springConstant": 0.0095,
          "damping": 0.3
        }
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # Generate the HTML content as a string
    html_content = net.generate_html()

    # Write the HTML content to a file with utf-8 encoding
    with open('knowledge_graph.html', 'w', encoding='utf-8') as file:
        file.write(html_content)

def process_entries(entries, entry_type, added_nodes):
    item_nodes = []
    if entries not in ['nan', '']:
        items = entries.split(',')
        for item in items:
            item = item.strip()
            if item and item not in added_nodes:
                net.add_node(item, label=item, color=color_map[entry_type],
                             size=15, shape=shape_map[entry_type])  # Fixed size for all other nodes
                added_nodes.add(item)
                item_nodes.append(item)
            elif item:
                item_nodes.append(item)
    return item_nodes

create_knowledge_graph(project_data, descriptors_data.columns)


# Display the graph in the Streamlit app
html_path = 'knowledge_graph.html'
try:
    with open(html_path, 'r', encoding='utf-8') as HtmlFile:
        html_content = HtmlFile.read()

    components.html(html_content, height=800, width=1000)

except FileNotFoundError:
    st.warning(f"HTML file not found at {html_path}.")
except Exception as e:
    st.error(f"An error occurred while reading the HTML file: {e}")

# Details about filtering 
st.markdown("""
### Selecting a Node

- **Select a Node by ID**: top row of filtering feature
  - You can use the dropdown menu labeled "Select a Node by ID" to choose a specific node. This will highlight the node and its connections, helping you focus on a particular part of the graph. 
  - See [Guide to Possible Selection Choices below](#guide-to-possible-selection-choices)
  - See the table below for the filter property cheat sheet.

### Selecting an Edge

- **Select a Network Item (Edge)**:
  - When you select "edge" from the "Select a network item" dropdown, you can choose properties related to the edges. This is useful for highlighting or filtering specific relationships in the graph.

- **Edge Properties**:
  - Properties for edges might include from (starting node), to (ending node), color, width, etc.
  - Example: You might filter edges to show only those connected to a particular node or of a specific color.

### Filtering and Resetting

- **Filter**:
  - After selecting the node or edge and specifying the properties, clicking the "Filter" button will apply the filter to the graph, highlighting the nodes or edges that match your criteria.

- **Reset Selection**:
  - Clicking "Reset Selection" will clear the current filter, returning the graph to its default state where all nodes and edges are visible.

### Practical Use Case

- Highlight Specific **Nodes**: let's say you want to highlight relationships between all research programs and CDEs. You would:
  - Select "node" from "Select a network item".
  - Choose `shape` from "Select a property".
  - Enter 'Triangle' and 'Dot' in "Select value(s)" and click "Filter".

- Highlight Specific **Edges**: to highlight edges to focus on the pathway:
  - Select "edge" from "Select a network item".
  - Choose 'from' or 'to' in "Select a property".
  - See the table below for the filter property cheat sheet.
 
""")

# Data for the table
data = {
    "Network Item": ["Node", "Node", "Node", "Node", "Node", "Node", "Node", "Edge", "Edge", "Edge"],
    "Property": ["color", "font", "id", "label", "shape", "shape", "shape", "from", "to", "id"],
    "Possible Selections": ["ignore", "ignore", "Any unique values", "ignore", "Dot - CDEs", "Ellipse - Domain", "Triangle - Res. prog.", "Study name, Research program, CDE name, Domain, Questionnaire", "Research program, PI Name, Study Name, Domain, Questionnaire", "ignore"],
    "Description": ["Not human readable", "Not human readable", "Expansion of the ‘Select Node’, slowly building connections", "Glitchy - do not use", "Based on selected shapes, will show relationship between shapes", "", "", "Study name alone will show connection to PI Name (1 edge)\nRes. prog alone will show connection to study name and PI name (2 edges)\nCDE name alone will show connections to research programs, study names, domain, and questionnaire (4 edges)\nDomain alone will show connections to research programs, study and PI name (4 edges)\nQuestionnaire alone will show connections to domain, research programs, study and PI name (4 edges)", "Research program alone will show connections to CDEs, Questionnaires, and Domain (3 edges)\nPI Names alone will show connections to study name, CDEs, Domain, Questionnaire, Research Program, Questionnaire (6 edges)\nStudy name alone will show connections to research program, CDEs, domain, questionnaire (4 edges)\nDomain alone will show a connection to CDEs (1 edge)\nQuestionnaire alone will show connections to domain and CDEs (2 edges)", "Not human readable"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the table
st.table(df)

# Generate and display the guide table for possible selection choices
unique_values = {column: descriptors_data[column].unique().tolist() for column in descriptors_data.columns}
st.title('Guide to Possible Selection Choices')

# Iterate through each column and create a table for its unique values
for column in descriptors_data.columns:
    # Extract unique values for the column, ensuring to drop NaNs
    unique_values = descriptors_data[column].dropna().unique()
    
    # Handle splitting for specific columns like 'Domain'
    if column == 'Domain':
        split_values = []
        for value in unique_values:
            # Split by comma and strip spaces
            split_values.extend([item.strip() for item in value.split(',')])
        # Get unique values from split results
        unique_values = pd.Series(split_values).unique()
    
    # Create a DataFrame for the unique values
    df_unique = pd.DataFrame(unique_values, columns=[column])
    
    # Display a subheader for the column name
    st.subheader(f"Unique values in {column}")
    
    # Display DataFrame as a table
    st.table(df_unique)