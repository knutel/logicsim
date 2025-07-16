"""
Graph data structures and management for LogicSim
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the graph"""
    INPUT = "input"
    OUTPUT = "output"
    AND_GATE = "and"
    OR_GATE = "or"
    NOT_GATE = "not"


@dataclass
class Connector:
    """Represents a connector point on a node"""
    id: str
    x_offset: float  # Offset from node position
    y_offset: float  # Offset from node position
    is_input: bool   # True for input connectors, False for output


@dataclass
class Node:
    """Represents a node in the graph"""
    id: str
    node_type: NodeType
    x: float
    y: float
    width: float
    height: float
    label: str
    connectors: List[Connector]


@dataclass
class Connection:
    """Represents a connection between two connectors"""
    id: str
    from_node_id: str
    from_connector_id: str
    to_node_id: str
    to_connector_id: str


class GraphData:
    """Manages the graph data and provides methods to convert to Slint format"""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.connections: Dict[str, Connection] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.logger.debug(f"Added node: {node.id} ({node.node_type.value})")
    
    def add_connection(self, connection: Connection) -> None:
        """Add a connection to the graph"""
        self.connections[connection.id] = connection
        self.logger.debug(f"Added connection: {connection.from_node_id}:{connection.from_connector_id} -> {connection.to_node_id}:{connection.to_connector_id}")
    
    def get_connector_absolute_position(self, node_id: str, connector_id: str) -> tuple[float, float]:
        """Get the absolute position of a connector"""
        node = self.nodes[node_id]
        connector = next(c for c in node.connectors if c.id == connector_id)
        
        abs_x = node.x + connector.x_offset
        abs_y = node.y + connector.y_offset
        
        return abs_x, abs_y
    
    def to_slint_format(self) -> Dict[str, Any]:
        """Convert graph data to format suitable for Slint"""
        
        # Convert nodes
        slint_nodes = []
        for node in self.nodes.values():
            slint_node = {
                "id": node.id,
                "node_type": node.node_type.value,
                "x": node.x,
                "y": node.y,
                "width": node.width,
                "height": node.height,
                "label": node.label,
                "connectors": [
                    {
                        "id": c.id,
                        "x_offset": c.x_offset,
                        "y_offset": c.y_offset,
                        "is_input": c.is_input
                    } for c in node.connectors
                ]
            }
            slint_nodes.append(slint_node)
        
        # Convert connections with calculated positions
        slint_connections = []
        for conn in self.connections.values():
            start_x, start_y = self.get_connector_absolute_position(conn.from_node_id, conn.from_connector_id)
            end_x, end_y = self.get_connector_absolute_position(conn.to_node_id, conn.to_connector_id)
            
            slint_connection = {
                "id": conn.id,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y
            }
            slint_connections.append(slint_connection)
        
        return {
            "nodes": slint_nodes,
            "connections": slint_connections
        }


def create_demo_graph() -> GraphData:
    """Create a demonstration graph with logic gates"""
    graph = GraphData()
    
    # Input nodes
    input_a = Node(
        id="input_a",
        node_type=NodeType.INPUT,
        x=50,
        y=50,
        width=50,
        height=50,
        label="A",
        connectors=[
            Connector(id="out", x_offset=46, y_offset=21, is_input=False)  # Right side, center
        ]
    )
    
    input_b = Node(
        id="input_b",
        node_type=NodeType.INPUT,
        x=50,
        y=150,
        width=50,
        height=50,
        label="B",
        connectors=[
            Connector(id="out", x_offset=46, y_offset=21, is_input=False)  # Right side, center
        ]
    )
    
    input_c = Node(
        id="input_c",
        node_type=NodeType.INPUT,
        x=50,
        y=250,
        width=50,
        height=50,
        label="C",
        connectors=[
            Connector(id="out", x_offset=46, y_offset=21, is_input=False)  # Right side, center
        ]
    )
    
    # AND gate
    and_gate = Node(
        id="and_gate",
        node_type=NodeType.AND_GATE,
        x=200,
        y=80,
        width=80,
        height=60,
        label="AND",
        connectors=[
            Connector(id="in1", x_offset=0, y_offset=19, is_input=True),   # Left side, top
            Connector(id="in2", x_offset=0, y_offset=39, is_input=True),   # Left side, bottom
            Connector(id="out", x_offset=76, y_offset=26, is_input=False)  # Right side, center
        ]
    )
    
    # OR gate
    or_gate = Node(
        id="or_gate",
        node_type=NodeType.OR_GATE,
        x=200,
        y=220,
        width=80,
        height=60,
        label="OR",
        connectors=[
            Connector(id="in1", x_offset=0, y_offset=19, is_input=True),   # Left side, top
            Connector(id="in2", x_offset=0, y_offset=39, is_input=True),   # Left side, bottom
            Connector(id="out", x_offset=76, y_offset=26, is_input=False)  # Right side, center
        ]
    )
    
    # NOT gate
    not_gate = Node(
        id="not_gate",
        node_type=NodeType.NOT_GATE,
        x=400,
        y=150,
        width=80,
        height=60,
        label="NOT",
        connectors=[
            Connector(id="in", x_offset=0, y_offset=19, is_input=True),    # Left side
            Connector(id="out", x_offset=76, y_offset=26, is_input=False)  # Right side
        ]
    )
    
    # Output node
    output_node = Node(
        id="output",
        node_type=NodeType.OUTPUT,
        x=550,
        y=175,
        width=50,
        height=50,
        label="OUT",
        connectors=[
            Connector(id="in", x_offset=0, y_offset=21, is_input=True)     # Left side, center
        ]
    )
    
    # Add nodes to graph
    graph.add_node(input_a)
    graph.add_node(input_b)
    graph.add_node(input_c)
    graph.add_node(and_gate)
    graph.add_node(or_gate)
    graph.add_node(not_gate)
    graph.add_node(output_node)
    
    # Add connections
    connections = [
        Connection("c1", "input_a", "out", "and_gate", "in1"),
        Connection("c2", "input_b", "out", "and_gate", "in2"),
        Connection("c3", "input_b", "out", "or_gate", "in1"),
        Connection("c4", "input_c", "out", "or_gate", "in2"),
        Connection("c5", "and_gate", "out", "not_gate", "in"),
        Connection("c6", "not_gate", "out", "output", "in")
    ]
    
    for conn in connections:
        graph.add_connection(conn)
    
    return graph