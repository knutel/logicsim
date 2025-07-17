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
    
    def get_absolute_position(self, node_x: float, node_y: float) -> tuple[float, float]:
        """Calculate absolute position given node position"""
        return node_x + self.x_offset, node_y + self.y_offset


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
    
    @classmethod
    def create(cls, id: str, node_type: NodeType, x: float, y: float, width: float, height: float, label: str) -> 'Node':
        """Create a node with auto-generated connectors based on node type"""
        connectors = create_connectors_for_node_type(node_type, width, height)
        return cls(id, node_type, x, y, width, height, label, connectors)


@dataclass
class Connection:
    """Represents a connection between two connectors"""
    id: str
    from_node_id: str
    from_connector_id: str
    to_node_id: str
    to_connector_id: str


def create_connectors_for_node_type(node_type: NodeType, width: float, height: float) -> List[Connector]:
    """Create connectors for a node based on its type and dimensions"""
    connectors = []
    
    if node_type == NodeType.INPUT:
        # Input nodes have one output connector on the right side, center
        connectors.append(Connector(
            id="out",
            x_offset=width - 4,  # 4px from right edge (connector size/2)
            y_offset=height / 2 - 4,  # Center vertically, adjusted for connector size
            is_input=False
        ))
    
    elif node_type == NodeType.OUTPUT:
        # Output nodes have one input connector on the left side, center
        connectors.append(Connector(
            id="in",
            x_offset=-4,  # 4px outside left edge
            y_offset=height / 2 - 4,  # Center vertically, adjusted for connector size
            is_input=True
        ))
    
    elif node_type in [NodeType.AND_GATE, NodeType.OR_GATE]:
        # AND/OR gates have two input connectors on the left and one output on the right
        connectors.extend([
            Connector(
                id="in1",
                x_offset=-4,  # 4px outside left edge
                y_offset=15,  # Top input position
                is_input=True
            ),
            Connector(
                id="in2",
                x_offset=-4,  # 4px outside left edge
                y_offset=35,  # Bottom input position
                is_input=True
            ),
            Connector(
                id="out",
                x_offset=width - 4,  # 4px from right edge
                y_offset=height / 2 - 4,  # Center vertically
                is_input=False
            )
        ])
    
    elif node_type == NodeType.NOT_GATE:
        # NOT gates have one input connector on the left and one output on the right
        connectors.extend([
            Connector(
                id="in",
                x_offset=-4,  # 4px outside left edge
                y_offset=height / 2 - 4,  # Center vertically
                is_input=True
            ),
            Connector(
                id="out",
                x_offset=width - 4,  # 4px from right edge
                y_offset=height / 2 - 4,  # Center vertically
                is_input=False
            )
        ])
    
    return connectors


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
        
        abs_x = node.x + connector.x_offset + 3
        abs_y = node.y + connector.y_offset + 3
        
        return abs_x, abs_y
    
    def to_slint_format(self) -> Dict[str, Any]:
        """Convert graph data to format suitable for Slint"""
        
        # Convert nodes with connector absolute positions
        slint_nodes = []
        for node in self.nodes.values():
            # Calculate absolute positions for all connectors
            connectors_data = []
            for connector in node.connectors:
                abs_x, abs_y = connector.get_absolute_position(node.x, node.y)
                connectors_data.append({
                    "id": connector.id,
                    "x": abs_x,
                    "y": abs_y,
                    "is_input": connector.is_input
                })
            
            slint_node = {
                "id": node.id,
                "node_type": node.node_type.value,
                "x": node.x,
                "y": node.y,
                "width": node.width,
                "height": node.height,
                "label": node.label,
                "connectors": connectors_data
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
    
    # Input nodes - using Node.create() with automatic connector generation
    input_a = Node.create("input_a", NodeType.INPUT, 50, 50, 50, 50, "A")
    input_b = Node.create("input_b", NodeType.INPUT, 50, 150, 50, 50, "B")
    input_c = Node.create("input_c", NodeType.INPUT, 50, 250, 50, 50, "C")
    
    # Logic gates - using Node.create() with automatic connector generation
    and_gate = Node.create("and_gate", NodeType.AND_GATE, 200, 80, 80, 60, "AND")
    or_gate = Node.create("or_gate", NodeType.OR_GATE, 200, 220, 80, 60, "OR")
    not_gate = Node.create("not_gate", NodeType.NOT_GATE, 400, 150, 80, 60, "NOT")
    
    # Output nodes - using Node.create() with automatic connector generation
    output_node_a = Node.create("output_a", NodeType.OUTPUT, 550, 175, 50, 50, "OUT")
    output_node_b = Node.create("output_b", NodeType.OUTPUT, 550, 375, 50, 50, "OUT")
    
    # Add nodes to graph
    graph.add_node(input_a)
    graph.add_node(input_b)
    graph.add_node(input_c)
    graph.add_node(and_gate)
    graph.add_node(or_gate)
    graph.add_node(not_gate)
    graph.add_node(output_node_a)
    graph.add_node(output_node_b)
    
    # Add connections
    connections = [
        Connection("c1", "input_a", "out", "and_gate", "in1"),
        Connection("c2", "input_b", "out", "and_gate", "in2"),
        Connection("c3", "input_b", "out", "or_gate", "in1"),
        Connection("c4", "input_c", "out", "or_gate", "in2"),
        Connection("c5", "and_gate", "out", "not_gate", "in"),
        Connection("c6", "not_gate", "out", "output_a", "in"),
        Connection("c7", "or_gate", "out", "output_b", "in")
    ]
    
    for conn in connections:
        graph.add_connection(conn)
    
    return graph