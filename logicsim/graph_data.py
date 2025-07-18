"""
Graph data structures and management for LogicSim
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class PointerState(Enum):
    """Enumeration for tracking pointer interaction state"""
    IDLE = "idle"
    PRESSED = "pressed"
    DRAGGING = "dragging"


@dataclass
class ConnectorDefinition:
    """Defines a connector template for a node type"""
    id: str
    x_offset_ratio: float  # 0.0 to 1.0, relative to node width
    y_offset_ratio: float  # 0.0 to 1.0, relative to node height
    is_input: bool
    
    def create_connector(self, width: float, height: float) -> 'Connector':
        """Create a Connector instance from this definition"""
        x_offset = self.x_offset_ratio * width - 4  # -4 for connector centering
        y_offset = self.y_offset_ratio * height - 4  # -4 for connector centering
        return Connector(self.id, x_offset, y_offset, self.is_input)


@dataclass
class NodeDefinition:
    """Defines a node type with its properties and connectors"""
    name: str
    label: str
    default_width: float
    default_height: float
    color: str
    connectors: List[ConnectorDefinition]


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
    node_type: str
    x: float
    y: float
    width: float
    height: float
    label: str
    color: str
    connectors: List[Connector]
    
    @classmethod
    def create(cls, id: str, node_definition: NodeDefinition, x: float, y: float, width: float = None, height: float = None, label: str = None) -> 'Node':
        """Create a node with auto-generated connectors based on node definition"""
        # Use defaults from definition if not provided
        if width is None:
            width = node_definition.default_width
        if height is None:
            height = node_definition.default_height
        if label is None:
            label = node_definition.label
        
        # Generate connectors from definition
        connectors = [conn_def.create_connector(width, height) for conn_def in node_definition.connectors]
        
        return cls(id, node_definition.name, x, y, width, height, label, node_definition.color, connectors)


@dataclass
class Connection:
    """Represents a connection between two connectors"""
    id: str
    from_node_id: str
    from_connector_id: str
    to_node_id: str
    to_connector_id: str


class NodeDefinitionRegistry:
    """Registry for managing available node definitions"""
    
    def __init__(self):
        self.definitions: Dict[str, NodeDefinition] = {}
        self._create_standard_definitions()
    
    def _create_standard_definitions(self):
        """Create standard node type definitions"""
        # Input node: single output connector on right, center
        self.definitions["input"] = NodeDefinition(
            name="input",
            label="INPUT",
            default_width=50.0,
            default_height=50.0,
            color="rgb(144, 238, 144)",  # Light green
            connectors=[
                ConnectorDefinition("out", 1.0, 0.5, False)  # Right side, center
            ]
        )
        
        # Output node: single input connector on left, center
        self.definitions["output"] = NodeDefinition(
            name="output",
            label="OUTPUT",
            default_width=50.0,
            default_height=50.0,
            color="rgb(255, 182, 193)",  # Light pink
            connectors=[
                ConnectorDefinition("in", 0.0, 0.5, True)  # Left side, center
            ]
        )
        
        # AND gate: two inputs on left, one output on right
        self.definitions["and"] = NodeDefinition(
            name="and",
            label="AND",
            default_width=80.0,
            default_height=60.0,
            color="rgb(224, 224, 224)",  # Light gray
            connectors=[
                ConnectorDefinition("in1", 0.0, 0.25, True),  # Left side, upper
                ConnectorDefinition("in2", 0.0, 0.75, True),  # Left side, lower
                ConnectorDefinition("out", 1.0, 0.5, False)   # Right side, center
            ]
        )
        
        # OR gate: two inputs on left, one output on right
        self.definitions["or"] = NodeDefinition(
            name="or",
            label="OR",
            default_width=80.0,
            default_height=60.0,
            color="rgb(224, 224, 224)",  # Light gray
            connectors=[
                ConnectorDefinition("in1", 0.0, 0.25, True),  # Left side, upper
                ConnectorDefinition("in2", 0.0, 0.75, True),  # Left side, lower
                ConnectorDefinition("out", 1.0, 0.5, False)   # Right side, center
            ]
        )
        
        # NOT gate: one input on left, one output on right
        self.definitions["not"] = NodeDefinition(
            name="not",
            label="NOT",
            default_width=80.0,
            default_height=60.0,
            color="rgb(224, 224, 224)",  # Light gray
            connectors=[
                ConnectorDefinition("in", 0.0, 0.5, True),   # Left side, center
                ConnectorDefinition("out", 1.0, 0.5, False)  # Right side, center
            ]
        )
    
    def get_definition(self, name: str) -> NodeDefinition:
        """Get a node definition by name"""
        if name not in self.definitions:
            raise ValueError(f"Unknown node type: {name}")
        return self.definitions[name]
    
    def list_definitions(self) -> List[str]:
        """List all available node definition names"""
        return list(self.definitions.keys())
    
    def add_definition(self, definition: NodeDefinition):
        """Add a custom node definition"""
        self.definitions[definition.name] = definition


# Global registry instance
NODE_REGISTRY = NodeDefinitionRegistry()


class GraphData:
    """Manages the graph data and provides methods to convert to Slint format"""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.connections: Dict[str, Connection] = {}
        self.selected_node_id: str | None = None  # Track currently selected node
        
        # Movement state tracking
        self.pointer_state: PointerState = PointerState.IDLE
        self.drag_start_pos: tuple[float, float] = (0.0, 0.0)
        self.drag_node_id: str | None = None
        self.drag_offset: tuple[float, float] = (0.0, 0.0)  # Offset within node when drag started
        self.movement_threshold: float = 5.0  # Minimum pixels to consider movement
        self.click_selected_different_node: bool = False  # Track if we selected a different node on click
        
        self.logger = logging.getLogger(__name__)
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.logger.debug(f"Added node: {node.id} ({node.node_type})")
    
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
    
    def is_point_in_node(self, node: Node, x: float, y: float) -> bool:
        """Check if point (x,y) is within node bounds"""
        return (node.x <= x <= node.x + node.width and 
                node.y <= y <= node.y + node.height)
    
    def get_node_at_position(self, x: float, y: float) -> str | None:
        """Get the ID of the node at the given position, or None if no node"""
        # Iterate through nodes in reverse order (later nodes drawn on top take priority)
        for node_id, node in reversed(list(self.nodes.items())):
            if self.is_point_in_node(node, x, y):
                return node_id
        return None
    
    def select_node(self, node_id: str) -> None:
        """Select a specific node by ID"""
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' does not exist")
        
        previous_selection = self.selected_node_id
        self.selected_node_id = node_id
        
        if previous_selection != node_id:
            self.logger.debug(f"Selected node: {node_id} (previously: {previous_selection})")
    
    def deselect_node(self) -> None:
        """Deselect the currently selected node"""
        if self.selected_node_id is not None:
            previous_selection = self.selected_node_id
            self.selected_node_id = None
            self.logger.debug(f"Deselected node: {previous_selection}")
    
    def get_selected_node(self) -> str | None:
        """Get the ID of the currently selected node"""
        return self.selected_node_id
    
    def calculate_movement_distance(self, start_x: float, start_y: float, end_x: float, end_y: float) -> float:
        """Calculate the distance between two points"""
        return math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    
    def move_node(self, node_id: str, new_x: float, new_y: float) -> None:
        """Move a node to a new position"""
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' does not exist")
        
        node = self.nodes[node_id]
        old_x, old_y = node.x, node.y
        node.x = new_x
        node.y = new_y
        
        self.logger.debug(f"Moved node {node_id} from ({old_x}, {old_y}) to ({new_x}, {new_y})")
    
    def handle_pointer_down(self, x: float, y: float) -> bool:
        """Handle pointer down event. Returns True if UI should be refreshed."""
        if self.pointer_state != PointerState.IDLE:
            self.logger.warning(f"Pointer down received while in state {self.pointer_state}")
            return False
        
        clicked_node_id = self.get_node_at_position(x, y)
        
        # Store drag start information
        self.drag_start_pos = (x, y)
        self.pointer_state = PointerState.PRESSED
        self.click_selected_different_node = False
        
        if clicked_node_id is not None:
            self.drag_node_id = clicked_node_id
            node = self.nodes[clicked_node_id]
            
            # Calculate offset within the node
            self.drag_offset = (x - node.x, y - node.y)
            
            # If clicking on an unselected node, select it immediately
            if clicked_node_id != self.selected_node_id:
                self.select_node(clicked_node_id)
                self.click_selected_different_node = True
                return True
        else:
            self.drag_node_id = None
            self.drag_offset = (0.0, 0.0)
        
        return False
    
    def handle_pointer_move(self, x: float, y: float) -> bool:
        """Handle pointer move event. Returns True if UI should be refreshed."""
        if self.pointer_state == PointerState.IDLE:
            return False
        
        # Calculate movement distance from start position
        distance = self.calculate_movement_distance(
            self.drag_start_pos[0], self.drag_start_pos[1], x, y
        )
        
        # Check if we should transition to dragging state
        if self.pointer_state == PointerState.PRESSED and distance > self.movement_threshold:
            self.pointer_state = PointerState.DRAGGING
            self.logger.debug(f"Started dragging node {self.drag_node_id}")
        
        # If we're dragging and have a node to move
        if self.pointer_state == PointerState.DRAGGING and self.drag_node_id is not None:
            # Calculate new position accounting for drag offset
            new_x = x - self.drag_offset[0]
            new_y = y - self.drag_offset[1]
            
            # Move the node
            self.move_node(self.drag_node_id, new_x, new_y)
            return True
        
        return False
    
    def handle_pointer_up(self, x: float, y: float) -> bool:
        """Handle pointer up event. Returns True if UI should be refreshed."""
        if self.pointer_state == PointerState.IDLE:
            return False
        
        was_dragging = self.pointer_state == PointerState.DRAGGING
        ui_needs_refresh = False
        
        # If we were only pressed (not dragging), handle as selection
        if self.pointer_state == PointerState.PRESSED:
            # This is a click without significant movement
            if self.drag_node_id is None:
                # Clicked on empty area - deselect current selection
                if self.selected_node_id is not None:
                    self.deselect_node()
                    ui_needs_refresh = True
            elif self.drag_node_id == self.selected_node_id and not self.click_selected_different_node:
                # Clicked on already selected node - deselect it
                # Only deselect if we didn't just select it (i.e., it was already selected)
                self.deselect_node()
                ui_needs_refresh = True
            # If we clicked on a different node, it was already selected in handle_pointer_down
        
        # Reset drag state
        self.pointer_state = PointerState.IDLE
        self.drag_start_pos = (0.0, 0.0)
        self.drag_node_id = None
        self.drag_offset = (0.0, 0.0)
        self.click_selected_different_node = False
        
        # Return True if we were dragging or if selection changed
        return was_dragging or ui_needs_refresh
    
    def handle_mouse_click(self, x: float, y: float) -> bool:
        """Handle mouse click at given coordinates. Returns True if selection changed."""
        clicked_node_id = self.get_node_at_position(x, y)
        previous_selection = self.selected_node_id
        
        if clicked_node_id is None:
            # Clicked on empty area - deselect current selection
            self.deselect_node()
        elif clicked_node_id == self.selected_node_id:
            # Clicked on already selected node - deselect it
            self.deselect_node()
        else:
            # Clicked on a different node - select it
            self.select_node(clicked_node_id)
        
        # Return True if selection state changed
        return previous_selection != self.selected_node_id
    
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
                "node_type": node.node_type,
                "x": node.x,
                "y": node.y,
                "width": node.width,
                "height": node.height,
                "label": node.label,
                "color": node.color,
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
            "connections": slint_connections,
            "selected_nodes": [self.selected_node_id] if self.selected_node_id else []
        }


def create_demo_graph() -> GraphData:
    """Create a demonstration graph with logic gates"""
    graph = GraphData()
    
    # Input nodes - using Node.create() with automatic connector generation
    input_a = Node.create("input_a", NODE_REGISTRY.get_definition("input"), 50, 50, label="A")
    input_b = Node.create("input_b", NODE_REGISTRY.get_definition("input"), 50, 150, label="B")
    input_c = Node.create("input_c", NODE_REGISTRY.get_definition("input"), 50, 250, label="C")
    
    # Logic gates - using Node.create() with automatic connector generation
    and_gate = Node.create("and_gate", NODE_REGISTRY.get_definition("and"), 200, 80, label="AND")
    or_gate = Node.create("or_gate", NODE_REGISTRY.get_definition("or"), 200, 220, label="OR")
    not_gate = Node.create("not_gate", NODE_REGISTRY.get_definition("not"), 400, 150, label="NOT")
    
    # Output nodes - using Node.create() with automatic connector generation
    output_node_a = Node.create("output_a", NODE_REGISTRY.get_definition("output"), 550, 175, label="OUT")
    output_node_b = Node.create("output_b", NODE_REGISTRY.get_definition("output"), 550, 375, label="OUT")
    
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