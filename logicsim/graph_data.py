"""
Graph data structures and management for LogicSim
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum
import logging
import math
import time

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
        self.selected_connection_id: str | None = None  # Track currently selected connection
        
        # Movement state tracking
        self.pointer_state: PointerState = PointerState.IDLE
        self.drag_start_pos: tuple[float, float] = (0.0, 0.0)
        self.drag_node_id: str | None = None
        self.drag_offset: tuple[float, float] = (0.0, 0.0)  # Offset within node when drag started
        self.movement_threshold: float = 5.0  # Minimum pixels to consider movement
        self.click_selected_different_node: bool = False  # Track if we selected a different node on click
        self.click_selected_different_connection: bool = False  # Track if we selected a different connection on click
        
        # Label editing state tracking
        self.editing_node_id: str | None = None
        self.editing_text: str = ""
        self.last_click_time: float = 0.0
        self.double_click_threshold: float = 0.5  # seconds
        
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
    
    def is_point_on_connection(self, connection: Connection, x: float, y: float, tolerance: float = 8.0) -> bool:
        """Check if point (x,y) is within tolerance distance of connection line"""
        # Get connection endpoints
        start_x, start_y = self.get_connector_absolute_position(connection.from_node_id, connection.from_connector_id)
        end_x, end_y = self.get_connector_absolute_position(connection.to_node_id, connection.to_connector_id)
        
        # Calculate distance from point to line segment
        distance = self.point_to_line_segment_distance(x, y, start_x, start_y, end_x, end_y)
        return distance <= tolerance
    
    def point_to_line_segment_distance(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate the distance from a point to a line segment"""
        # Line segment vector
        dx = x2 - x1
        dy = y2 - y1
        
        # If line segment is actually a point
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        
        # Calculate parameter t that represents position along the line segment
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        # Find the closest point on the line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Calculate distance from point to closest point on segment
        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
    
    def get_connection_at_position(self, x: float, y: float) -> str | None:
        """Get the ID of the connection at the given position, or None if no connection"""
        # Iterate through connections in reverse order (later connections drawn on top take priority)
        for connection_id, connection in reversed(list(self.connections.items())):
            if self.is_point_on_connection(connection, x, y):
                return connection_id
        return None
    
    def select_node(self, node_id: str) -> None:
        """Select a specific node by ID"""
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' does not exist")
        
        previous_selection = self.selected_node_id
        
        # Deselect any selected connection (mutual exclusion)
        if self.selected_connection_id is not None:
            self.deselect_connection()
        
        self.selected_node_id = node_id
        
        if previous_selection != node_id:
            self.logger.debug(f"Selected node: {node_id} (previously: {previous_selection})")
    
    def deselect_node(self) -> None:
        """Deselect the currently selected node"""
        if self.selected_node_id is not None:
            previous_selection = self.selected_node_id
            
            # Cancel any active label editing for the deselected node
            if self.editing_node_id == previous_selection:
                self.cancel_label_edit()
                self.logger.debug(f"Cancelled label editing for deselected node: {previous_selection}")
            
            self.selected_node_id = None
            self.logger.debug(f"Deselected node: {previous_selection}")
    
    def get_selected_node(self) -> str | None:
        """Get the ID of the currently selected node"""
        return self.selected_node_id
    
    def select_connection(self, connection_id: str) -> None:
        """Select a specific connection by ID"""
        if connection_id not in self.connections:
            raise ValueError(f"Connection with ID '{connection_id}' does not exist")
        
        previous_selection = self.selected_connection_id
        
        # Deselect any selected node (mutual exclusion)
        if self.selected_node_id is not None:
            self.deselect_node()
        
        self.selected_connection_id = connection_id
        
        if previous_selection != connection_id:
            self.logger.debug(f"Selected connection: {connection_id} (previously: {previous_selection})")
    
    def deselect_connection(self) -> None:
        """Deselect the currently selected connection"""
        if self.selected_connection_id is not None:
            previous_selection = self.selected_connection_id
            self.selected_connection_id = None
            self.logger.debug(f"Deselected connection: {previous_selection}")
    
    def get_selected_connection(self) -> str | None:
        """Get the ID of the currently selected connection"""
        return self.selected_connection_id
    
    def calculate_movement_distance(self, start_x: float, start_y: float, end_x: float, end_y: float) -> float:
        """Calculate the distance between two points"""
        return math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its connections. Returns True if UI should be refreshed."""
        if node_id not in self.nodes:
            self.logger.warning(f"Cannot delete non-existent node: {node_id}")
            return False
        
        # Find all connections that reference this node
        connections_to_delete = []
        for connection_id, connection in self.connections.items():
            if connection.from_node_id == node_id or connection.to_node_id == node_id:
                connections_to_delete.append(connection_id)
        
        # Delete all connections that reference this node
        for connection_id in connections_to_delete:
            del self.connections[connection_id]
            self.logger.debug(f"Cascade deleted connection: {connection_id}")
        
        # Clean up selection state
        if self.selected_node_id == node_id:
            self.selected_node_id = None
            self.logger.debug(f"Cleared node selection due to deletion: {node_id}")
        
        # Clean up editing state
        if self.editing_node_id == node_id:
            self.editing_node_id = None
            self.editing_text = ""
            self.logger.debug(f"Cancelled label editing due to node deletion: {node_id}")
        
        # Delete the node
        del self.nodes[node_id]
        self.logger.debug(f"Deleted node: {node_id} (cascade deleted {len(connections_to_delete)} connections)")
        
        return True  # Always refresh UI after node deletion
    
    def delete_connection(self, connection_id: str) -> bool:
        """Delete a connection. Returns True if UI should be refreshed."""
        if connection_id not in self.connections:
            self.logger.warning(f"Cannot delete non-existent connection: {connection_id}")
            return False
        
        # Clean up selection state
        if self.selected_connection_id == connection_id:
            self.selected_connection_id = None
            self.logger.debug(f"Cleared connection selection due to deletion: {connection_id}")
        
        # Delete the connection
        del self.connections[connection_id]
        self.logger.debug(f"Deleted connection: {connection_id}")
        
        return True  # Always refresh UI after connection deletion
    
    def delete_selected(self) -> bool:
        """Delete the currently selected node or connection. Returns True if something was deleted."""
        if self.selected_node_id is not None:
            return self.delete_node(self.selected_node_id)
        elif self.selected_connection_id is not None:
            return self.delete_connection(self.selected_connection_id)
        else:
            self.logger.debug("Delete requested but nothing is selected")
            return False
    
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
        self.click_selected_different_connection = False
        
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
            
            # Check for connection selection if no node was clicked
            clicked_connection_id = self.get_connection_at_position(x, y)
            if clicked_connection_id is not None:
                # If clicking on an unselected connection, select it immediately
                if clicked_connection_id != self.selected_connection_id:
                    self.select_connection(clicked_connection_id)
                    self.click_selected_different_connection = True
                    return True
        
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
            if self.drag_node_id is None and not self.click_selected_different_connection:
                # Check if we clicked on a connection that was already selected
                clicked_connection_id = self.get_connection_at_position(x, y)
                if clicked_connection_id is not None and clicked_connection_id == self.selected_connection_id:
                    # Clicked on already selected connection - deselect it
                    self.deselect_connection()
                    ui_needs_refresh = True
                elif clicked_connection_id is None:
                    # Clicked on empty area - deselect current selection
                    if self.selected_node_id is not None:
                        self.deselect_node()
                        ui_needs_refresh = True
                    elif self.selected_connection_id is not None:
                        self.deselect_connection()
                        ui_needs_refresh = True
            elif self.drag_node_id == self.selected_node_id and not self.click_selected_different_node:
                # Clicked on already selected node - deselect it
                # Only deselect if we didn't just select it (i.e., it was already selected)
                self.deselect_node()
                ui_needs_refresh = True
            # If we clicked on a different node or connection, it was already selected in handle_pointer_down
        
        # Reset drag state
        self.pointer_state = PointerState.IDLE
        self.drag_start_pos = (0.0, 0.0)
        self.drag_node_id = None
        self.drag_offset = (0.0, 0.0)
        self.click_selected_different_node = False
        self.click_selected_different_connection = False
        
        # Return True if we were dragging or if selection changed
        return was_dragging or ui_needs_refresh
    
    def start_label_edit(self, node_id: str) -> bool:
        """Start editing a node's label. Returns True if UI should be refreshed."""
        if node_id not in self.nodes:
            self.logger.warning(f"Cannot edit label for non-existent node: {node_id}")
            return False
        
        # Cancel any existing edit
        if self.editing_node_id is not None:
            self.cancel_label_edit()
        
        node = self.nodes[node_id]
        self.editing_node_id = node_id
        self.editing_text = node.label
        
        self.logger.debug(f"Started editing label for node {node_id}: '{node.label}'")
        return True
    
    def complete_label_edit(self, node_id: str, new_label: str) -> bool:
        """Complete label editing and update node. Returns True if UI should be refreshed."""
        if self.editing_node_id != node_id:
            self.logger.warning(f"Attempted to complete edit for wrong node: {node_id} (editing: {self.editing_node_id})")
            return False
        
        if node_id not in self.nodes:
            self.logger.warning(f"Cannot complete edit for non-existent node: {node_id}")
            self.cancel_label_edit()
            return True
        
        # Update the node's label
        old_label = self.nodes[node_id].label
        self.nodes[node_id].label = new_label.strip()
        
        # Clear editing state
        self.editing_node_id = None
        self.editing_text = ""
        
        self.logger.debug(f"Completed editing label for node {node_id}: '{old_label}' -> '{new_label.strip()}'")
        return True
    
    def cancel_label_edit(self) -> bool:
        """Cancel label editing without saving. Returns True if UI should be refreshed."""
        if self.editing_node_id is None:
            return False
        
        self.logger.debug(f"Cancelled editing label for node {self.editing_node_id}")
        
        self.editing_node_id = None
        self.editing_text = ""
        return True
    
    def update_editing_text(self, text: str) -> None:
        """Update the editing text as user types."""
        self.editing_text = text
    
    def handle_double_click(self, x: float, y: float) -> bool:
        """Handle double-click for label editing. Returns True if UI should be refreshed."""
        clicked_node_id = self.get_node_at_position(x, y)
        
        if clicked_node_id is not None:
            # Start editing the clicked node's label
            return self.start_label_edit(clicked_node_id)
        
        return False
    
    def is_double_click(self, current_time: float) -> bool:
        """Check if current click is a double-click based on timing."""
        time_diff = current_time - self.last_click_time
        self.last_click_time = current_time
        
        return time_diff <= self.double_click_threshold
    
    def handle_mouse_click(self, x: float, y: float) -> bool:
        """Handle mouse click at given coordinates. Returns True if selection changed."""
        clicked_node_id = self.get_node_at_position(x, y)
        previous_node_selection = self.selected_node_id
        previous_connection_selection = self.selected_connection_id
        
        if clicked_node_id is not None:
            # Clicked on a node
            if clicked_node_id == self.selected_node_id:
                # Clicked on already selected node - deselect it
                self.deselect_node()
            else:
                # Clicked on a different node - select it
                self.select_node(clicked_node_id)
        else:
            # No node clicked, check for connection
            clicked_connection_id = self.get_connection_at_position(x, y)
            if clicked_connection_id is not None:
                # Clicked on a connection
                if clicked_connection_id == self.selected_connection_id:
                    # Clicked on already selected connection - deselect it
                    self.deselect_connection()
                else:
                    # Clicked on a different connection - select it
                    self.select_connection(clicked_connection_id)
            else:
                # Clicked on empty area - deselect current selection
                if self.selected_node_id is not None:
                    self.deselect_node()
                elif self.selected_connection_id is not None:
                    self.deselect_connection()
        
        # Return True if selection state changed
        return (previous_node_selection != self.selected_node_id or 
                previous_connection_selection != self.selected_connection_id)
    
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
            "selected_nodes": [self.selected_node_id] if self.selected_node_id else [],
            "selected_connections": [self.selected_connection_id] if self.selected_connection_id else [],
            "editing_node_id": self.editing_node_id or "",
            "editing_text": self.editing_text
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