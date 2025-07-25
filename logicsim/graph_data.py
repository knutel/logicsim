"""
Graph data structures and management for LogicSim
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
from collections import deque
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
    value: Optional[bool] = None  # Node state for simulation
    
    @classmethod
    def create(cls, id: str, node_definition: NodeDefinition, x: float, y: float, width: float = None, height: float = None, label: str = None, value: Optional[bool] = None) -> 'Node':
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
        
        return cls(id, node_definition.name, x, y, width, height, label, node_definition.color, connectors, value)


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
        
        # NOR gate: two inputs on left, one output on right
        self.definitions["nor"] = NodeDefinition(
            name="nor",
            label="NOR",
            default_width=80.0,
            default_height=60.0,
            color="rgb(255, 160, 160)",  # Light red to distinguish from OR
            connectors=[
                ConnectorDefinition("in1", 0.0, 0.25, True),  # Left side, upper
                ConnectorDefinition("in2", 0.0, 0.75, True),  # Left side, lower
                ConnectorDefinition("out", 1.0, 0.5, False)   # Right side, center
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
        
        # Connection creation state tracking
        self.creating_connection: bool = False
        self.connection_start_node_id: str | None = None
        self.connection_start_connector_id: str | None = None
        self.pending_connection_end_x: float = 0.0
        self.pending_connection_end_y: float = 0.0
        
        # Toolbox and node creation state tracking
        self.selected_node_type: str | None = None
        self.toolbox_creation_mode: bool = False
        
        # Simulation mode state tracking
        self.simulation_mode: bool = False  # False = Edit Mode, True = Simulation Mode
        
        # Feedback simulation configuration
        self.max_simulation_iterations: int = 100  # Maximum iterations for convergence
        self.convergence_tolerance: int = 3  # Stable iterations required for convergence
        
        self.logger = logging.getLogger(__name__)
        
        # Import here to avoid circular imports
        from .simulation import CircuitEvaluator
        self.evaluator = CircuitEvaluator()
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.logger.debug(f"Added node: {node.id} ({node.node_type})")
    
    def add_connection(self, connection: Connection) -> None:
        """Add a connection to the graph"""
        self.connections[connection.id] = connection
        self.logger.debug(f"Added connection: {connection.from_node_id}:{connection.from_connector_id} -> {connection.to_node_id}:{connection.to_connector_id}")
    
    def set_input_value(self, node_id: str, value: bool) -> bool:
        """
        Set the value of an input node.
        
        Args:
            node_id: ID of the input node to set
            value: Boolean value to assign to the input node
            
        Returns:
            bool: True if UI should be refreshed
            
        Raises:
            ValueError: If node doesn't exist or is not an input node
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' does not exist")
        
        node = self.nodes[node_id]
        if node.node_type != "input":
            raise ValueError(f"Node '{node_id}' is not an input node (type: {node.node_type})")
        
        old_value = node.value
        node.value = value
        
        self.logger.debug(f"Set input node '{node_id}' value: {old_value} -> {value}")
        return True  # UI should refresh to show new value
    
    def get_node_value(self, node_id: str) -> Optional[bool]:
        """
        Get the current value of a node.
        
        Args:
            node_id: ID of the node to get value from
            
        Returns:
            Optional[bool]: The node's current value, or None if node doesn't exist or has no value
        """
        if node_id not in self.nodes:
            return None
        
        return self.nodes[node_id].value
    
    def _get_evaluation_order(self) -> List[str]:
        """
        Get the order in which nodes should be evaluated using topological sorting.
        
        Returns:
            List[str]: List of node IDs in evaluation order
            
        Raises:
            ValueError: If circular dependencies are detected
        """
        # Build adjacency list representing dependencies
        # dependency_graph[node_id] = list of nodes that depend on this node
        dependency_graph: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes.keys()}
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self.nodes.keys()}
        
        # Count input connections for each node to calculate in-degree
        for connection in self.connections.values():
            from_node = connection.from_node_id
            to_node = connection.to_node_id
            
            # Add dependency: from_node must be evaluated before to_node
            dependency_graph[from_node].append(to_node)
            in_degree[to_node] += 1
        
        # Initialize queue with nodes that have no dependencies (in-degree = 0)
        # Input nodes and unconnected nodes start here
        queue = deque()
        for node_id, degree in in_degree.items():
            if degree == 0:
                queue.append(node_id)
        
        evaluation_order = []
        
        # Process nodes in topological order
        while queue:
            current_node = queue.popleft()
            evaluation_order.append(current_node)
            
            # Remove this node from the graph and update in-degrees
            for dependent_node in dependency_graph[current_node]:
                in_degree[dependent_node] -= 1
                if in_degree[dependent_node] == 0:
                    queue.append(dependent_node)
        
        # Check for circular dependencies
        if len(evaluation_order) != len(self.nodes):
            remaining_nodes = set(self.nodes.keys()) - set(evaluation_order)
            raise ValueError(f"Circular dependency detected involving nodes: {remaining_nodes}")
        
        self.logger.debug(f"Node evaluation order: {evaluation_order}")
        return evaluation_order
    
    def _tarjan_scc(self) -> List[Set[str]]:
        """
        Find strongly connected components using Tarjan's algorithm.
        
        Returns:
            List[Set[str]]: List of strongly connected components (sets of node IDs)
        """
        # Tarjan's algorithm state
        index_counter = [0]  # Use list to allow modification in nested function
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []
        
        def strongconnect(node_id: str):
            # Set the depth index for this node to the smallest unused index
            index[node_id] = index_counter[0]
            lowlinks[node_id] = index_counter[0]
            index_counter[0] += 1
            stack.append(node_id)
            on_stack[node_id] = True
            
            # Consider successors of this node
            for connection in self.connections.values():
                if connection.from_node_id == node_id:
                    successor = connection.to_node_id
                    
                    if successor not in index:
                        # Successor has not been visited; recurse on it
                        strongconnect(successor)
                        lowlinks[node_id] = min(lowlinks[node_id], lowlinks[successor])
                    elif on_stack[successor]:
                        # Successor is in stack and hence in the current SCC
                        lowlinks[node_id] = min(lowlinks[node_id], index[successor])
            
            # If this node is a root node, pop the stack and print an SCC
            if lowlinks[node_id] == index[node_id]:
                scc = set()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.add(w)
                    if w == node_id:
                        break
                sccs.append(scc)
        
        # Initialize all nodes as unvisited
        for node_id in self.nodes:
            if node_id not in index:
                strongconnect(node_id)
        
        return sccs
    
    def _detect_feedback_loops(self) -> Tuple[List[Set[str]], bool]:
        """
        Detect feedback loops in the circuit using strongly connected components.
        
        Returns:
            Tuple[List[Set[str]], bool]: (list of SCCs, has_feedback_loops flag)
                - SCCs with more than one node represent feedback loops
                - Single-node SCCs with self-loops also represent feedback loops
        """
        sccs = self._tarjan_scc()
        has_feedback = False
        feedback_components = []
        
        for scc in sccs:
            if len(scc) > 1:
                # Multi-node SCC is definitely a feedback loop
                has_feedback = True
                feedback_components.append(scc)
                self.logger.debug(f"Feedback loop detected: {scc}")
            elif len(scc) == 1:
                # Check for self-loop
                node_id = next(iter(scc))
                for connection in self.connections.values():
                    if connection.from_node_id == node_id and connection.to_node_id == node_id:
                        has_feedback = True
                        feedback_components.append(scc)
                        self.logger.debug(f"Self-loop detected: {node_id}")
                        break
        
        return feedback_components, has_feedback
    
    def simulate(self) -> bool:
        """
        Simulate the entire circuit, handling both combinational and sequential circuits.
        For circuits with feedback loops, uses iterative simulation until convergence.
        
        Returns:
            bool: True if simulation succeeded, False if there were errors
            
        Raises:
            ValueError: If input nodes have no values set or simulation fails to converge
        """
        self.logger.info("Starting circuit simulation")
        
        # Validate that all input nodes have values set
        input_nodes = [node for node in self.nodes.values() if node.node_type == "input"]
        unset_inputs = [node.id for node in input_nodes if node.value is None]
        
        if unset_inputs:
            raise ValueError(f"Input nodes must have values set before simulation: {unset_inputs}")
        
        # Detect feedback loops
        feedback_components, has_feedback = self._detect_feedback_loops()
        
        if has_feedback:
            self.logger.info(f"Feedback loops detected, using iterative simulation")
            return self._simulate_iterative(feedback_components)
        else:
            self.logger.info("No feedback loops detected, using single-pass simulation")
            return self._simulate_combinational()
    
    def _simulate_combinational(self) -> bool:
        """
        Simulate combinational circuit using topological sorting (original method).
        
        Returns:
            bool: True if simulation succeeded
        """
        # Get evaluation order using topological sorting
        try:
            evaluation_order = self._get_evaluation_order()
        except ValueError as e:
            self.logger.error(f"Failed to determine evaluation order: {e}")
            raise
        
        # Clear existing values for non-input nodes
        for node in self.nodes.values():
            if node.node_type != "input":
                node.value = None
        
        # Evaluate nodes in dependency order
        evaluated_count = 0
        for node_id in evaluation_order:
            node = self.nodes[node_id]
            
            # Skip input nodes - they already have values
            if node.node_type == "input":
                self.logger.debug(f"Skipping input node {node_id}: value={node.value}")
                continue
            
            # Collect input values for this node
            input_values = []
            for connection in self.connections.values():
                if connection.to_node_id == node_id:
                    source_node = self.nodes[connection.from_node_id]
                    
                    if source_node.value is None:
                        # This shouldn't happen with proper topological sorting
                        raise ValueError(f"Source node {source_node.id} has no value when evaluating {node_id}")
                    
                    input_values.append(source_node.value)
                    self.logger.debug(f"Input to {node_id} from {source_node.id}: {source_node.value}")
            
            # Evaluate the node using the circuit evaluator
            try:
                result = self.evaluator.evaluate_node(node, input_values)
                node.value = result
                evaluated_count += 1
                self.logger.debug(f"Evaluated {node_id} ({node.node_type}): inputs={input_values} -> {result}")
            except ValueError as e:
                self.logger.error(f"Failed to evaluate node {node_id}: {e}")
                raise ValueError(f"Evaluation failed for node {node_id}: {e}")
        
        self.logger.info(f"Combinational simulation completed: evaluated {evaluated_count} nodes")
        return True
    
    def _simulate_iterative(self, feedback_components: List[Set[str]]) -> bool:
        """
        Simulate sequential circuit using iterative evaluation until convergence.
        
        Args:
            feedback_components: List of strongly connected components (feedback loops)
            
        Returns:
            bool: True if simulation converged, raises ValueError if not
        """
        # Initialize feedback nodes if they don't have values
        self._initialize_feedback_nodes(feedback_components)
        
        # Store previous states for convergence detection
        previous_states = {}
        stable_iterations = 0
        
        for iteration in range(self.max_simulation_iterations):
            self.logger.debug(f"Simulation iteration {iteration + 1}")
            
            # Store current state
            current_state = {}
            for node_id, node in self.nodes.items():
                if node.node_type != "input":
                    current_state[node_id] = node.value
            
            # Evaluate all non-input nodes
            for node_id, node in self.nodes.items():
                if node.node_type == "input":
                    continue
                
                # Collect input values for this node
                input_values = []
                for connection in self.connections.values():
                    if connection.to_node_id == node_id:
                        source_node = self.nodes[connection.from_node_id]
                        if source_node.value is not None:
                            input_values.append(source_node.value)
                
                # Only evaluate if we have all required inputs
                if self._node_has_all_inputs(node_id, len(input_values)):
                    try:
                        new_value = self.evaluator.evaluate_node(node, input_values)
                        if node.value != new_value:
                            node.value = new_value
                            self.logger.debug(f"Iter {iteration + 1}: {node_id} -> {new_value}")
                    except ValueError as e:
                        self.logger.error(f"Failed to evaluate node {node_id}: {e}")
                        raise ValueError(f"Evaluation failed for node {node_id}: {e}")
            
            # Check for convergence by comparing with previous iteration
            if iteration > 0 and current_state == previous_states.get(iteration - 1, {}):
                stable_iterations += 1
                if stable_iterations >= self.convergence_tolerance:
                    self.logger.info(f"Circuit converged after {iteration + 1} iterations")
                    return True
            else:
                stable_iterations = 0
            
            # Store state for next iteration
            previous_states[iteration] = current_state.copy()
        
        # Did not converge within maximum iterations
        self.logger.warning(f"Circuit did not converge after {self.max_simulation_iterations} iterations")
        raise ValueError(f"Circuit simulation did not converge after {self.max_simulation_iterations} iterations")
    
    def _initialize_feedback_nodes(self, feedback_components: List[Set[str]]) -> None:
        """
        Initialize nodes in feedback loops with default values if they are None.
        
        Args:
            feedback_components: List of feedback loop node sets
        """
        for component in feedback_components:
            for node_id in component:
                node = self.nodes[node_id]
                if node.node_type != "input" and node.value is None:
                    # Initialize with False (power-on reset state)
                    node.value = False
                    self.logger.debug(f"Initialized feedback node {node_id} to False")
    
    def _node_has_all_inputs(self, node_id: str, available_inputs: int) -> bool:
        """
        Check if a node has all its required inputs available.
        
        Args:
            node_id: ID of the node to check
            available_inputs: Number of available input values
            
        Returns:
            bool: True if all required inputs are available
        """
        # Count expected inputs based on connections
        expected_inputs = sum(1 for conn in self.connections.values() 
                            if conn.to_node_id == node_id)
        
        return available_inputs >= expected_inputs
    
    def _get_connection_value(self, connection: Connection) -> tuple[bool, bool]:
        """
        Get the logical value carried by a connection.
        
        Args:
            connection: The connection to evaluate
            
        Returns:
            tuple[bool, bool]: (value, has_value) where:
                - value: the logical value (True/False) 
                - has_value: whether the connection carries a valid value
        """
        # Get the source node
        source_node = self.nodes.get(connection.from_node_id)
        if source_node is None:
            return False, False
        
        # For input and output nodes, use the node's value directly
        if source_node.node_type in ["input", "output"]:
            if source_node.value is not None:
                return source_node.value, True
            else:
                return False, False
        
        # For logic gates (AND, OR, NOT, NOR), use the node's computed output value
        # All current logic gates have a single output, so we can use the node's value
        if source_node.node_type in ["and", "or", "not", "nor"]:
            if source_node.value is not None:
                return source_node.value, True
            else:
                return False, False
        
        # For future multi-output nodes, we would need to:
        # 1. Determine which output connector this connection comes from
        # 2. Get the value for that specific output connector
        # For now, return no value for unknown node types
        return False, False
    
    def reset_simulation(self) -> bool:
        """
        Reset the simulation by clearing all node values.
        
        Returns:
            bool: True if UI should be refreshed (values were cleared)
        """
        self.logger.info("Resetting circuit simulation")
        
        values_cleared = False
        for node in self.nodes.values():
            if node.value is not None:
                node.value = None
                values_cleared = True
        
        if values_cleared:
            self.logger.debug("Cleared all node values")
            return True
        else:
            self.logger.debug("No node values to clear")
            return False
    
    def toggle_input_node(self, node_id: str) -> bool:
        """
        Toggle the value of an input node.
        - In Edit Mode: Cycle None -> True -> False -> None
        - In Simulation Mode: Cycle True -> False -> True (with auto-simulation)
        
        Args:
            node_id: ID of the input node to toggle
            
        Returns:
            bool: True if UI should be refreshed
            
        Raises:
            ValueError: If node doesn't exist or is not an input node
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' does not exist")
        
        node = self.nodes[node_id]
        if node.node_type != "input":
            raise ValueError(f"Node '{node_id}' is not an input node (type: {node.node_type})")
        
        if self.simulation_mode:
            # In simulation mode: toggle between True and False only
            node.value = not node.value
            self.logger.debug(f"Toggled input node '{node_id}' to: {node.value}")
            
            # Auto-simulate the circuit after input change
            try:
                self.simulate()
                self.logger.debug("Auto-simulation completed after input toggle")
            except ValueError as e:
                self.logger.error(f"Auto-simulation failed: {e}")
        else:
            # In edit mode: cycle through None -> True -> False -> None
            if node.value is None:
                node.value = True
            elif node.value is True:
                node.value = False
            else:  # node.value is False
                node.value = None
            
            self.logger.debug(f"Toggled input node '{node_id}' to: {node.value}")
        
        return True  # UI should refresh to show new value
    
    def enter_simulation_mode(self) -> bool:
        """
        Enter simulation mode. Initialize all input nodes with False value and disable editing.
        For sequential circuits, initialize feedback nodes to power-on reset state.
        
        Returns:
            bool: True if UI should be refreshed
        """
        if self.simulation_mode:
            self.logger.debug("Already in simulation mode")
            return False
        
        self.logger.info("Entering simulation mode")
        self.simulation_mode = True
        
        # Initialize all input nodes with False value (default)
        for node in self.nodes.values():
            if node.node_type == "input":
                node.value = False
                self.logger.debug(f"Set input node '{node.id}' to default value: False")
        
        # Detect feedback loops and initialize sequential circuit state
        feedback_components, has_feedback = self._detect_feedback_loops()
        if has_feedback:
            self.logger.info("Sequential circuit detected, initializing feedback nodes")
            # Clear all non-input nodes to None, then initialize feedback nodes
            for node in self.nodes.values():
                if node.node_type != "input":
                    node.value = None
            # Initialize feedback nodes to False (power-on reset)
            self._initialize_feedback_nodes(feedback_components)
        else:
            # For combinational circuits, clear all non-input nodes
            for node in self.nodes.values():
                if node.node_type != "input":
                    node.value = None
        
        # Clear any active editing states
        if self.editing_node_id is not None:
            self.editing_node_id = None
            self.editing_text = ""
        
        # Clear toolbox selection
        if self.selected_node_type is not None or self.toolbox_creation_mode:
            self.selected_node_type = None
            self.toolbox_creation_mode = False
        
        return True  # Always refresh to update button states
    
    def enter_edit_mode(self) -> bool:
        """
        Enter edit mode. Re-enable all editing functionality.
        
        Returns:
            bool: True if UI should be refreshed
        """
        if not self.simulation_mode:
            self.logger.debug("Already in edit mode")
            return False
        
        self.logger.info("Entering edit mode")
        self.simulation_mode = False
        
        # Reset all node values to None (no value set)
        ui_needs_refresh = False
        for node in self.nodes.values():
            if node.value is not None:
                node.value = None
                ui_needs_refresh = True
        
        if ui_needs_refresh:
            self.logger.debug("Cleared all node values for edit mode")
        
        return True  # Always refresh to update button states
    
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
    
    def get_connector_at_position(self, x: float, y: float, tolerance: float = 8.0) -> tuple[str, str] | None:
        """Get the connector at the given position. Returns (node_id, connector_id) or None if no connector"""
        # Check all nodes and their connectors
        # Iterate in reverse order so later-added nodes take priority
        for node_id, node in reversed(list(self.nodes.items())):
            for connector in node.connectors:
                connector_x, connector_y = connector.get_absolute_position(node.x, node.y)
                # Add 3 pixels offset to match the rendering offset used in get_connector_absolute_position
                connector_x += 3
                connector_y += 3
                
                # Check if point is within tolerance distance of connector center
                distance = math.sqrt((x - connector_x) ** 2 + (y - connector_y) ** 2)
                if distance <= tolerance:
                    return (node_id, connector.id)
        
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
        if self.simulation_mode:
            self.logger.debug("Deletion is blocked in simulation mode")
            return False
        
        if self.selected_node_id is not None:
            return self.delete_node(self.selected_node_id)
        elif self.selected_connection_id is not None:
            return self.delete_connection(self.selected_connection_id)
        else:
            self.logger.debug("Delete requested but nothing is selected")
            return False
    
    def start_connection_creation(self, node_id: str, connector_id: str) -> bool:
        """Start creating a connection from the specified connector. Returns True if UI should be refreshed."""
        if node_id not in self.nodes:
            self.logger.warning(f"Cannot start connection from non-existent node: {node_id}")
            return False
        
        node = self.nodes[node_id]
        connector = None
        for conn in node.connectors:
            if conn.id == connector_id:
                connector = conn
                break
        
        if connector is None:
            self.logger.warning(f"Cannot start connection from non-existent connector: {node_id}:{connector_id}")
            return False
        
        # Cancel any existing label editing
        if self.editing_node_id is not None:
            self.cancel_label_edit()
        
        # Clear any existing selections
        if self.selected_node_id is not None:
            self.deselect_node()
        if self.selected_connection_id is not None:
            self.deselect_connection()
        
        # Start connection creation
        self.creating_connection = True
        self.connection_start_node_id = node_id
        self.connection_start_connector_id = connector_id
        
        # Initialize pending connection end point at the connector position
        connector_x, connector_y = self.get_connector_absolute_position(node_id, connector_id)
        self.pending_connection_end_x = connector_x
        self.pending_connection_end_y = connector_y
        
        self.logger.debug(f"Started connection creation from {node_id}:{connector_id}")
        return True
    
    def update_pending_connection(self, x: float, y: float) -> bool:
        """Update the end point of the pending connection. Returns True if UI should be refreshed."""
        if not self.creating_connection:
            return False
        
        self.pending_connection_end_x = x
        self.pending_connection_end_y = y
        return True
    
    def complete_connection_creation(self, end_node_id: str, end_connector_id: str) -> bool:
        """Complete connection creation by connecting to the specified connector. Returns True if UI should be refreshed."""
        if not self.creating_connection:
            self.logger.warning("Cannot complete connection creation - not in creation mode")
            return False
        
        if self.connection_start_node_id is None or self.connection_start_connector_id is None:
            self.logger.warning("Cannot complete connection creation - missing start connector")
            self.cancel_connection_creation()
            return True
        
        # Validate the connection can be created
        if not self.can_create_connection(self.connection_start_node_id, self.connection_start_connector_id, 
                                        end_node_id, end_connector_id):
            self.logger.debug(f"Cannot create connection from {self.connection_start_node_id}:{self.connection_start_connector_id} to {end_node_id}:{end_connector_id}")
            self.cancel_connection_creation()
            return True
        
        # Generate unique connection ID
        connection_id = f"conn_{len(self.connections) + 1}"
        while connection_id in self.connections:
            connection_id = f"conn_{len(self.connections) + hash(connection_id) % 1000}"
        
        # Create the connection
        connection = Connection(
            id=connection_id,
            from_node_id=self.connection_start_node_id,
            from_connector_id=self.connection_start_connector_id,
            to_node_id=end_node_id,
            to_connector_id=end_connector_id
        )
        
        self.add_connection(connection)
        
        # Clear connection creation state
        self.cancel_connection_creation()
        
        self.logger.debug(f"Created connection: {connection.id}")
        return True
    
    def cancel_connection_creation(self) -> bool:
        """Cancel connection creation and return to idle state. Returns True if UI should be refreshed."""
        if not self.creating_connection:
            return False
        
        self.creating_connection = False
        self.connection_start_node_id = None
        self.connection_start_connector_id = None
        self.pending_connection_end_x = 0.0
        self.pending_connection_end_y = 0.0
        
        self.logger.debug("Cancelled connection creation")
        return True
    
    def can_create_connection(self, from_node_id: str, from_connector_id: str, to_node_id: str, to_connector_id: str) -> bool:
        """Check if a connection can be created between the specified connectors"""
        # Check that both nodes exist
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return False
        
        # Can't connect a node to itself
        if from_node_id == to_node_id:
            return False
        
        # Get the connectors
        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]
        
        from_connector = None
        for conn in from_node.connectors:
            if conn.id == from_connector_id:
                from_connector = conn
                break
        
        to_connector = None
        for conn in to_node.connectors:
            if conn.id == to_connector_id:
                to_connector = conn
                break
        
        if from_connector is None or to_connector is None:
            return False
        
        # Connection must be from output to input or input to output
        if from_connector.is_input == to_connector.is_input:
            return False
        
        # Check for duplicate connections
        for connection in self.connections.values():
            if (connection.from_node_id == from_node_id and connection.from_connector_id == from_connector_id and
                connection.to_node_id == to_node_id and connection.to_connector_id == to_connector_id):
                return False
            # Also check reverse direction
            if (connection.from_node_id == to_node_id and connection.from_connector_id == to_connector_id and
                connection.to_node_id == from_node_id and connection.to_connector_id == from_connector_id):
                return False
        
        # Input connectors can only have one incoming connection
        # Check if we're connecting TO an input connector that already has a connection
        if to_connector.is_input:
            for connection in self.connections.values():
                if connection.to_node_id == to_node_id and connection.to_connector_id == to_connector_id:
                    return False
        
        # If from_connector is input, check if it already has incoming connection
        if from_connector.is_input:
            for connection in self.connections.values():
                if connection.to_node_id == from_node_id and connection.to_connector_id == from_connector_id:
                    return False
        
        return True
    
    def select_toolbox_node_type(self, node_type: str) -> bool:
        """Select a node type in the toolbox for creation. Returns True if UI should be refreshed."""
        if self.simulation_mode:
            self.logger.debug("Toolbox selection is blocked in simulation mode")
            return False
        
        if node_type not in NODE_REGISTRY.list_definitions():
            self.logger.warning(f"Cannot select unknown node type: {node_type}")
            return False
        
        # Clear any existing states when entering toolbox creation mode
        if self.creating_connection:
            self.cancel_connection_creation()
        
        if self.selected_node_id is not None:
            self.deselect_node()
        
        if self.selected_connection_id is not None:
            self.deselect_connection()
        
        if self.editing_node_id is not None:
            self.cancel_label_edit()
        
        self.selected_node_type = node_type
        self.toolbox_creation_mode = True
        
        self.logger.debug(f"Selected toolbox node type: {node_type}")
        return True
    
    def deselect_toolbox_node_type(self) -> bool:
        """Deselect the current toolbox node type. Returns True if UI should be refreshed."""
        if self.selected_node_type is None:
            return False
        
        previous_selection = self.selected_node_type
        self.selected_node_type = None
        self.toolbox_creation_mode = False
        
        self.logger.debug(f"Deselected toolbox node type: {previous_selection}")
        return True
    
    def create_node_at_position(self, node_type: str, x: float, y: float) -> bool:
        """Create a new node of the specified type at the given position. Returns True if UI should be refreshed."""
        if self.simulation_mode:
            self.logger.debug("Node creation is blocked in simulation mode")
            return False
        
        if node_type not in NODE_REGISTRY.list_definitions():
            self.logger.warning(f"Cannot create node of unknown type: {node_type}")
            return False
        
        # Generate unique node ID
        node_id = f"{node_type}_{len([n for n in self.nodes.values() if n.node_type == node_type]) + 1}"
        while node_id in self.nodes:
            node_id = f"{node_type}_{hash(node_id) % 10000}"
        
        # Create the node using the registry
        node_definition = NODE_REGISTRY.get_definition(node_type)
        new_node = Node.create(node_id, node_definition, x, y)
        
        self.add_node(new_node)
        
        # After creating a node, return to idle state
        self.deselect_toolbox_node_type()
        
        self.logger.debug(f"Created new {node_type} node: {node_id} at ({x}, {y})")
        return True
    
    def get_toolbox_data(self) -> List[Dict[str, Any]]:
        """Get toolbox data for UI display"""
        toolbox_items = []
        
        for node_type in NODE_REGISTRY.list_definitions():
            definition = NODE_REGISTRY.get_definition(node_type)
            toolbox_items.append({
                "node_type": node_type,
                "label": definition.label,
                "color": definition.color,
                "is_selected": node_type == self.selected_node_type
            })
        
        return toolbox_items
    
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
        
        # Block most editing operations in simulation mode
        if self.simulation_mode:
            # Only allow node selection (for input nodes) in simulation mode
            clicked_node_id = self.get_node_at_position(x, y)
            if clicked_node_id is not None:
                # Select the node but don't allow dragging or other operations
                if clicked_node_id != self.selected_node_id:
                    self.select_node(clicked_node_id)
                    return True
            else:
                # Clear selection when clicking empty area
                if self.selected_node_id is not None:
                    self.deselect_node()
                    return True
            return False
        
        # If we're creating a connection, handle it first
        if self.creating_connection:
            # Check if we clicked on a connector to complete the connection
            clicked_connector = self.get_connector_at_position(x, y)
            if clicked_connector is not None:
                node_id, connector_id = clicked_connector
                return self.complete_connection_creation(node_id, connector_id)
            else:
                # Clicked on empty area - cancel connection creation
                return self.cancel_connection_creation()
        
        # Check if we clicked on a connector to start connection creation
        clicked_connector = self.get_connector_at_position(x, y)
        if clicked_connector is not None:
            node_id, connector_id = clicked_connector
            return self.start_connection_creation(node_id, connector_id)
        
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
        # Block dragging and connection operations in simulation mode
        if self.simulation_mode:
            return False
        
        # If we're creating a connection, update the pending connection
        if self.creating_connection:
            return self.update_pending_connection(x, y)
        
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
        
        # In simulation mode, just reset pointer state and return
        if self.simulation_mode:
            self.pointer_state = PointerState.IDLE
            self.drag_node_id = None
            self.drag_offset = (0.0, 0.0)
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
        """Handle double-click for input node toggling or label editing. Returns True if UI should be refreshed."""
        clicked_node_id = self.get_node_at_position(x, y)
        
        if clicked_node_id is not None:
            clicked_node = self.nodes[clicked_node_id]
            
            # For input nodes, toggle their value
            if clicked_node.node_type == "input":
                return self.toggle_input_node(clicked_node_id)
            else:
                # For other nodes, start editing the label only in edit mode
                if not self.simulation_mode:
                    return self.start_label_edit(clicked_node_id)
                else:
                    # In simulation mode, non-input nodes can't be edited
                    self.logger.debug(f"Ignoring double-click on {clicked_node.node_type} node in simulation mode")
                    return False
        
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
                "connectors": connectors_data,
                "value": node.value,
                "has_value": node.value is not None
            }
            slint_nodes.append(slint_node)
        
        # Convert connections with calculated positions and values
        slint_connections = []
        for conn in self.connections.values():
            start_x, start_y = self.get_connector_absolute_position(conn.from_node_id, conn.from_connector_id)
            end_x, end_y = self.get_connector_absolute_position(conn.to_node_id, conn.to_connector_id)
            
            # Get connection value for simulation mode visualization
            connection_value, has_connection_value = self._get_connection_value(conn)
            
            slint_connection = {
                "id": conn.id,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "value": connection_value,
                "has_value": has_connection_value,
                "simulation_mode": self.simulation_mode
            }
            slint_connections.append(slint_connection)
        
        # Get pending connection start position if creating connection
        pending_start_x = 0.0
        pending_start_y = 0.0
        if self.creating_connection and self.connection_start_node_id and self.connection_start_connector_id:
            pending_start_x, pending_start_y = self.get_connector_absolute_position(
                self.connection_start_node_id, self.connection_start_connector_id)
        
        return {
            "nodes": slint_nodes,
            "connections": slint_connections,
            "selected_nodes": [self.selected_node_id] if self.selected_node_id else [],
            "selected_connections": [self.selected_connection_id] if self.selected_connection_id else [],
            "editing_node_id": self.editing_node_id or "",
            "editing_text": self.editing_text,
            "creating_connection": self.creating_connection,
            "pending_start_x": pending_start_x,
            "pending_start_y": pending_start_y,
            "pending_end_x": self.pending_connection_end_x,
            "pending_end_y": self.pending_connection_end_y,
            "toolbox_items": self.get_toolbox_data(),
            "toolbox_creation_mode": self.toolbox_creation_mode,
            "simulation_mode": self.simulation_mode
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


def create_sr_nor_latch_demo() -> GraphData:
    """Create a demonstration SR NOR latch circuit with feedback"""
    graph = GraphData()
    
    # Create input nodes (S and R)
    input_s = Node.create("set", NODE_REGISTRY.get_definition("input"), 50, 100)
    input_r = Node.create("reset", NODE_REGISTRY.get_definition("input"), 50, 200)
    
    # Create NOR gates
    nor_gate_1 = Node.create("nor1", NODE_REGISTRY.get_definition("nor"), 200, 100)
    nor_gate_2 = Node.create("nor2", NODE_REGISTRY.get_definition("nor"), 200, 200)
    
    # Create output nodes (Q and Q̅)
    output_q = Node.create("q", NODE_REGISTRY.get_definition("output"), 350, 100)
    output_q_not = Node.create("q_not", NODE_REGISTRY.get_definition("output"), 350, 200)
    
    # Add nodes to graph
    graph.add_node(input_s)
    graph.add_node(input_r)
    graph.add_node(nor_gate_1)
    graph.add_node(nor_gate_2)
    graph.add_node(output_q)
    graph.add_node(output_q_not)
    
    # Add connections for SR NOR latch
    # The feedback connections create the latch behavior:
    # - R input goes to NOR1 input 1 (NOR1 produces Q)
    # - S input goes to NOR2 input 1 (NOR2 produces Q̅)
    # - NOR1 output (Q) feeds back to NOR2 input 2
    # - NOR2 output (Q̅) feeds back to NOR1 input 2
    connections = [
        Connection("c1", "reset", "out", "nor1", "in1"),    # R -> NOR1 (Q gate)
        Connection("c2", "set", "out", "nor2", "in1"),      # S -> NOR2 (Q̅ gate)
        Connection("c3", "nor1", "out", "nor2", "in2"),     # Q -> NOR2 (feedback)
        Connection("c4", "nor2", "out", "nor1", "in2"),     # Q̅ -> NOR1 (feedback)
        Connection("c5", "nor1", "out", "q", "in"),         # Q output
        Connection("c6", "nor2", "out", "q_not", "in")      # Q̅ output
    ]
    
    for conn in connections:
        graph.add_connection(conn)
    
    return graph