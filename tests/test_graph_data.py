"""
Unit tests for graph_data.py module
"""

import pytest
from unittest.mock import patch, MagicMock
import logging

from logicsim.graph_data import (
    ConnectorDefinition,
    NodeDefinition,
    NodeDefinitionRegistry,
    NODE_REGISTRY,
    Connector,
    Node,
    Connection,
    GraphData,
    PointerState,
    create_demo_graph
)


class TestConnector:
    """Test cases for the Connector class"""
    
    def test_connector_creation(self):
        """Test basic connector creation"""
        connector = Connector(
            id="test_connector",
            x_offset=10.0,
            y_offset=20.0,
            is_input=True
        )
        
        assert connector.id == "test_connector"
        assert connector.x_offset == 10.0
        assert connector.y_offset == 20.0
        assert connector.is_input == True
    
    def test_get_absolute_position(self):
        """Test absolute position calculation"""
        connector = Connector(
            id="test",
            x_offset=5.0,
            y_offset=10.0,
            is_input=False
        )
        
        # Test with various node positions
        abs_x, abs_y = connector.get_absolute_position(100.0, 200.0)
        assert abs_x == 105.0
        assert abs_y == 210.0
        
        # Test with zero position
        abs_x, abs_y = connector.get_absolute_position(0.0, 0.0)
        assert abs_x == 5.0
        assert abs_y == 10.0
        
        # Test with negative offsets
        connector_neg = Connector("test", -5.0, -10.0, True)
        abs_x, abs_y = connector_neg.get_absolute_position(50.0, 60.0)
        assert abs_x == 45.0
        assert abs_y == 50.0


class TestConnectorDefinition:
    """Test cases for the ConnectorDefinition class"""
    
    def test_connector_definition_creation(self):
        """Test basic connector definition creation"""
        conn_def = ConnectorDefinition(
            id="test_conn",
            x_offset_ratio=0.5,
            y_offset_ratio=0.25,
            is_input=True
        )
        
        assert conn_def.id == "test_conn"
        assert conn_def.x_offset_ratio == 0.5
        assert conn_def.y_offset_ratio == 0.25
        assert conn_def.is_input == True
    
    def test_create_connector(self):
        """Test creating a connector from definition"""
        conn_def = ConnectorDefinition("out", 1.0, 0.5, False)
        
        # Test with 80x60 dimensions
        connector = conn_def.create_connector(80.0, 60.0)
        
        assert connector.id == "out"
        assert connector.x_offset == 76.0  # 1.0 * 80 - 4
        assert connector.y_offset == 26.0  # 0.5 * 60 - 4
        assert connector.is_input == False
    
    def test_create_connector_different_dimensions(self):
        """Test connector creation with different dimensions"""
        conn_def = ConnectorDefinition("in", 0.0, 0.5, True)
        
        # Test with 50x50 dimensions
        connector = conn_def.create_connector(50.0, 50.0)
        
        assert connector.id == "in"
        assert connector.x_offset == -4.0  # 0.0 * 50 - 4
        assert connector.y_offset == 21.0  # 0.5 * 50 - 4
        assert connector.is_input == True


class TestNodeDefinition:
    """Test cases for the NodeDefinition class"""
    
    def test_node_definition_creation(self):
        """Test basic node definition creation"""
        connectors = [
            ConnectorDefinition("in", 0.0, 0.5, True),
            ConnectorDefinition("out", 1.0, 0.5, False)
        ]
        
        node_def = NodeDefinition(
            name="test_node",
            label="TEST",
            default_width=100.0,
            default_height=80.0,
            color="#FF0000",
            connectors=connectors
        )
        
        assert node_def.name == "test_node"
        assert node_def.label == "TEST"
        assert node_def.default_width == 100.0
        assert node_def.default_height == 80.0
        assert node_def.color == "#FF0000"
        assert len(node_def.connectors) == 2
        assert node_def.connectors[0].id == "in"
        assert node_def.connectors[1].id == "out"


class TestNodeDefinitionRegistry:
    """Test cases for the NodeDefinitionRegistry class"""
    
    def test_registry_initialization(self):
        """Test that registry initializes with standard definitions"""
        registry = NodeDefinitionRegistry()
        
        expected_types = ["input", "output", "and", "or", "not"]
        for node_type in expected_types:
            assert node_type in registry.definitions
            assert isinstance(registry.definitions[node_type], NodeDefinition)
    
    def test_get_definition(self):
        """Test getting node definitions by name"""
        registry = NodeDefinitionRegistry()
        
        input_def = registry.get_definition("input")
        assert input_def.name == "input"
        assert input_def.label == "INPUT"
        assert len(input_def.connectors) == 1
        assert input_def.connectors[0].id == "out"
        assert input_def.connectors[0].is_input == False
    
    def test_get_definition_unknown_type(self):
        """Test error handling for unknown node types"""
        registry = NodeDefinitionRegistry()
        
        with pytest.raises(ValueError, match="Unknown node type: unknown"):
            registry.get_definition("unknown")
    
    def test_list_definitions(self):
        """Test listing all available definitions"""
        registry = NodeDefinitionRegistry()
        
        definitions = registry.list_definitions()
        expected_types = ["input", "output", "and", "or", "not"]
        
        for node_type in expected_types:
            assert node_type in definitions
    
    def test_add_definition(self):
        """Test adding custom node definitions"""
        registry = NodeDefinitionRegistry()
        
        custom_def = NodeDefinition(
            name="custom",
            label="CUSTOM",
            default_width=60.0,
            default_height=40.0,
            color="#0000FF",
            connectors=[ConnectorDefinition("test", 0.5, 0.5, True)]
        )
        
        registry.add_definition(custom_def)
        
        assert "custom" in registry.definitions
        retrieved_def = registry.get_definition("custom")
        assert retrieved_def.name == "custom"
        assert retrieved_def.label == "CUSTOM"
        assert retrieved_def.color == "#0000FF"
    
    @pytest.mark.parametrize("node_type,expected_connector_count", [
        ("input", 1),
        ("output", 1),
        ("and", 3),
        ("or", 3),
        ("not", 2)
    ])
    def test_standard_node_connector_counts(self, node_type, expected_connector_count):
        """Test that standard node types have correct connector counts"""
        registry = NodeDefinitionRegistry()
        definition = registry.get_definition(node_type)
        
        assert len(definition.connectors) == expected_connector_count
    
    @pytest.mark.parametrize("node_type,expected_color", [
        ("input", "rgb(144, 238, 144)"),
        ("output", "rgb(255, 182, 193)"),
        ("and", "rgb(224, 224, 224)"),
        ("or", "rgb(224, 224, 224)"),
        ("not", "rgb(224, 224, 224)")
    ])
    def test_standard_node_colors(self, node_type, expected_color):
        """Test that standard node types have correct colors"""
        registry = NodeDefinitionRegistry()
        definition = registry.get_definition(node_type)
        
        assert definition.color == expected_color


class TestNode:
    """Test cases for the Node class"""
    
    def test_node_creation_manual(self):
        """Test manual node creation with explicit connectors"""
        connectors = [
            Connector("in1", -4.0, 15.0, True),
            Connector("out", 76.0, 26.0, False)
        ]
        
        node = Node(
            id="test_node",
            node_type="and",
            x=100.0,
            y=200.0,
            width=80.0,
            height=60.0,
            label="TEST",
            color="#E0E0E0",
            connectors=connectors
        )
        
        assert node.id == "test_node"
        assert node.node_type == "and"
        assert node.x == 100.0
        assert node.y == 200.0
        assert node.width == 80.0
        assert node.height == 60.0
        assert node.label == "TEST"
        assert node.color == "#E0E0E0"
        assert len(node.connectors) == 2
        assert node.connectors[0].id == "in1"
        assert node.connectors[1].id == "out"
    
    @pytest.mark.parametrize("node_type,expected_label,expected_color", [
        ("input", "INPUT", "rgb(144, 238, 144)"),
        ("output", "OUTPUT", "rgb(255, 182, 193)"),
        ("and", "AND", "rgb(224, 224, 224)"),
        ("or", "OR", "rgb(224, 224, 224)"),
        ("not", "NOT", "rgb(224, 224, 224)")
    ])
    def test_node_create_with_defaults(self, node_type, expected_label, expected_color):
        """Test Node.create() with default values from definition"""
        definition = NODE_REGISTRY.get_definition(node_type)
        node = Node.create(f"{node_type}_1", definition, 100.0, 200.0)
        
        assert node.id == f"{node_type}_1"
        assert node.node_type == node_type
        assert node.x == 100.0
        assert node.y == 200.0
        assert node.width == definition.default_width
        assert node.height == definition.default_height
        assert node.label == expected_label
        assert node.color == expected_color
        assert len(node.connectors) == len(definition.connectors)
    
    def test_node_create_with_custom_values(self):
        """Test Node.create() with custom values overriding defaults"""
        definition = NODE_REGISTRY.get_definition("input")
        node = Node.create("custom_input", definition, 50.0, 50.0, 100.0, 80.0, "CUSTOM")
        
        assert node.id == "custom_input"
        assert node.node_type == "input"
        assert node.x == 50.0
        assert node.y == 50.0
        assert node.width == 100.0  # Custom width
        assert node.height == 80.0  # Custom height
        assert node.label == "CUSTOM"  # Custom label
        assert node.color == "rgb(144, 238, 144)"  # From definition
    
    def test_node_create_connector_positioning(self):
        """Test that connectors are positioned correctly based on node dimensions"""
        definition = NODE_REGISTRY.get_definition("and")
        node = Node.create("and_test", definition, 0.0, 0.0, 100.0, 80.0)
        
        # Check that connectors are created with correct positions
        in1 = next(c for c in node.connectors if c.id == "in1")
        in2 = next(c for c in node.connectors if c.id == "in2")
        out = next(c for c in node.connectors if c.id == "out")
        
        # in1: 0.0 * 100 - 4 = -4, 0.25 * 80 - 4 = 16
        assert in1.x_offset == -4.0
        assert in1.y_offset == 16.0
        assert in1.is_input == True
        
        # in2: 0.0 * 100 - 4 = -4, 0.75 * 80 - 4 = 56
        assert in2.x_offset == -4.0
        assert in2.y_offset == 56.0
        assert in2.is_input == True
        
        # out: 1.0 * 100 - 4 = 96, 0.5 * 80 - 4 = 36
        assert out.x_offset == 96.0
        assert out.y_offset == 36.0
        assert out.is_input == False


class TestConnection:
    """Test cases for the Connection class"""
    
    def test_connection_creation(self):
        """Test basic connection creation"""
        connection = Connection(
            id="conn1",
            from_node_id="node1",
            from_connector_id="out",
            to_node_id="node2",
            to_connector_id="in"
        )
        
        assert connection.id == "conn1"
        assert connection.from_node_id == "node1"
        assert connection.from_connector_id == "out"
        assert connection.to_node_id == "node2"
        assert connection.to_connector_id == "in"




class TestGraphData:
    """Test cases for the GraphData class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
        
        # Create test nodes using new system
        self.input_node = Node.create("input1", NODE_REGISTRY.get_definition("input"), 50.0, 50.0, label="A")
        self.and_gate = Node.create("and1", NODE_REGISTRY.get_definition("and"), 200.0, 80.0, label="AND")
        self.output_node = Node.create("output1", NODE_REGISTRY.get_definition("output"), 350.0, 90.0, label="OUT")
        
        # Create test connection
        self.connection = Connection("conn1", "input1", "out", "and1", "in1")
    
    @patch('logicsim.graph_data.logging.getLogger')
    def test_graph_data_initialization(self, mock_logger):
        """Test GraphData initialization"""
        graph = GraphData()
        
        assert isinstance(graph.nodes, dict)
        assert isinstance(graph.connections, dict)
        assert len(graph.nodes) == 0
        assert len(graph.connections) == 0
        mock_logger.assert_called_once()
    
    def test_add_node(self):
        """Test adding nodes to the graph"""
        with patch.object(self.graph, 'logger') as mock_logger:
            self.graph.add_node(self.input_node)
            
            assert "input1" in self.graph.nodes
            assert self.graph.nodes["input1"] == self.input_node
            mock_logger.debug.assert_called_once_with("Added node: input1 (input)")
    
    def test_add_multiple_nodes(self):
        """Test adding multiple nodes to the graph"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
        self.graph.add_node(self.output_node)
        
        assert len(self.graph.nodes) == 3
        assert "input1" in self.graph.nodes
        assert "and1" in self.graph.nodes
        assert "output1" in self.graph.nodes
    
    def test_add_connection(self):
        """Test adding connections to the graph"""
        with patch.object(self.graph, 'logger') as mock_logger:
            self.graph.add_connection(self.connection)
            
            assert "conn1" in self.graph.connections
            assert self.graph.connections["conn1"] == self.connection
            mock_logger.debug.assert_called_once_with("Added connection: input1:out -> and1:in1")
    
    def test_get_connector_absolute_position(self):
        """Test getting absolute position of connectors"""
        self.graph.add_node(self.input_node)
        
        # Get position of output connector on input node
        abs_x, abs_y = self.graph.get_connector_absolute_position("input1", "out")
        
        # Input node at (50, 50) with output connector at offset (46, 21)
        # Plus the +3 adjustment in the method
        expected_x = 50 + 46 + 3  # 99
        expected_y = 50 + 21 + 3  # 74
        
        assert abs_x == expected_x
        assert abs_y == expected_y
    
    def test_get_connector_absolute_position_error(self):
        """Test error handling for invalid node/connector IDs"""
        self.graph.add_node(self.input_node)
        
        # Test with invalid node ID
        with pytest.raises(KeyError):
            self.graph.get_connector_absolute_position("invalid_node", "out")
        
        # Test with invalid connector ID
        with pytest.raises(StopIteration):
            self.graph.get_connector_absolute_position("input1", "invalid_connector")
    
    def test_to_slint_format_empty_graph(self):
        """Test Slint format conversion with empty graph"""
        result = self.graph.to_slint_format()
        
        assert "nodes" in result
        assert "connections" in result
        assert "selected_nodes" in result
        assert isinstance(result["nodes"], list)
        assert isinstance(result["connections"], list)
        assert isinstance(result["selected_nodes"], list)
        assert len(result["nodes"]) == 0
        assert len(result["connections"]) == 0
        assert len(result["selected_nodes"]) == 0
    
    def test_to_slint_format_with_nodes(self):
        """Test Slint format conversion with nodes"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
        
        result = self.graph.to_slint_format()
        
        assert len(result["nodes"]) == 2
        assert len(result["connections"]) == 0
        
        # Check node structure
        node_ids = [node["id"] for node in result["nodes"]]
        assert "input1" in node_ids
        assert "and1" in node_ids
        
        # Check input node structure
        input_node_data = next(node for node in result["nodes"] if node["id"] == "input1")
        assert input_node_data["node_type"] == "input"
        assert input_node_data["x"] == 50.0
        assert input_node_data["y"] == 50.0
        assert input_node_data["width"] == 50.0
        assert input_node_data["height"] == 50.0
        assert input_node_data["label"] == "A"
        assert input_node_data["color"] == "rgb(144, 238, 144)"
        assert len(input_node_data["connectors"]) == 1
        
        # Check connector data
        connector_data = input_node_data["connectors"][0]
        assert connector_data["id"] == "out"
        assert connector_data["x"] == 96.0  # 50 + 46
        assert connector_data["y"] == 71.0  # 50 + 21
        assert connector_data["is_input"] == False
    
    def test_to_slint_format_with_connections(self):
        """Test Slint format conversion with connections"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
        self.graph.add_connection(self.connection)
        
        result = self.graph.to_slint_format()
        
        assert len(result["nodes"]) == 2
        assert len(result["connections"]) == 1
        
        # Check connection structure
        conn_data = result["connections"][0]
        assert conn_data["id"] == "conn1"
        assert conn_data["start_x"] == 99.0  # 50 + 46 + 3
        assert conn_data["start_y"] == 74.0  # 50 + 21 + 3
        assert conn_data["end_x"] == 199.0   # 200 + (-4) + 3
        assert conn_data["end_y"] == 94.0    # 80 + 11 + 3 (0.25 * 60 - 4 = 11)
    
    def test_selection_initialization(self):
        """Test that selection is initially None"""
        assert self.graph.selected_node_id is None
        assert self.graph.get_selected_node() is None
    
    def test_select_node(self):
        """Test selecting a node"""
        self.graph.add_node(self.input_node)
        
        with patch.object(self.graph, 'logger') as mock_logger:
            self.graph.select_node("input1")
            
            assert self.graph.selected_node_id == "input1"
            assert self.graph.get_selected_node() == "input1"
            mock_logger.debug.assert_called_once_with("Selected node: input1 (previously: None)")
    
    def test_select_node_invalid_id(self):
        """Test selecting a node with invalid ID"""
        with pytest.raises(ValueError, match="Node with ID 'invalid' does not exist"):
            self.graph.select_node("invalid")
    
    def test_select_different_node(self):
        """Test selecting a different node"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
        
        # Select first node
        self.graph.select_node("input1")
        assert self.graph.selected_node_id == "input1"
        
        # Select second node
        with patch.object(self.graph, 'logger') as mock_logger:
            self.graph.select_node("and1")
            
            assert self.graph.selected_node_id == "and1"
            assert self.graph.get_selected_node() == "and1"
            mock_logger.debug.assert_called_once_with("Selected node: and1 (previously: input1)")
    
    def test_select_same_node_twice(self):
        """Test selecting the same node twice (should not log)"""
        self.graph.add_node(self.input_node)
        self.graph.select_node("input1")
        
        with patch.object(self.graph, 'logger') as mock_logger:
            self.graph.select_node("input1")
            
            assert self.graph.selected_node_id == "input1"
            # Should not log when selecting the same node
            mock_logger.debug.assert_not_called()
    
    def test_deselect_node(self):
        """Test deselecting a node"""
        self.graph.add_node(self.input_node)
        self.graph.select_node("input1")
        
        with patch.object(self.graph, 'logger') as mock_logger:
            self.graph.deselect_node()
            
            assert self.graph.selected_node_id is None
            assert self.graph.get_selected_node() is None
            mock_logger.debug.assert_called_once_with("Deselected node: input1")
    
    def test_deselect_node_when_none_selected(self):
        """Test deselecting when no node is selected"""
        with patch.object(self.graph, 'logger') as mock_logger:
            self.graph.deselect_node()
            
            assert self.graph.selected_node_id is None
            # Should not log when no node was selected
            mock_logger.debug.assert_not_called()
    
    def test_is_point_in_node(self):
        """Test point-in-node hit testing"""
        self.graph.add_node(self.input_node)
        node = self.graph.nodes["input1"]
        
        # Test points inside the node (50, 50, 50x50)
        assert self.graph.is_point_in_node(node, 60.0, 60.0) == True
        assert self.graph.is_point_in_node(node, 50.0, 50.0) == True  # Top-left corner
        assert self.graph.is_point_in_node(node, 100.0, 100.0) == True  # Bottom-right corner
        assert self.graph.is_point_in_node(node, 75.0, 75.0) == True  # Center
        
        # Test points outside the node
        assert self.graph.is_point_in_node(node, 49.0, 60.0) == False  # Left of node
        assert self.graph.is_point_in_node(node, 101.0, 60.0) == False  # Right of node
        assert self.graph.is_point_in_node(node, 60.0, 49.0) == False  # Above node
        assert self.graph.is_point_in_node(node, 60.0, 101.0) == False  # Below node
        assert self.graph.is_point_in_node(node, 0.0, 0.0) == False  # Far away
    
    def test_get_node_at_position(self):
        """Test getting node at position"""
        self.graph.add_node(self.input_node)  # At (50, 50) 50x50
        self.graph.add_node(self.and_gate)    # At (200, 80) 80x60
        
        # Test hitting the input node
        assert self.graph.get_node_at_position(60.0, 60.0) == "input1"
        assert self.graph.get_node_at_position(50.0, 50.0) == "input1"
        assert self.graph.get_node_at_position(100.0, 100.0) == "input1"
        
        # Test hitting the and gate
        assert self.graph.get_node_at_position(220.0, 100.0) == "and1"
        assert self.graph.get_node_at_position(200.0, 80.0) == "and1"
        assert self.graph.get_node_at_position(280.0, 140.0) == "and1"
        
        # Test hitting empty space
        assert self.graph.get_node_at_position(0.0, 0.0) is None
        assert self.graph.get_node_at_position(150.0, 150.0) is None
        assert self.graph.get_node_at_position(500.0, 500.0) is None
    
    def test_get_node_at_position_overlapping(self):
        """Test node priority when nodes overlap (later nodes on top)"""
        # Create two overlapping nodes
        node1 = Node.create("node1", NODE_REGISTRY.get_definition("input"), 50.0, 50.0)
        node2 = Node.create("node2", NODE_REGISTRY.get_definition("input"), 60.0, 60.0)
        
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        
        # Point (70, 70) is in both nodes, but node2 was added later so it should take priority
        assert self.graph.get_node_at_position(70.0, 70.0) == "node2"
    
    def test_handle_mouse_click_select_node(self):
        """Test mouse click selecting a node"""
        self.graph.add_node(self.input_node)
        
        # Click on the node
        result = self.graph.handle_mouse_click(60.0, 60.0)
        
        assert result == True  # Selection changed
        assert self.graph.selected_node_id == "input1"
    
    def test_handle_mouse_click_deselect_same_node(self):
        """Test mouse click deselecting the same node"""
        self.graph.add_node(self.input_node)
        self.graph.select_node("input1")
        
        # Click on the same node again
        result = self.graph.handle_mouse_click(60.0, 60.0)
        
        assert result == True  # Selection changed
        assert self.graph.selected_node_id is None
    
    def test_handle_mouse_click_select_different_node(self):
        """Test mouse click selecting a different node"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
        self.graph.select_node("input1")
        
        # Click on the other node
        result = self.graph.handle_mouse_click(220.0, 100.0)
        
        assert result == True  # Selection changed
        assert self.graph.selected_node_id == "and1"
    
    def test_handle_mouse_click_empty_area_with_selection(self):
        """Test mouse click on empty area when a node is selected"""
        self.graph.add_node(self.input_node)
        self.graph.select_node("input1")
        
        # Click on empty area
        result = self.graph.handle_mouse_click(0.0, 0.0)
        
        assert result == True  # Selection changed
        assert self.graph.selected_node_id is None
    
    def test_handle_mouse_click_empty_area_no_selection(self):
        """Test mouse click on empty area when no node is selected"""
        self.graph.add_node(self.input_node)
        
        # Click on empty area
        result = self.graph.handle_mouse_click(0.0, 0.0)
        
        assert result == False  # Selection did not change
        assert self.graph.selected_node_id is None
    
    def test_to_slint_format_with_selection(self):
        """Test Slint format conversion with node selection"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
        self.graph.select_node("input1")
        
        result = self.graph.to_slint_format()
        
        assert "selected_nodes" in result
        assert result["selected_nodes"] == ["input1"]
    
    def test_to_slint_format_no_selection(self):
        """Test Slint format conversion with no selection"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
        
        result = self.graph.to_slint_format()
        
        assert "selected_nodes" in result
        assert result["selected_nodes"] == []


class TestGraphDataMovement:
    """Test cases for node movement functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
        
        # Create test nodes
        self.node1 = Node.create("node1", NODE_REGISTRY.get_definition("input"), 100, 100)
        self.node2 = Node.create("node2", NODE_REGISTRY.get_definition("and"), 200, 150)
        self.node3 = Node.create("node3", NODE_REGISTRY.get_definition("output"), 300, 200)
        
        # Add nodes to graph
        self.graph.add_node(self.node1)
        self.graph.add_node(self.node2)
        self.graph.add_node(self.node3)
    
    def test_movement_state_initialization(self):
        """Test that movement state is properly initialized"""
        assert self.graph.pointer_state == PointerState.IDLE
        assert self.graph.drag_start_pos == (0.0, 0.0)
        assert self.graph.drag_node_id is None
        assert self.graph.drag_offset == (0.0, 0.0)
        assert self.graph.movement_threshold == 5.0
    
    def test_calculate_movement_distance(self):
        """Test movement distance calculation"""
        # Test basic distance calculation
        distance = self.graph.calculate_movement_distance(0, 0, 3, 4)
        assert distance == 5.0
        
        # Test zero distance
        distance = self.graph.calculate_movement_distance(10, 20, 10, 20)
        assert distance == 0.0
        
        # Test negative coordinates
        distance = self.graph.calculate_movement_distance(-5, -5, 5, 5)
        assert distance == pytest.approx(14.142, rel=1e-3)
    
    def test_move_node_valid(self):
        """Test moving a node to a valid position"""
        original_x, original_y = self.node1.x, self.node1.y
        new_x, new_y = 150.0, 175.0
        
        self.graph.move_node("node1", new_x, new_y)
        
        assert self.node1.x == new_x
        assert self.node1.y == new_y
        assert self.node1.x != original_x
        assert self.node1.y != original_y
    
    def test_move_node_invalid_id(self):
        """Test moving a non-existent node raises error"""
        with pytest.raises(ValueError, match="Node with ID 'nonexistent' does not exist"):
            self.graph.move_node("nonexistent", 100, 100)
    
    def test_handle_pointer_down_on_node(self):
        """Test pointer down on a node"""
        # Test clicking on node1
        ui_changed = self.graph.handle_pointer_down(110, 110)
        
        # Should transition to PRESSED state
        assert self.graph.pointer_state == PointerState.PRESSED
        assert self.graph.drag_start_pos == (110, 110)
        assert self.graph.drag_node_id == "node1"
        assert self.graph.drag_offset == (10, 10)  # offset within node
        
        # Should select the node and return True for UI refresh
        assert ui_changed is True
        assert self.graph.selected_node_id == "node1"
    
    def test_handle_pointer_down_on_selected_node(self):
        """Test pointer down on already selected node"""
        # Select node first
        self.graph.select_node("node1")
        
        # Click on the same node
        ui_changed = self.graph.handle_pointer_down(110, 110)
        
        # Should transition to PRESSED state
        assert self.graph.pointer_state == PointerState.PRESSED
        assert self.graph.drag_node_id == "node1"
        
        # Should not change selection (already selected)
        assert ui_changed is False
        assert self.graph.selected_node_id == "node1"
    
    def test_handle_pointer_down_on_empty_area(self):
        """Test pointer down on empty area"""
        # Select a node first
        self.graph.select_node("node1")
        
        # Click on empty area
        ui_changed = self.graph.handle_pointer_down(50, 50)
        
        # Should transition to PRESSED state
        assert self.graph.pointer_state == PointerState.PRESSED
        assert self.graph.drag_start_pos == (50, 50)
        assert self.graph.drag_node_id is None
        assert self.graph.drag_offset == (0.0, 0.0)
        
        # Should not change selection yet (selection changes on mouse up)
        assert ui_changed is False
        assert self.graph.selected_node_id == "node1"
    
    def test_handle_pointer_down_invalid_state(self):
        """Test pointer down when not in IDLE state"""
        # Set to non-IDLE state
        self.graph.pointer_state = PointerState.PRESSED
        
        # Try to handle pointer down
        ui_changed = self.graph.handle_pointer_down(100, 100)
        
        # Should not change state and return False
        assert ui_changed is False
        assert self.graph.pointer_state == PointerState.PRESSED
    
    def test_handle_pointer_move_idle_state(self):
        """Test pointer move when in IDLE state"""
        ui_changed = self.graph.handle_pointer_move(100, 100)
        
        # Should not change anything
        assert ui_changed is False
        assert self.graph.pointer_state == PointerState.IDLE
    
    def test_handle_pointer_move_small_movement(self):
        """Test pointer move with small movement (below threshold)"""
        # Start with pointer down
        self.graph.handle_pointer_down(110, 110)
        
        # Move just under threshold
        ui_changed = self.graph.handle_pointer_move(113, 113)
        
        # Should stay in PRESSED state (distance = ~4.24 < 5.0)
        assert self.graph.pointer_state == PointerState.PRESSED
        assert ui_changed is False
    
    def test_handle_pointer_move_large_movement_with_node(self):
        """Test pointer move with large movement on a node"""
        # Start with pointer down on node1
        self.graph.handle_pointer_down(110, 110)
        
        # Move beyond threshold
        ui_changed = self.graph.handle_pointer_move(120, 120)
        
        # Should transition to DRAGGING state
        assert self.graph.pointer_state == PointerState.DRAGGING
        assert ui_changed is True
        
        # Node should be moved to new position (accounting for offset)
        assert self.node1.x == 110.0  # 120 - 10 (offset)
        assert self.node1.y == 110.0  # 120 - 10 (offset)
    
    def test_handle_pointer_move_large_movement_no_node(self):
        """Test pointer move with large movement on empty area"""
        # Start with pointer down on empty area
        self.graph.handle_pointer_down(50, 50)
        
        # Move beyond threshold
        ui_changed = self.graph.handle_pointer_move(60, 60)
        
        # Should transition to DRAGGING state but no node to move
        assert self.graph.pointer_state == PointerState.DRAGGING
        assert ui_changed is False
        assert self.graph.drag_node_id is None
    
    def test_handle_pointer_move_continuous_dragging(self):
        """Test continuous pointer move during dragging"""
        # Start dragging
        self.graph.handle_pointer_down(110, 110)
        self.graph.handle_pointer_move(120, 120)  # Start dragging
        
        # Continue dragging
        ui_changed = self.graph.handle_pointer_move(130, 125)
        
        # Should continue dragging
        assert self.graph.pointer_state == PointerState.DRAGGING
        assert ui_changed is True
        
        # Node should be at new position
        assert self.node1.x == 120.0  # 130 - 10 (offset)
        assert self.node1.y == 115.0  # 125 - 10 (offset)
    
    def test_handle_pointer_up_after_click(self):
        """Test pointer up after a click (no significant movement)"""
        # Select node2 first
        self.graph.select_node("node2")
        
        # Click on node1 (just press, no significant movement)
        ui_changed_down = self.graph.handle_pointer_down(110, 110)
        ui_changed_up = self.graph.handle_pointer_up(112, 112)
        
        # Should reset to IDLE state
        assert self.graph.pointer_state == PointerState.IDLE
        assert self.graph.drag_node_id is None
        assert self.graph.drag_start_pos == (0.0, 0.0)
        assert self.graph.drag_offset == (0.0, 0.0)
        
        # Should have selected node1 (was selected in pointer_down)
        assert ui_changed_down is True  # Selection changed in pointer_down
        assert ui_changed_up is False   # No additional change needed in pointer_up
        assert self.graph.selected_node_id == "node1"
    
    def test_handle_pointer_up_after_click_same_node(self):
        """Test pointer up after clicking on already selected node"""
        # Select node1
        self.graph.select_node("node1")
        
        # Click on same node
        self.graph.handle_pointer_down(110, 110)
        ui_changed = self.graph.handle_pointer_up(112, 112)
        
        # Should deselect the node
        assert self.graph.pointer_state == PointerState.IDLE
        assert ui_changed is True
        assert self.graph.selected_node_id is None
    
    def test_handle_pointer_up_after_click_empty_area(self):
        """Test pointer up after clicking on empty area"""
        # Select a node first
        self.graph.select_node("node1")
        
        # Click on empty area
        self.graph.handle_pointer_down(50, 50)
        ui_changed = self.graph.handle_pointer_up(52, 52)
        
        # Should deselect current selection
        assert self.graph.pointer_state == PointerState.IDLE
        assert ui_changed is True
        assert self.graph.selected_node_id is None
    
    def test_handle_pointer_up_after_dragging(self):
        """Test pointer up after dragging a node"""
        # Start dragging
        self.graph.handle_pointer_down(110, 110)
        self.graph.handle_pointer_move(120, 120)  # Start dragging
        
        # Finish dragging
        ui_changed = self.graph.handle_pointer_up(125, 125)
        
        # Should reset to IDLE state
        assert self.graph.pointer_state == PointerState.IDLE
        assert self.graph.drag_node_id is None
        
        # Should refresh UI since we were dragging
        assert ui_changed is True
        
        # Node should stay selected
        assert self.graph.selected_node_id == "node1"
    
    def test_handle_pointer_up_idle_state(self):
        """Test pointer up when in IDLE state"""
        ui_changed = self.graph.handle_pointer_up(100, 100)
        
        # Should not change anything
        assert ui_changed is False
        assert self.graph.pointer_state == PointerState.IDLE
    
    def test_movement_integration_scenario(self):
        """Test complete movement scenario integration"""
        # Initial state: no selection
        assert self.graph.selected_node_id is None
        assert self.graph.pointer_state == PointerState.IDLE
        
        # 1. Click and drag node1
        self.graph.handle_pointer_down(110, 110)  # Click on node1
        assert self.graph.selected_node_id == "node1"
        assert self.graph.pointer_state == PointerState.PRESSED
        
        # 2. Move beyond threshold to start dragging
        self.graph.handle_pointer_move(120, 120)
        assert self.graph.pointer_state == PointerState.DRAGGING
        
        # 3. Continue dragging
        self.graph.handle_pointer_move(150, 130)
        assert self.node1.x == 140.0  # 150 - 10 (offset)
        assert self.node1.y == 120.0  # 130 - 10 (offset)
        
        # 4. Release mouse
        self.graph.handle_pointer_up(150, 130)
        assert self.graph.pointer_state == PointerState.IDLE
        assert self.graph.selected_node_id == "node1"  # Still selected
        
        # 5. Click on same node again (should deselect)
        self.graph.handle_pointer_down(150, 130)
        self.graph.handle_pointer_up(152, 132)  # Small movement
        assert self.graph.selected_node_id is None  # Deselected
    
    def test_movement_with_connection_updates(self):
        """Test that connections are properly updated when nodes move"""
        # Add a connection between node1 and node2
        connection = Connection("conn1", "node1", "out", "node2", "in1")
        self.graph.add_connection(connection)
        
        # Get initial connection position
        initial_result = self.graph.to_slint_format()
        initial_conn = initial_result["connections"][0]
        initial_start_x = initial_conn["start_x"]
        initial_start_y = initial_conn["start_y"]
        
        # Move node1
        self.graph.move_node("node1", 200, 200)
        
        # Check that connection positions updated
        updated_result = self.graph.to_slint_format()
        updated_conn = updated_result["connections"][0]
        
        # Start position should have changed (node1 moved)
        assert updated_conn["start_x"] != initial_start_x
        assert updated_conn["start_y"] != initial_start_y
        
        # End position should be unchanged (node2 didn't move)
        assert updated_conn["end_x"] == initial_conn["end_x"]
        assert updated_conn["end_y"] == initial_conn["end_y"]


class TestCreateDemoGraph:
    """Test cases for the create_demo_graph function"""
    
    def test_demo_graph_creation(self):
        """Test that demo graph is created correctly"""
        graph = create_demo_graph()
        
        assert isinstance(graph, GraphData)
        assert len(graph.nodes) == 8  # 3 inputs + 3 gates + 2 outputs
        assert len(graph.connections) == 7  # 7 connections
    
    def test_demo_graph_nodes(self):
        """Test that demo graph contains expected nodes"""
        graph = create_demo_graph()
        
        expected_node_ids = [
            "input_a", "input_b", "input_c",
            "and_gate", "or_gate", "not_gate",
            "output_a", "output_b"
        ]
        
        for node_id in expected_node_ids:
            assert node_id in graph.nodes
    
    def test_demo_graph_node_types(self):
        """Test that demo graph nodes have correct types"""
        graph = create_demo_graph()
        
        assert graph.nodes["input_a"].node_type == "input"
        assert graph.nodes["input_b"].node_type == "input"
        assert graph.nodes["input_c"].node_type == "input"
        assert graph.nodes["and_gate"].node_type == "and"
        assert graph.nodes["or_gate"].node_type == "or"
        assert graph.nodes["not_gate"].node_type == "not"
        assert graph.nodes["output_a"].node_type == "output"
        assert graph.nodes["output_b"].node_type == "output"
    
    def test_demo_graph_connections(self):
        """Test that demo graph contains expected connections"""
        graph = create_demo_graph()
        
        expected_connections = [
            ("c1", "input_a", "out", "and_gate", "in1"),
            ("c2", "input_b", "out", "and_gate", "in2"),
            ("c3", "input_b", "out", "or_gate", "in1"),
            ("c4", "input_c", "out", "or_gate", "in2"),
            ("c5", "and_gate", "out", "not_gate", "in"),
            ("c6", "not_gate", "out", "output_a", "in"),
            ("c7", "or_gate", "out", "output_b", "in"),
        ]
        
        for conn_id, from_node, from_conn, to_node, to_conn in expected_connections:
            assert conn_id in graph.connections
            connection = graph.connections[conn_id]
            assert connection.from_node_id == from_node
            assert connection.from_connector_id == from_conn
            assert connection.to_node_id == to_node
            assert connection.to_connector_id == to_conn
    
    def test_demo_graph_slint_format(self):
        """Test that demo graph can be converted to Slint format"""
        graph = create_demo_graph()
        result = graph.to_slint_format()
        
        assert "nodes" in result
        assert "connections" in result
        assert len(result["nodes"]) == 8
        assert len(result["connections"]) == 7
        
        # Verify that all nodes have proper connector data
        for node_data in result["nodes"]:
            assert "connectors" in node_data
            assert len(node_data["connectors"]) > 0
            for connector in node_data["connectors"]:
                assert "id" in connector
                assert "x" in connector
                assert "y" in connector
                assert "is_input" in connector


class TestGraphDataLabelEditing:
    """Test cases for label editing functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
        
        # Create test nodes
        self.input_node = Node.create("input1", NODE_REGISTRY.get_definition("input"), 50, 50, label="Test Input")
        self.and_gate = Node.create("and1", NODE_REGISTRY.get_definition("and"), 200, 80, label="Test AND")
        
        # Add nodes to graph
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
    
    def test_editing_state_initialization(self):
        """Test that editing state is properly initialized"""
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
        assert self.graph.last_click_time == 0.0
        assert self.graph.double_click_threshold == 0.5
    
    def test_start_label_edit_valid_node(self):
        """Test starting label edit for a valid node"""
        result = self.graph.start_label_edit("input1")
        
        assert result == True  # UI should refresh
        assert self.graph.editing_node_id == "input1"
        assert self.graph.editing_text == "Test Input"
    
    def test_start_label_edit_invalid_node(self):
        """Test starting label edit for an invalid node"""
        with patch.object(self.graph.logger, 'warning') as mock_warning:
            result = self.graph.start_label_edit("nonexistent")
            
            assert result == False  # UI should not refresh
            assert self.graph.editing_node_id is None
            assert self.graph.editing_text == ""
            mock_warning.assert_called_once()
    
    def test_start_label_edit_cancels_existing_edit(self):
        """Test that starting a new edit cancels any existing edit"""
        # Start first edit
        self.graph.start_label_edit("input1")
        assert self.graph.editing_node_id == "input1"
        
        # Start second edit - should cancel first
        result = self.graph.start_label_edit("and1")
        
        assert result == True
        assert self.graph.editing_node_id == "and1"
        assert self.graph.editing_text == "Test AND"
    
    def test_complete_label_edit_valid(self):
        """Test completing a valid label edit"""
        self.graph.start_label_edit("input1")
        
        result = self.graph.complete_label_edit("input1", "  New Label  ")
        
        assert result == True  # UI should refresh
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
        assert self.graph.nodes["input1"].label == "New Label"  # Should be stripped
    
    def test_complete_label_edit_wrong_node(self):
        """Test completing edit for wrong node"""
        self.graph.start_label_edit("input1")
        
        with patch.object(self.graph.logger, 'warning') as mock_warning:
            result = self.graph.complete_label_edit("and1", "Wrong Node")
            
            assert result == False
            assert self.graph.editing_node_id == "input1"  # Should still be editing input1
            mock_warning.assert_called_once()
    
    def test_complete_label_edit_nonexistent_node(self):
        """Test completing edit for nonexistent node"""
        self.graph.start_label_edit("input1")
        
        # Remove the node while editing
        del self.graph.nodes["input1"]
        
        with patch.object(self.graph.logger, 'warning') as mock_warning:
            result = self.graph.complete_label_edit("input1", "New Label")
            
            assert result == True  # UI should refresh (edit was cancelled)
            assert self.graph.editing_node_id is None
            assert self.graph.editing_text == ""
            mock_warning.assert_called_once()
    
    def test_complete_label_edit_not_editing(self):
        """Test completing edit when not currently editing"""
        with patch.object(self.graph.logger, 'warning') as mock_warning:
            result = self.graph.complete_label_edit("input1", "New Label")
            
            assert result == False
            mock_warning.assert_called_once()
    
    def test_cancel_label_edit_while_editing(self):
        """Test cancelling an active edit"""
        self.graph.start_label_edit("input1")
        original_label = self.graph.nodes["input1"].label
        
        result = self.graph.cancel_label_edit()
        
        assert result == True  # UI should refresh
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
        assert self.graph.nodes["input1"].label == original_label  # Unchanged
    
    def test_cancel_label_edit_not_editing(self):
        """Test cancelling when not editing"""
        result = self.graph.cancel_label_edit()
        
        assert result == False  # UI should not refresh
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
    
    def test_update_editing_text(self):
        """Test updating editing text"""
        self.graph.start_label_edit("input1")
        
        self.graph.update_editing_text("New Text")
        
        assert self.graph.editing_text == "New Text"
        # Original node label should be unchanged until complete
        assert self.graph.nodes["input1"].label == "Test Input"
    
    def test_handle_double_click_on_node(self):
        """Test double-click handling on a node"""
        result = self.graph.handle_double_click(60.0, 60.0)  # Click on input1
        
        assert result == True  # UI should refresh
        assert self.graph.editing_node_id == "input1"
        assert self.graph.editing_text == "Test Input"
    
    def test_handle_double_click_on_empty_area(self):
        """Test double-click handling on empty area"""
        result = self.graph.handle_double_click(0.0, 0.0)  # Click on empty area
        
        assert result == False  # UI should not refresh
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
    
    def test_is_double_click_timing(self):
        """Test double-click timing detection"""
        import time
        
        # First click
        current_time = time.time()
        is_double = self.graph.is_double_click(current_time)
        assert is_double == False  # First click is never a double-click
        
        # Second click within threshold
        is_double = self.graph.is_double_click(current_time + 0.3)
        assert is_double == True
        
        # Third click outside threshold
        is_double = self.graph.is_double_click(current_time + 1.0)
        assert is_double == False
    
    def test_is_double_click_threshold_boundary(self):
        """Test double-click threshold boundary conditions"""
        import time
        
        current_time = time.time()
        self.graph.is_double_click(current_time)  # First click
        
        # Click exactly at threshold
        is_double = self.graph.is_double_click(current_time + self.graph.double_click_threshold)
        assert is_double == True
        
        # Click just outside threshold
        current_time = time.time()
        self.graph.is_double_click(current_time)  # Reset
        is_double = self.graph.is_double_click(current_time + self.graph.double_click_threshold + 0.01)
        assert is_double == False
    
    def test_to_slint_format_includes_editing_state_not_editing(self):
        """Test that Slint format includes editing state when not editing"""
        result = self.graph.to_slint_format()
        
        assert "editing_node_id" in result
        assert "editing_text" in result
        assert result["editing_node_id"] == ""
        assert result["editing_text"] == ""
    
    def test_to_slint_format_includes_editing_state_while_editing(self):
        """Test that Slint format includes editing state while editing"""
        self.graph.start_label_edit("input1")
        self.graph.update_editing_text("Modified Text")
        
        result = self.graph.to_slint_format()
        
        assert "editing_node_id" in result
        assert "editing_text" in result
        assert result["editing_node_id"] == "input1"
        assert result["editing_text"] == "Modified Text"
    
    def test_label_editing_integration_scenario(self):
        """Test a complete label editing scenario"""
        # Start editing
        self.graph.start_label_edit("input1")
        assert self.graph.editing_node_id == "input1"
        
        # Update text as user types
        self.graph.update_editing_text("New")
        self.graph.update_editing_text("New Label")
        assert self.graph.editing_text == "New Label"
        
        # Complete editing
        self.graph.complete_label_edit("input1", "Final Label")
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
        assert self.graph.nodes["input1"].label == "Final Label"
    
    def test_label_editing_with_whitespace_handling(self):
        """Test that label editing properly handles whitespace"""
        self.graph.start_label_edit("input1")
        
        # Complete with whitespace that should be stripped
        self.graph.complete_label_edit("input1", "   Spaced Label   ")
        
        assert self.graph.nodes["input1"].label == "Spaced Label"
    
    def test_multiple_nodes_editing_isolation(self):
        """Test that editing one node doesn't affect others"""
        original_and_label = self.graph.nodes["and1"].label
        
        # Edit input node
        self.graph.start_label_edit("input1")
        self.graph.complete_label_edit("input1", "Modified Input")
        
        # AND node should be unchanged
        assert self.graph.nodes["and1"].label == original_and_label
        assert self.graph.nodes["input1"].label == "Modified Input"
    
    def test_deselect_node_cancels_editing(self):
        """Test that deselecting a node cancels any active label editing"""
        # Select and start editing a node
        self.graph.select_node("input1")
        self.graph.start_label_edit("input1")
        assert self.graph.editing_node_id == "input1"
        assert self.graph.selected_node_id == "input1"
        
        # Deselect the node
        self.graph.deselect_node()
        
        # Editing should be cancelled
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
        assert self.graph.selected_node_id is None
    
    def test_deselect_node_different_from_editing_node(self):
        """Test that deselecting a different node doesn't cancel editing"""
        # Start editing one node
        self.graph.start_label_edit("input1")
        
        # Select and then deselect a different node
        self.graph.select_node("and1")
        self.graph.deselect_node()
        
        # Original editing should remain active
        assert self.graph.editing_node_id == "input1"
        assert self.graph.editing_text == "Test Input"
        assert self.graph.selected_node_id is None
    
    def test_deselect_node_not_editing(self):
        """Test that deselecting when not editing works normally"""
        # Select a node without editing
        self.graph.select_node("input1")
        assert self.graph.selected_node_id == "input1"
        assert self.graph.editing_node_id is None
        
        # Deselect the node
        self.graph.deselect_node()
        
        # Should just deselect without affecting editing state
        assert self.graph.selected_node_id is None
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
    
    def test_handle_mouse_click_empty_area_cancels_editing(self):
        """Test that clicking empty area cancels editing of selected node"""
        # Select and start editing a node
        self.graph.select_node("input1")
        self.graph.start_label_edit("input1")
        assert self.graph.editing_node_id == "input1"
        
        # Click on empty area (this should deselect and cancel editing)
        result = self.graph.handle_mouse_click(0.0, 0.0)
        
        # Should deselect and cancel editing
        assert result == True  # Selection changed
        assert self.graph.selected_node_id is None
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
    
    def test_handle_mouse_click_same_node_cancels_editing(self):
        """Test that clicking the same selected/editing node cancels editing"""
        # Select and start editing a node
        self.graph.select_node("input1")
        self.graph.start_label_edit("input1")
        assert self.graph.editing_node_id == "input1"
        
        # Click on the same node (this should deselect and cancel editing)
        result = self.graph.handle_mouse_click(60.0, 60.0)  # Click on input1
        
        # Should deselect and cancel editing
        assert result == True  # Selection changed
        assert self.graph.selected_node_id is None
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
    
    def test_select_different_node_while_editing(self):
        """Test that selecting a different node while editing cancels current edit"""
        # Start editing one node
        self.graph.select_node("input1")
        self.graph.start_label_edit("input1")
        assert self.graph.editing_node_id == "input1"
        
        # Select a different node (this implicitly deselects the first)
        self.graph.select_node("and1")
        
        # The editing should still be active since we only changed selection
        # (This tests that selection change alone doesn't cancel editing)
        assert self.graph.editing_node_id == "input1"  # Still editing input1
        assert self.graph.selected_node_id == "and1"   # But and1 is selected
        
        # However, if we explicitly deselect the currently selected node
        self.graph.deselect_node()
        
        # Editing should remain since we deselected and1, not input1
        assert self.graph.editing_node_id == "input1"  # Still editing input1
        assert self.graph.selected_node_id is None     # Nothing selected


class TestGraphDataConnectionSelection:
    """Test cases for connection selection functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
        
        # Create test nodes
        input_node = Node.create("input1", NODE_REGISTRY.get_definition("input"), 50, 50, label="A")
        and_gate = Node.create("and1", NODE_REGISTRY.get_definition("and"), 200, 80, label="AND")
        output_node = Node.create("output1", NODE_REGISTRY.get_definition("output"), 350, 100, label="OUT")
        
        # Add nodes to graph
        self.graph.add_node(input_node)
        self.graph.add_node(and_gate)
        self.graph.add_node(output_node)
        
        # Add test connections
        self.connection1 = Connection("c1", "input1", "out", "and1", "in1")
        self.connection2 = Connection("c2", "and1", "out", "output1", "in")
        self.graph.add_connection(self.connection1)
        self.graph.add_connection(self.connection2)
    
    def test_connection_selection_state_initialization(self):
        """Test that connection selection state is properly initialized"""
        assert self.graph.selected_connection_id is None
    
    def test_point_to_line_segment_distance(self):
        """Test distance calculation from point to line segment"""
        # Point on line segment
        distance = self.graph.point_to_line_segment_distance(50, 50, 0, 0, 100, 100)
        assert abs(distance - 0.0) < 0.1  # Should be approximately 0
        
        # Point perpendicular to line segment center
        distance = self.graph.point_to_line_segment_distance(50, 0, 0, 0, 100, 0)
        assert abs(distance - 0.0) < 0.1
        
        # Point away from horizontal line
        distance = self.graph.point_to_line_segment_distance(50, 10, 0, 0, 100, 0)
        assert abs(distance - 10.0) < 0.1
        
        # Point beyond line segment end
        distance = self.graph.point_to_line_segment_distance(150, 0, 0, 0, 100, 0)
        assert abs(distance - 50.0) < 0.1
    
    def test_is_point_on_connection(self):
        """Test point hit-testing on connections"""
        # This test uses actual connection coordinates
        # connection1: input1.out -> and1.in1
        # Should be approximately from (99,74) to (199,94) based on connector positions
        
        # Point near the line should hit
        assert self.graph.is_point_on_connection(self.connection1, 149, 84)  # Midpoint approximately
        
        # Point far from line should not hit
        assert not self.graph.is_point_on_connection(self.connection1, 100, 150)
        
        # Test with custom tolerance
        assert self.graph.is_point_on_connection(self.connection1, 149, 94, tolerance=15.0)  # Larger tolerance
        assert not self.graph.is_point_on_connection(self.connection1, 149, 94, tolerance=5.0)  # Smaller tolerance
    
    def test_get_connection_at_position(self):
        """Test getting connection at position"""
        # Should find connection1 near its path
        result = self.graph.get_connection_at_position(149, 84)
        assert result == "c1"
        
        # Should not find connection at empty area
        result = self.graph.get_connection_at_position(500, 500)
        assert result is None
        
        # Test priority - later connections should take priority if overlapping
        # For this we need overlapping connections, but our test setup doesn't have them
        # So this tests the basic case
        result = self.graph.get_connection_at_position(314, 116)  # Near connection2
        assert result == "c2"
    
    def test_select_connection_valid(self):
        """Test selecting a valid connection"""
        self.graph.select_connection("c1")
        
        assert self.graph.selected_connection_id == "c1"
        assert self.graph.selected_node_id is None  # Should deselect nodes
    
    def test_select_connection_invalid(self):
        """Test selecting an invalid connection"""
        with pytest.raises(ValueError, match="Connection with ID 'nonexistent' does not exist"):
            self.graph.select_connection("nonexistent")
    
    def test_select_connection_deselects_node(self):
        """Test that selecting a connection deselects any selected node"""
        # First select a node
        self.graph.select_node("input1")
        assert self.graph.selected_node_id == "input1"
        
        # Then select a connection
        self.graph.select_connection("c1")
        
        assert self.graph.selected_connection_id == "c1"
        assert self.graph.selected_node_id is None
    
    def test_select_node_deselects_connection(self):
        """Test that selecting a node deselects any selected connection"""
        # First select a connection
        self.graph.select_connection("c1")
        assert self.graph.selected_connection_id == "c1"
        
        # Then select a node
        self.graph.select_node("input1")
        
        assert self.graph.selected_node_id == "input1"
        assert self.graph.selected_connection_id is None
    
    def test_deselect_connection(self):
        """Test deselecting a connection"""
        self.graph.select_connection("c1")
        assert self.graph.selected_connection_id == "c1"
        
        self.graph.deselect_connection()
        
        assert self.graph.selected_connection_id is None
    
    def test_deselect_connection_when_none_selected(self):
        """Test deselecting when no connection is selected"""
        self.graph.deselect_connection()
        
        assert self.graph.selected_connection_id is None
    
    def test_get_selected_connection(self):
        """Test getting selected connection"""
        assert self.graph.get_selected_connection() is None
        
        self.graph.select_connection("c1")
        assert self.graph.get_selected_connection() == "c1"
    
    def test_handle_pointer_down_connection_selection(self):
        """Test pointer down handling for connection selection"""
        # Click near connection1
        result = self.graph.handle_pointer_down(149, 84)
        
        assert result == True  # UI should refresh
        assert self.graph.selected_connection_id == "c1"
        assert self.graph.selected_node_id is None
    
    def test_handle_pointer_down_node_priority(self):
        """Test that nodes have priority over connections"""
        # Click on a node position (should select node, not connection)
        result = self.graph.handle_pointer_down(75, 75)  # On input1
        
        assert result == True  # UI should refresh
        assert self.graph.selected_node_id == "input1"
        assert self.graph.selected_connection_id is None
    
    def test_handle_mouse_click_connection_selection(self):
        """Test mouse click connection selection"""
        # Click on connection
        result = self.graph.handle_mouse_click(149, 84)
        
        assert result == True  # Selection changed
        assert self.graph.selected_connection_id == "c1"
        assert self.graph.selected_node_id is None
    
    def test_handle_mouse_click_connection_deselection(self):
        """Test clicking same connection deselects it"""
        self.graph.select_connection("c1")
        
        # Click on same connection again
        result = self.graph.handle_mouse_click(149, 84)
        
        assert result == True  # Selection changed
        assert self.graph.selected_connection_id is None
    
    def test_handle_mouse_click_different_connection(self):
        """Test clicking different connection switches selection"""
        self.graph.select_connection("c1")
        
        # Click on different connection
        result = self.graph.handle_mouse_click(314, 116)  # Near connection2
        
        assert result == True  # Selection changed
        assert self.graph.selected_connection_id == "c2"
    
    def test_handle_mouse_click_empty_area_deselects_connection(self):
        """Test clicking empty area deselects connection"""
        self.graph.select_connection("c1")
        
        # Click on empty area
        result = self.graph.handle_mouse_click(500, 500)
        
        assert result == True  # Selection changed
        assert self.graph.selected_connection_id is None
    
    def test_to_slint_format_includes_selected_connections_none(self):
        """Test Slint format includes empty selected connections"""
        result = self.graph.to_slint_format()
        
        assert "selected_connections" in result
        assert result["selected_connections"] == []
    
    def test_to_slint_format_includes_selected_connections_with_selection(self):
        """Test Slint format includes selected connection"""
        self.graph.select_connection("c1")
        
        result = self.graph.to_slint_format()
        
        assert "selected_connections" in result
        assert result["selected_connections"] == ["c1"]
    
    def test_connection_selection_integration_scenario(self):
        """Test a complete connection selection scenario"""
        # Start with no selection
        assert self.graph.selected_connection_id is None
        assert self.graph.selected_node_id is None
        
        # Select a connection
        self.graph.select_connection("c1")
        assert self.graph.selected_connection_id == "c1"
        
        # Select a different connection
        self.graph.select_connection("c2")
        assert self.graph.selected_connection_id == "c2"
        
        # Select a node (should deselect connection)
        self.graph.select_node("input1")
        assert self.graph.selected_node_id == "input1"
        assert self.graph.selected_connection_id is None
        
        # Select connection again (should deselect node)
        self.graph.select_connection("c1")
        assert self.graph.selected_connection_id == "c1"
        assert self.graph.selected_node_id is None
    
    def test_connection_hit_testing_edge_cases(self):
        """Test connection hit-testing edge cases"""
        # Test with zero-length line (degenerate case)
        # This shouldn't happen in practice but tests the math
        fake_connection = Connection("test", "input1", "out", "input1", "out")
        
        # Point at same location should hit
        assert self.graph.is_point_on_connection(fake_connection, 99, 74, tolerance=1.0)
        
        # Point away should not hit
        assert not self.graph.is_point_on_connection(fake_connection, 200, 200, tolerance=1.0)


class TestGraphDataDeletion:
    """Test cases for node and connection deletion functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
        
        # Create test nodes
        input_node = Node.create("input1", NODE_REGISTRY.get_definition("input"), 50, 50, label="A")
        and_gate = Node.create("and1", NODE_REGISTRY.get_definition("and"), 200, 80, label="AND")
        or_gate = Node.create("or1", NODE_REGISTRY.get_definition("or"), 200, 220, label="OR")
        output_node = Node.create("output1", NODE_REGISTRY.get_definition("output"), 350, 100, label="OUT")
        
        # Add nodes to graph
        self.graph.add_node(input_node)
        self.graph.add_node(and_gate)
        self.graph.add_node(or_gate)
        self.graph.add_node(output_node)
        
        # Add test connections
        self.connection1 = Connection("c1", "input1", "out", "and1", "in1")
        self.connection2 = Connection("c2", "and1", "out", "output1", "in")
        self.connection3 = Connection("c3", "input1", "out", "or1", "in1")
        self.graph.add_connection(self.connection1)
        self.graph.add_connection(self.connection2)
        self.graph.add_connection(self.connection3)
    
    def test_delete_node_valid(self):
        """Test deleting a valid node"""
        # Verify node exists
        assert "and1" in self.graph.nodes
        assert len(self.graph.nodes) == 4
        
        # Delete the node
        result = self.graph.delete_node("and1")
        
        # Verify node is deleted
        assert result == True
        assert "and1" not in self.graph.nodes
        assert len(self.graph.nodes) == 3
    
    def test_delete_node_invalid(self):
        """Test deleting a non-existent node"""
        initial_count = len(self.graph.nodes)
        
        result = self.graph.delete_node("nonexistent")
        
        assert result == False
        assert len(self.graph.nodes) == initial_count
    
    def test_delete_node_cascade_connections(self):
        """Test that deleting a node also deletes its connections"""
        # Verify initial state
        assert len(self.graph.connections) == 3
        assert "c1" in self.graph.connections  # input1 -> and1
        assert "c2" in self.graph.connections  # and1 -> output1
        assert "c3" in self.graph.connections  # input1 -> or1
        
        # Delete and1 node (should cascade delete c1 and c2)
        result = self.graph.delete_node("and1")
        
        assert result == True
        assert len(self.graph.connections) == 1  # Only c3 should remain
        assert "c1" not in self.graph.connections
        assert "c2" not in self.graph.connections
        assert "c3" in self.graph.connections  # Unrelated connection preserved
    
    def test_delete_node_selected_state_cleanup(self):
        """Test that deleting a selected node clears selection"""
        # Select the node first
        self.graph.select_node("and1")
        assert self.graph.selected_node_id == "and1"
        
        # Delete the node
        result = self.graph.delete_node("and1")
        
        assert result == True
        assert self.graph.selected_node_id is None
    
    def test_delete_node_editing_state_cleanup(self):
        """Test that deleting a node being edited cancels editing"""
        # Start editing the node
        self.graph.start_label_edit("and1")
        assert self.graph.editing_node_id == "and1"
        assert self.graph.editing_text == "AND"
        
        # Delete the node
        result = self.graph.delete_node("and1")
        
        assert result == True
        assert self.graph.editing_node_id is None
        assert self.graph.editing_text == ""
    
    def test_delete_node_unrelated_state_preserved(self):
        """Test that deleting a node doesn't affect unrelated selections"""
        # Select a different node and start editing another
        self.graph.select_node("input1")
        self.graph.start_label_edit("or1")
        
        # Delete unrelated node
        result = self.graph.delete_node("and1")
        
        assert result == True
        assert self.graph.selected_node_id == "input1"  # Preserved
        assert self.graph.editing_node_id == "or1"      # Preserved
        assert self.graph.editing_text == "OR"          # Preserved
    
    def test_delete_connection_valid(self):
        """Test deleting a valid connection"""
        # Verify connection exists
        assert "c1" in self.graph.connections
        assert len(self.graph.connections) == 3
        
        # Delete the connection
        result = self.graph.delete_connection("c1")
        
        # Verify connection is deleted
        assert result == True
        assert "c1" not in self.graph.connections
        assert len(self.graph.connections) == 2
        
        # Verify nodes are unaffected
        assert len(self.graph.nodes) == 4
    
    def test_delete_connection_invalid(self):
        """Test deleting a non-existent connection"""
        initial_count = len(self.graph.connections)
        
        result = self.graph.delete_connection("nonexistent")
        
        assert result == False
        assert len(self.graph.connections) == initial_count
    
    def test_delete_connection_selected_state_cleanup(self):
        """Test that deleting a selected connection clears selection"""
        # Select the connection first
        self.graph.select_connection("c1")
        assert self.graph.selected_connection_id == "c1"
        
        # Delete the connection
        result = self.graph.delete_connection("c1")
        
        assert result == True
        assert self.graph.selected_connection_id is None
    
    def test_delete_selected_node(self):
        """Test deleting selected node via delete_selected"""
        # Select a node
        self.graph.select_node("and1")
        initial_node_count = len(self.graph.nodes)
        initial_connection_count = len(self.graph.connections)
        
        # Delete selected item
        result = self.graph.delete_selected()
        
        assert result == True
        assert len(self.graph.nodes) == initial_node_count - 1
        assert len(self.graph.connections) < initial_connection_count  # Cascade deletion
        assert self.graph.selected_node_id is None
    
    def test_delete_selected_connection(self):
        """Test deleting selected connection via delete_selected"""
        # Select a connection
        self.graph.select_connection("c1")
        initial_connection_count = len(self.graph.connections)
        initial_node_count = len(self.graph.nodes)
        
        # Delete selected item
        result = self.graph.delete_selected()
        
        assert result == True
        assert len(self.graph.connections) == initial_connection_count - 1
        assert len(self.graph.nodes) == initial_node_count  # Nodes unaffected
        assert self.graph.selected_connection_id is None
    
    def test_delete_selected_nothing(self):
        """Test delete_selected when nothing is selected"""
        # Ensure nothing is selected
        assert self.graph.selected_node_id is None
        assert self.graph.selected_connection_id is None
        
        initial_node_count = len(self.graph.nodes)
        initial_connection_count = len(self.graph.connections)
        
        # Attempt to delete
        result = self.graph.delete_selected()
        
        assert result == False
        assert len(self.graph.nodes) == initial_node_count
        assert len(self.graph.connections) == initial_connection_count
    
    def test_cascade_deletion_complex_graph(self):
        """Test cascade deletion with node having multiple connections"""
        # input1 has connections c1 and c3, so deleting it should remove both
        assert "c1" in self.graph.connections  # input1 -> and1
        assert "c3" in self.graph.connections  # input1 -> or1
        assert len(self.graph.connections) == 3
        
        # Delete input1
        result = self.graph.delete_node("input1")
        
        assert result == True
        assert "input1" not in self.graph.nodes
        assert "c1" not in self.graph.connections  # Cascade deleted
        assert "c3" not in self.graph.connections  # Cascade deleted
        assert "c2" in self.graph.connections     # Unrelated connection preserved
        assert len(self.graph.connections) == 1
    
    def test_delete_integration_scenario(self):
        """Test a complete deletion workflow scenario"""
        # Start with full graph
        assert len(self.graph.nodes) == 4
        assert len(self.graph.connections) == 3
        
        # Select and delete a connection
        self.graph.select_connection("c2")
        result1 = self.graph.delete_selected()
        assert result1 == True
        assert len(self.graph.connections) == 2
        assert self.graph.selected_connection_id is None
        
        # Select and delete a node (with cascade)
        self.graph.select_node("input1")
        result2 = self.graph.delete_selected()
        assert result2 == True
        assert len(self.graph.nodes) == 3
        assert len(self.graph.connections) == 0  # c1 and c3 cascade deleted
        assert self.graph.selected_node_id is None
        
        # Try to delete when nothing selected
        result3 = self.graph.delete_selected()
        assert result3 == False
        
        # Graph should be stable
        assert len(self.graph.nodes) == 3
        assert len(self.graph.connections) == 0
    
    def test_delete_preserves_other_selections(self):
        """Test that deletion of unrelated items preserves current selection"""
        # Test 1: Select node1, but delete node2 (unrelated)
        self.graph.select_node("input1")
        result = self.graph.delete_node("or1")  # Delete unrelated node
        
        assert result == True
        assert self.graph.selected_node_id == "input1"  # Selection preserved
        
        # Test 2: Select one connection, delete a different one
        # After deleting or1, we still have c1 (input1->and1) and c2 (and1->output1)
        # c3 (input1->or1) was cascade deleted with or1
        self.graph.select_connection("c1")
        result = self.graph.delete_connection("c2")  # Delete different connection
        
        assert result == True
        assert self.graph.selected_connection_id == "c1"  # Selection preserved