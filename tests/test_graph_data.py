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