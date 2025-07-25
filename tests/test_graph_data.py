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
    Net,
    NetConnector,
    NetWaypoint,
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
    
    def test_node_state_initialization_default(self):
        """Test that nodes are initialized with None value by default"""
        definition = NODE_REGISTRY.get_definition("input")
        node = Node.create("test_input", definition, 0.0, 0.0)
        
        assert node.value is None
    
    def test_node_state_initialization_with_value(self):
        """Test creating nodes with specific state values"""
        definition = NODE_REGISTRY.get_definition("input")
        
        # Test with True value
        node_true = Node.create("input_true", definition, 0.0, 0.0, value=True)
        assert node_true.value is True
        
        # Test with False value  
        node_false = Node.create("input_false", definition, 0.0, 0.0, value=False)
        assert node_false.value is False
        
        # Test with None value explicitly
        node_none = Node.create("input_none", definition, 0.0, 0.0, value=None)
        assert node_none.value is None
    
    def test_node_state_manual_creation(self):
        """Test manual node creation with state values"""
        connectors = [Connector("out", 46.0, 21.0, False)]
        
        # Test with different state values
        node_high = Node("test1", "input", 0.0, 0.0, 50.0, 50.0, "HIGH", "green", connectors, True)
        assert node_high.value is True
        
        node_low = Node("test2", "input", 0.0, 0.0, 50.0, 50.0, "LOW", "red", connectors, False)
        assert node_low.value is False
        
        node_undefined = Node("test3", "input", 0.0, 0.0, 50.0, 50.0, "UNDEF", "gray", connectors, None)
        assert node_undefined.value is None


class TestNet:
    """Test cases for the Net class"""
    
    def test_net_creation_two_point(self):
        """Test basic two-point net creation"""
        net = Net.create_two_point(
            id="net1",
            from_node_id="node1",
            from_connector_id="out",
            to_node_id="node2",
            to_connector_id="in"
        )
        
        assert net.id == "net1"
        assert len(net.connectors) == 2
        assert net.connectors[0].node_id == "node1"
        assert net.connectors[0].connector_id == "out"
        assert net.connectors[1].node_id == "node2"
        assert net.connectors[1].connector_id == "in"
    
    def test_net_creation_multi_point(self):
        """Test multi-point net creation"""
        connectors = [
            ("node1", "out"),
            ("node2", "in1"),
            ("node3", "in2"),
            ("node4", "in3")
        ]
        net = Net.create_multi_point("multi_net", connectors)
        
        assert net.id == "multi_net"
        assert len(net.connectors) == 4
        assert net.connectors[0].node_id == "node1"
        assert net.connectors[0].connector_id == "out"
        assert net.connectors[3].node_id == "node4"
        assert net.connectors[3].connector_id == "in3"
    
    def test_net_add_connector(self):
        """Test adding connectors to existing net"""
        net = Net.create_two_point("net1", "node1", "out", "node2", "in")
        
        # Add a third connector
        net.add_connector("node3", "in2")
        assert len(net.connectors) == 3
        assert net.has_connector("node3", "in2")
        
        # Try to add duplicate - should raise error
        with pytest.raises(ValueError, match="already exists"):
            net.add_connector("node1", "out")
    
    def test_net_remove_connector(self):
        """Test removing connectors from net"""
        net = Net.create_multi_point("net1", [("node1", "out"), ("node2", "in1"), ("node3", "in2")])
        
        # Remove existing connector
        assert net.remove_connector("node2", "in1") == True
        assert len(net.connectors) == 2
        assert not net.has_connector("node2", "in1")
        
        # Try to remove non-existent connector
        assert net.remove_connector("node5", "in") == False
        assert len(net.connectors) == 2
    
    def test_net_has_connector(self):
        """Test checking if net has specific connector"""
        net = Net.create_two_point("net1", "node1", "out", "node2", "in")
        
        assert net.has_connector("node1", "out") == True
        assert net.has_connector("node2", "in") == True
        assert net.has_connector("node3", "in") == False
        assert net.has_connector("node1", "in") == False
    
    def test_net_get_node_ids(self):
        """Test getting unique node IDs from net"""
        net = Net.create_multi_point("net1", [
            ("node1", "out"),
            ("node2", "in1"), 
            ("node2", "in2"),  # Same node, different connector
            ("node3", "in")
        ])
        
        node_ids = net.get_node_ids()
        assert node_ids == {"node1", "node2", "node3"}
        assert len(node_ids) == 3
    
    def test_net_empty_validation(self):
        """Test that empty nets are not allowed"""
        with pytest.raises(ValueError, match="must have at least one connector"):
            Net("empty_net", [])
    
    def test_netconnector_creation(self):
        """Test NetConnector dataclass"""
        connector = NetConnector("node1", "out")
        assert connector.node_id == "node1"
        assert connector.connector_id == "out"


class TestMultiNodeNet:
    """Test cases for multi-node net functionality"""
    
    def setup_method(self):
        """Set up test fixtures for multi-node net tests"""
        self.graph = GraphData()
        
        # Create test nodes: 1 output driver, 3 input receivers (fan-out scenario)
        self.driver_node = Node.create("driver", NODE_REGISTRY.get_definition("input"), 50, 50, label="DRIVER")
        self.receiver1 = Node.create("recv1", NODE_REGISTRY.get_definition("and"), 200, 50, label="AND1")
        self.receiver2 = Node.create("recv2", NODE_REGISTRY.get_definition("or"), 200, 150, label="OR1") 
        self.receiver3 = Node.create("recv3", NODE_REGISTRY.get_definition("not"), 200, 250, label="NOT1")
        
        # Add nodes to graph
        self.graph.add_node(self.driver_node)
        self.graph.add_node(self.receiver1)
        self.graph.add_node(self.receiver2)
        self.graph.add_node(self.receiver3)
    
    def test_create_fan_out_net(self):
        """Test creating a net with one driver and multiple receivers (fan-out)"""
        # Create a multi-point net: driver.out -> recv1.in1, recv2.in1, recv3.in
        connectors = [
            ("driver", "out"),      # Output connector (driver)
            ("recv1", "in1"),       # Input connector (receiver)
            ("recv2", "in1"),       # Input connector (receiver)
            ("recv3", "in")         # Input connector (receiver)
        ]
        fan_out_net = Net.create_multi_point("fan_out", connectors)
        
        self.graph.add_net(fan_out_net)
        
        # Verify net structure
        assert len(fan_out_net.connectors) == 4
        assert fan_out_net.get_node_ids() == {"driver", "recv1", "recv2", "recv3"}
        
        # Verify helper methods work correctly
        driving_nodes = self.graph.get_driving_nodes("fan_out")
        receiving_nodes = self.graph.get_receiving_nodes("fan_out")
        
        assert driving_nodes == ["driver"]
        assert set(receiving_nodes) == {"recv1", "recv2", "recv3"}
    
    def test_create_multiple_driver_net(self):
        """Test creating a net with multiple drivers (conflict scenario)"""
        # Add another input node as second driver
        driver2 = Node.create("driver2", NODE_REGISTRY.get_definition("input"), 50, 150, label="DRIVER2")
        self.graph.add_node(driver2)
        
        # Create net with two drivers and one receiver
        connectors = [
            ("driver", "out"),      # First driver
            ("driver2", "out"),     # Second driver (conflict)
            ("recv1", "in1")        # Single receiver
        ]
        conflict_net = Net.create_multi_point("conflict", connectors)
        
        self.graph.add_net(conflict_net)
        
        # Verify net structure
        driving_nodes = self.graph.get_driving_nodes("conflict")
        receiving_nodes = self.graph.get_receiving_nodes("conflict")
        
        assert len(driving_nodes) == 2
        assert set(driving_nodes) == {"driver", "driver2"}
        assert receiving_nodes == ["recv1"]
    
    def test_multi_node_net_simulation(self):
        """Test simulation with multi-node nets"""
        # Create a fan-out scenario: input drives three logic gates
        connectors = [
            ("driver", "out"),
            ("recv1", "in1"),  # AND gate input 1
            ("recv2", "in1"),  # OR gate input 1  
            ("recv3", "in")    # NOT gate input
        ]
        fan_out_net = Net.create_multi_point("fan_out", connectors)
        self.graph.add_net(fan_out_net)
        
        # Set driver input value
        self.driver_node.value = True
        
        # Add second inputs to AND and OR gates to make them evaluable
        input2 = Node.create("input2", NODE_REGISTRY.get_definition("input"), 50, 200, label="INPUT2")
        self.graph.add_node(input2)
        input2.value = False
        
        and_input2_net = Net.create_two_point("and_in2", "input2", "out", "recv1", "in2")
        or_input2_net = Net.create_two_point("or_in2", "input2", "out", "recv2", "in2")
        self.graph.add_net(and_input2_net)
        self.graph.add_net(or_input2_net)
        
        # Run simulation
        success = self.graph.simulate()
        assert success
        
        # Verify results: 
        # AND(True, False) = False
        # OR(True, False) = True  
        # NOT(True) = False
        assert self.receiver1.value == False  # AND result
        assert self.receiver2.value == True   # OR result
        assert self.receiver3.value == False  # NOT result
    
    def test_multi_driver_conflict_simulation(self):
        """Test simulation with conflicting drivers"""
        # Add second driver
        driver2 = Node.create("driver2", NODE_REGISTRY.get_definition("input"), 50, 150, label="DRIVER2")
        self.graph.add_node(driver2)
        
        # Create net with conflicting drivers
        connectors = [
            ("driver", "out"),    # Driver 1: True
            ("driver2", "out"),   # Driver 2: False (conflict)
            ("recv3", "in")       # NOT gate input
        ]
        conflict_net = Net.create_multi_point("conflict", connectors)
        self.graph.add_net(conflict_net)
        
        # Set conflicting values
        self.driver_node.value = True
        driver2.value = False
        
        # Simulation should still work (using first driver's value)
        success = self.graph.simulate()
        assert success
        
        # Should use first driver's value: NOT(True) = False
        assert self.receiver3.value == False
    
    def test_net_helper_methods(self):
        """Test helper methods for multi-node nets"""
        # Create a complex net
        connectors = [
            ("driver", "out"),
            ("recv1", "in1"),
            ("recv2", "in1")
        ]
        test_net = Net.create_multi_point("test", connectors)
        self.graph.add_net(test_net)
        
        # Test get_net_connectors
        net_connectors = self.graph.get_net_connectors("test")
        assert len(net_connectors) == 3
        assert any(conn.node_id == "driver" and conn.connector_id == "out" for conn in net_connectors)
        
        # Test add_connector_to_net
        success = self.graph.add_connector_to_net("test", "recv3", "in")
        assert success
        assert len(self.graph.nets["test"].connectors) == 4
        
        # Test remove_connector_from_net
        success = self.graph.remove_connector_from_net("test", "recv2", "in1")
        assert success
        assert len(self.graph.nets["test"].connectors) == 3
        
        # Test with non-existent net
        assert not self.graph.add_connector_to_net("nonexistent", "recv1", "in1")
        assert not self.graph.remove_connector_from_net("nonexistent", "recv1", "in1")
    
    def test_delete_node_with_multi_connector_net(self):
        """Test deleting a node that's connected to multi-connector nets"""
        # Create net with multiple connectors
        connectors = [
            ("driver", "out"),
            ("recv1", "in1"),
            ("recv2", "in1"),
            ("recv3", "in")
        ]
        multi_net = Net.create_multi_point("multi", connectors)
        self.graph.add_net(multi_net)
        
        # Delete one of the receiver nodes
        self.graph.delete_node("recv2")
        
        # Net should still exist but with fewer connectors
        assert "multi" in self.graph.nets
        net = self.graph.nets["multi"]
        assert len(net.connectors) == 3
        assert not net.has_connector("recv2", "in1")
        assert net.has_connector("driver", "out")
        assert net.has_connector("recv1", "in1")
        assert net.has_connector("recv3", "in")
        
        # Delete the driver node
        self.graph.delete_node("driver")
        
        # Net should still exist with remaining connectors
        assert "multi" in self.graph.nets
        net = self.graph.nets["multi"]
        assert len(net.connectors) == 2
        assert not net.has_connector("driver", "out")
        
        # Delete all remaining nodes - net should be removed
        self.graph.delete_node("recv1")
        self.graph.delete_node("recv3")
        
        # Net should be completely removed
        assert "multi" not in self.graph.nets




class TestGraphData:
    """Test cases for the GraphData class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
        
        # Create test nodes using new system
        self.input_node = Node.create("input1", NODE_REGISTRY.get_definition("input"), 50.0, 50.0, label="A")
        self.and_gate = Node.create("and1", NODE_REGISTRY.get_definition("and"), 200.0, 80.0, label="AND")
        self.output_node = Node.create("output1", NODE_REGISTRY.get_definition("output"), 350.0, 90.0, label="OUT")
        
        # Create test net
        self.net = Net.create_two_point("conn1", "input1", "out", "and1", "in1")
    
    def test_graph_data_initialization(self):
        """Test GraphData initialization"""
        graph = GraphData()
        
        assert isinstance(graph.nodes, dict)
        assert isinstance(graph.nets, dict)
        assert len(graph.nodes) == 0
        assert len(graph.nets) == 0
        assert hasattr(graph, 'evaluator')
        assert graph.evaluator is not None
    
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
    
    def test_add_net(self):
        """Test adding nets to the graph"""
        with patch.object(self.graph, 'logger') as mock_logger:
            self.graph.add_net(self.net)
            
            assert "conn1" in self.graph.nets
            assert self.graph.nets["conn1"] == self.net
            mock_logger.debug.assert_called_once_with("Added net conn1 with connectors: [input1:out, and1:in1]")
    
    def test_set_input_value_valid_input_node(self):
        """Test setting value on a valid input node"""
        self.graph.add_node(self.input_node)
        
        with patch.object(self.graph, 'logger') as mock_logger:
            # Set input to True
            result = self.graph.set_input_value("input1", True)
            
            assert result == True  # Should return True for UI refresh
            assert self.graph.nodes["input1"].value == True
            mock_logger.debug.assert_called_once_with("Set input node 'input1' value: None -> True")
            
            # Set input to False
            mock_logger.reset_mock()
            result = self.graph.set_input_value("input1", False)
            
            assert result == True
            assert self.graph.nodes["input1"].value == False
            mock_logger.debug.assert_called_once_with("Set input node 'input1' value: True -> False")
    
    def test_set_input_value_nonexistent_node(self):
        """Test setting value on nonexistent node"""
        with pytest.raises(ValueError, match="Node with ID 'nonexistent' does not exist"):
            self.graph.set_input_value("nonexistent", True)
    
    def test_set_input_value_non_input_node(self):
        """Test setting value on non-input node types"""
        self.graph.add_node(self.and_gate)
        
        with pytest.raises(ValueError, match="Node 'and1' is not an input node \\(type: and\\)"):
            self.graph.set_input_value("and1", True)
        
        # Test with output node
        output_node = Node.create("output1", NODE_REGISTRY.get_definition("output"), 300, 100, label="OUT")
        self.graph.add_node(output_node)
        
        with pytest.raises(ValueError, match="Node 'output1' is not an input node \\(type: output\\)"):
            self.graph.set_input_value("output1", False)
    
    def test_get_node_value_existing_node(self):
        """Test getting value from existing nodes"""
        # Test input node with no value set
        self.graph.add_node(self.input_node)
        assert self.graph.get_node_value("input1") is None
        
        # Set value and test again
        self.graph.set_input_value("input1", True)
        assert self.graph.get_node_value("input1") == True
        
        # Test with False value
        self.graph.set_input_value("input1", False)
        assert self.graph.get_node_value("input1") == False
        
        # Test other node types with values
        and_node = Node.create("and1", NODE_REGISTRY.get_definition("and"), 100, 100, value=True)
        self.graph.add_node(and_node)
        assert self.graph.get_node_value("and1") == True
    
    def test_get_node_value_nonexistent_node(self):
        """Test getting value from nonexistent node"""
        assert self.graph.get_node_value("nonexistent") is None
    
    def test_input_value_integration_with_slint_format(self):
        """Test that set input values appear in Slint format"""
        # Create multiple input nodes
        input_a = Node.create("input_a", NODE_REGISTRY.get_definition("input"), 0, 0, label="A")
        input_b = Node.create("input_b", NODE_REGISTRY.get_definition("input"), 0, 50, label="B")
        
        self.graph.add_node(input_a)
        self.graph.add_node(input_b)
        
        # Set different values
        self.graph.set_input_value("input_a", True)
        self.graph.set_input_value("input_b", False)
        
        # Check Slint format includes the values
        result = self.graph.to_slint_format()
        
        node_values = {node["id"]: node["value"] for node in result["nodes"]}
        assert node_values["input_a"] == True
        assert node_values["input_b"] == False
    
    def test_input_value_workflow_scenario(self):
        """Test complete workflow of setting and getting input values"""
        # Create a circuit with multiple inputs
        input_nodes = []
        for i, label in enumerate(["A", "B", "C"]):
            node = Node.create(f"input_{label.lower()}", NODE_REGISTRY.get_definition("input"), 
                             i * 60, 0, label=label)
            input_nodes.append(node)
            self.graph.add_node(node)
        
        # Initially all should be None
        for node in input_nodes:
            assert self.graph.get_node_value(node.id) is None
        
        # Set values sequentially
        self.graph.set_input_value("input_a", True)
        self.graph.set_input_value("input_b", False)
        self.graph.set_input_value("input_c", True)
        
        # Verify all values are set correctly
        assert self.graph.get_node_value("input_a") == True
        assert self.graph.get_node_value("input_b") == False
        assert self.graph.get_node_value("input_c") == True
        
        # Toggle some values
        self.graph.set_input_value("input_a", False)
        self.graph.set_input_value("input_b", True)
        
        # Verify changes
        assert self.graph.get_node_value("input_a") == False
        assert self.graph.get_node_value("input_b") == True
        assert self.graph.get_node_value("input_c") == True  # Unchanged
    
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
        assert "nets" in result
        assert "selected_nodes" in result
        assert isinstance(result["nodes"], list)
        assert isinstance(result["nets"], list)
        assert isinstance(result["selected_nodes"], list)
        assert len(result["nodes"]) == 0
        assert len(result["nets"]) == 0
        assert len(result["selected_nodes"]) == 0
    
    def test_to_slint_format_with_nodes(self):
        """Test Slint format conversion with nodes"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
        
        result = self.graph.to_slint_format()
        
        assert len(result["nodes"]) == 2
        assert len(result["nets"]) == 0
        
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
        
        # Check that value field is included (defaults to None)
        assert "value" in input_node_data
        assert input_node_data["value"] is None
    
    def test_to_slint_format_with_nets(self):
        """Test Slint format conversion with nets"""
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
        self.graph.add_net(self.net)
        
        result = self.graph.to_slint_format()
        
        assert len(result["nodes"]) == 2
        assert len(result["nets"]) == 1
        
        # Check net structure
        conn_data = result["nets"][0]
        assert conn_data["id"] == "conn1"
        # Check that net has segments
        assert "segments" in conn_data
        assert len(conn_data["segments"]) == 1  # Two-point net should have one segment
        segment = conn_data["segments"][0]
        assert segment["start_x"] == 99.0  # 50 + 46 + 3
        assert segment["start_y"] == 74.0  # 50 + 21 + 3
        assert segment["end_x"] == 199.0   # 200 + (-4) + 3
        assert segment["end_y"] == 94.0    # 80 + 11 + 3 (0.25 * 60 - 4 = 11)
    
    def test_to_slint_format_with_node_states(self):
        """Test Slint format conversion includes node state values"""
        # Create nodes with different state values
        input_def = NODE_REGISTRY.get_definition("input")
        and_def = NODE_REGISTRY.get_definition("and")
        
        input_true = Node.create("input_true", input_def, 0.0, 0.0, value=True)
        input_false = Node.create("input_false", input_def, 100.0, 0.0, value=False)
        and_undefined = Node.create("and_undef", and_def, 200.0, 0.0, value=None)
        
        self.graph.add_node(input_true)
        self.graph.add_node(input_false)
        self.graph.add_node(and_undefined)
        
        result = self.graph.to_slint_format()
        
        # Verify all nodes have value field
        assert len(result["nodes"]) == 3
        for node_data in result["nodes"]:
            assert "value" in node_data
        
        # Check specific values
        node_values = {node["id"]: node["value"] for node in result["nodes"]}
        assert node_values["input_true"] is True
        assert node_values["input_false"] is False
        assert node_values["and_undef"] is None
    
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
    
    def test_movement_with_net_updates(self):
        """Test that nets are properly updated when nodes move"""
        # Add a net between node1 and node2
        net = Net.create_two_point("conn1", "node1", "out", "node2", "in1")
        self.graph.add_net(net)
        
        # Get initial net position
        initial_result = self.graph.to_slint_format()
        initial_conn = initial_result["nets"][0]
        initial_segment = initial_conn["segments"][0]
        initial_start_x = initial_segment["start_x"]
        initial_start_y = initial_segment["start_y"]
        initial_end_x = initial_segment["end_x"]
        initial_end_y = initial_segment["end_y"]
        
        # Move node1
        self.graph.move_node("node1", 200, 200)
        
        # Check that net positions updated
        updated_result = self.graph.to_slint_format()
        updated_conn = updated_result["nets"][0]
        updated_segment = updated_conn["segments"][0]
        
        # Start position should have changed (node1 moved)
        assert updated_segment["start_x"] != initial_start_x
        assert updated_segment["start_y"] != initial_start_y
        
        # End position should be unchanged (node2 didn't move)
        assert updated_segment["end_x"] == initial_end_x
        assert updated_segment["end_y"] == initial_end_y


class TestCreateDemoGraph:
    """Test cases for the create_demo_graph function"""
    
    def test_demo_graph_creation(self):
        """Test that demo graph is created correctly"""
        graph = create_demo_graph()
        
        assert isinstance(graph, GraphData)
        assert len(graph.nodes) == 8  # 3 inputs + 3 gates + 2 outputs
        assert len(graph.nets) == 7  # 7 nets
    
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
    
    def test_demo_graph_nets(self):
        """Test that demo graph contains expected nets"""
        graph = create_demo_graph()
        
        expected_nets = [
            ("c1", "input_a", "out", "and_gate", "in1"),
            ("c2", "input_b", "out", "and_gate", "in2"),
            ("c3", "input_b", "out", "or_gate", "in1"),
            ("c4", "input_c", "out", "or_gate", "in2"),
            ("c5", "and_gate", "out", "not_gate", "in"),
            ("c6", "not_gate", "out", "output_a", "in"),
            ("c7", "or_gate", "out", "output_b", "in"),
        ]
        
        for conn_id, from_node, from_conn, to_node, to_conn in expected_nets:
            assert conn_id in graph.nets
            net = graph.nets[conn_id]
            # Check that net has exactly 2 connectors matching the expected connections
            assert len(net.connectors) == 2
            connector_pairs = {(conn.node_id, conn.connector_id) for conn in net.connectors}
            expected_pairs = {(from_node, from_conn), (to_node, to_conn)}
            assert connector_pairs == expected_pairs
    
    def test_demo_graph_slint_format(self):
        """Test that demo graph can be converted to Slint format"""
        graph = create_demo_graph()
        result = graph.to_slint_format()
        
        assert "nodes" in result
        assert "nets" in result
        assert len(result["nodes"]) == 8
        assert len(result["nets"]) == 7
        
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
    
    def test_handle_double_click_on_input_node(self):
        """Test double-click handling on input node toggles value"""
        # Initially None
        assert self.graph.get_node_value("input1") == None
        
        result = self.graph.handle_double_click(60.0, 60.0)  # Click on input1
        
        assert result == True  # UI should refresh
        assert self.graph.get_node_value("input1") == True  # Should be toggled to True
        assert self.graph.editing_node_id == None  # Should not start editing
    
    def test_handle_double_click_on_non_input_node(self):
        """Test double-click handling on non-input node starts label editing"""
        result = self.graph.handle_double_click(220.0, 90.0)  # Click on and1 gate
        
        assert result == True  # UI should refresh
        assert self.graph.editing_node_id == "and1"
        assert self.graph.editing_text == "Test AND"
    
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


class TestGraphDataNetSelection:
    """Test cases for net selection functionality"""
    
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
        
        # Add test nets
        self.net1 = Net.create_two_point("c1", "input1", "out", "and1", "in1")
        self.net2 = Net.create_two_point("c2", "and1", "out", "output1", "in")
        self.graph.add_net(self.net1)
        self.graph.add_net(self.net2)
    
    def test_net_selection_state_initialization(self):
        """Test that net selection state is properly initialized"""
        assert self.graph.selected_net_id is None
    
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
    
    def test_is_point_on_net(self):
        """Test point hit-testing on nets"""
        # This test uses actual net coordinates
        # net1: input1.out -> and1.in1
        # Should be approximately from (99,74) to (199,94) based on connector positions
        
        # Point near the line should hit
        assert self.graph.is_point_on_net(self.net1, 149, 84)  # Midpoint approximately
        
        # Point far from line should not hit
        assert not self.graph.is_point_on_net(self.net1, 100, 150)
        
        # Test with custom tolerance
        assert self.graph.is_point_on_net(self.net1, 149, 94, tolerance=15.0)  # Larger tolerance
        assert not self.graph.is_point_on_net(self.net1, 149, 94, tolerance=5.0)  # Smaller tolerance
    
    def test_get_net_at_position(self):
        """Test getting net at position"""
        # Should find net1 near its path
        result = self.graph.get_net_at_position(149, 84)
        assert result == "c1"
        
        # Should not find net at empty area
        result = self.graph.get_net_at_position(500, 500)
        assert result is None
        
        # Test priority - later nets should take priority if overlapping
        # For this we need overlapping nets, but our test setup doesn't have them
        # So this tests the basic case
        result = self.graph.get_net_at_position(314, 116)  # Near net2
        assert result == "c2"
    
    def test_select_net_valid(self):
        """Test selecting a valid net"""
        self.graph.select_net("c1")
        
        assert self.graph.selected_net_id == "c1"
        assert self.graph.selected_node_id is None  # Should deselect nodes
    
    def test_select_net_invalid(self):
        """Test selecting an invalid net"""
        with pytest.raises(ValueError, match="Net with ID 'nonexistent' does not exist"):
            self.graph.select_net("nonexistent")
    
    def test_select_net_deselects_node(self):
        """Test that selecting a net deselects any selected node"""
        # First select a node
        self.graph.select_node("input1")
        assert self.graph.selected_node_id == "input1"
        
        # Then select a net
        self.graph.select_net("c1")
        
        assert self.graph.selected_net_id == "c1"
        assert self.graph.selected_node_id is None
    
    def test_select_node_deselects_net(self):
        """Test that selecting a node deselects any selected net"""
        # First select a net
        self.graph.select_net("c1")
        assert self.graph.selected_net_id == "c1"
        
        # Then select a node
        self.graph.select_node("input1")
        
        assert self.graph.selected_node_id == "input1"
        assert self.graph.selected_net_id is None
    
    def test_deselect_net(self):
        """Test deselecting a net"""
        self.graph.select_net("c1")
        assert self.graph.selected_net_id == "c1"
        
        self.graph.deselect_net()
        
        assert self.graph.selected_net_id is None
    
    def test_deselect_net_when_none_selected(self):
        """Test deselecting when no net is selected"""
        self.graph.deselect_net()
        
        assert self.graph.selected_net_id is None
    
    def test_get_selected_net(self):
        """Test getting selected net"""
        assert self.graph.get_selected_net() is None
        
        self.graph.select_net("c1")
        assert self.graph.get_selected_net() == "c1"
    
    def test_handle_pointer_down_net_selection(self):
        """Test pointer down handling for net selection"""
        # Click near net1
        result = self.graph.handle_pointer_down(149, 84)
        
        assert result == True  # UI should refresh
        assert self.graph.selected_net_id == "c1"
        assert self.graph.selected_node_id is None
    
    def test_handle_pointer_down_node_priority(self):
        """Test that nodes have priority over nets"""
        # Click on a node position (should select node, not net)
        result = self.graph.handle_pointer_down(75, 75)  # On input1
        
        assert result == True  # UI should refresh
        assert self.graph.selected_node_id == "input1"
        assert self.graph.selected_net_id is None
    
    def test_handle_mouse_click_net_selection(self):
        """Test mouse click net selection"""
        # Click on net
        result = self.graph.handle_mouse_click(149, 84)
        
        assert result == True  # Selection changed
        assert self.graph.selected_net_id == "c1"
        assert self.graph.selected_node_id is None
    
    def test_handle_mouse_click_net_deselection(self):
        """Test clicking same net deselects it"""
        self.graph.select_net("c1")
        
        # Click on same net again
        result = self.graph.handle_mouse_click(149, 84)
        
        assert result == True  # Selection changed
        assert self.graph.selected_net_id is None
    
    def test_handle_mouse_click_different_net(self):
        """Test clicking different net switches selection"""
        self.graph.select_net("c1")
        
        # Click on different net
        result = self.graph.handle_mouse_click(314, 116)  # Near net2
        
        assert result == True  # Selection changed
        assert self.graph.selected_net_id == "c2"
    
    def test_handle_mouse_click_empty_area_deselects_net(self):
        """Test clicking empty area deselects net"""
        self.graph.select_net("c1")
        
        # Click on empty area
        result = self.graph.handle_mouse_click(500, 500)
        
        assert result == True  # Selection changed
        assert self.graph.selected_net_id is None
    
    def test_to_slint_format_includes_selected_nets_none(self):
        """Test Slint format includes empty selected nets"""
        result = self.graph.to_slint_format()
        
        assert "selected_nets" in result
        assert result["selected_nets"] == []
    
    def test_to_slint_format_includes_selected_nets_with_selection(self):
        """Test Slint format includes selected net"""
        self.graph.select_net("c1")
        
        result = self.graph.to_slint_format()
        
        assert "selected_nets" in result
        assert result["selected_nets"] == ["c1"]
    
    def test_net_selection_integration_scenario(self):
        """Test a complete net selection scenario"""
        # Start with no selection
        assert self.graph.selected_net_id is None
        assert self.graph.selected_node_id is None
        
        # Select a net
        self.graph.select_net("c1")
        assert self.graph.selected_net_id == "c1"
        
        # Select a different net
        self.graph.select_net("c2")
        assert self.graph.selected_net_id == "c2"
        
        # Select a node (should deselect net)
        self.graph.select_node("input1")
        assert self.graph.selected_node_id == "input1"
        assert self.graph.selected_net_id is None
        
        # Select net again (should deselect node)
        self.graph.select_net("c1")
        assert self.graph.selected_net_id == "c1"
        assert self.graph.selected_node_id is None
    
    def test_net_hit_testing_edge_cases(self):
        """Test net hit-testing edge cases"""
        # Test with zero-length line (degenerate case)
        # This shouldn't happen in practice but tests the math
        fake_net = Net.create_two_point("test", "input1", "out", "input1", "out")
        
        # Point at same location should hit
        assert self.graph.is_point_on_net(fake_net, 99, 74, tolerance=1.0)
        
        # Point away should not hit
        assert not self.graph.is_point_on_net(fake_net, 200, 200, tolerance=1.0)


class TestGraphDataDeletion:
    """Test cases for node and net deletion functionality"""
    
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
        
        # Add test nets
        self.net1 = Net.create_two_point("c1", "input1", "out", "and1", "in1")
        self.net2 = Net.create_two_point("c2", "and1", "out", "output1", "in")
        self.net3 = Net.create_two_point("c3", "input1", "out", "or1", "in1")
        self.graph.add_net(self.net1)
        self.graph.add_net(self.net2)
        self.graph.add_net(self.net3)
    
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
    
    def test_delete_node_cascade_nets(self):
        """Test that deleting a node also deletes its nets"""
        # Verify initial state
        assert len(self.graph.nets) == 3
        assert "c1" in self.graph.nets  # input1 -> and1
        assert "c2" in self.graph.nets  # and1 -> output1
        assert "c3" in self.graph.nets  # input1 -> or1
        
        # Delete and1 node (should cascade delete c1 and c2)
        result = self.graph.delete_node("and1")
        
        assert result == True
        assert len(self.graph.nets) == 1  # Only c3 should remain
        assert "c1" not in self.graph.nets
        assert "c2" not in self.graph.nets
        assert "c3" in self.graph.nets  # Unrelated net preserved
    
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
    
    def test_delete_net_valid(self):
        """Test deleting a valid net"""
        # Verify net exists
        assert "c1" in self.graph.nets
        assert len(self.graph.nets) == 3
        
        # Delete the net
        result = self.graph.delete_net("c1")
        
        # Verify net is deleted
        assert result == True
        assert "c1" not in self.graph.nets
        assert len(self.graph.nets) == 2
        
        # Verify nodes are unaffected
        assert len(self.graph.nodes) == 4
    
    def test_delete_net_invalid(self):
        """Test deleting a non-existent net"""
        initial_count = len(self.graph.nets)
        
        result = self.graph.delete_net("nonexistent")
        
        assert result == False
        assert len(self.graph.nets) == initial_count
    
    def test_delete_net_selected_state_cleanup(self):
        """Test that deleting a selected net clears selection"""
        # Select the net first
        self.graph.select_net("c1")
        assert self.graph.selected_net_id == "c1"
        
        # Delete the net
        result = self.graph.delete_net("c1")
        
        assert result == True
        assert self.graph.selected_net_id is None
    
    def test_delete_selected_node(self):
        """Test deleting selected node via delete_selected"""
        # Select a node
        self.graph.select_node("and1")
        initial_node_count = len(self.graph.nodes)
        initial_net_count = len(self.graph.nets)
        
        # Delete selected item
        result = self.graph.delete_selected()
        
        assert result == True
        assert len(self.graph.nodes) == initial_node_count - 1
        assert len(self.graph.nets) < initial_net_count  # Cascade deletion
        assert self.graph.selected_node_id is None
    
    def test_delete_selected_net(self):
        """Test deleting selected net via delete_selected"""
        # Select a net
        self.graph.select_net("c1")
        initial_net_count = len(self.graph.nets)
        initial_node_count = len(self.graph.nodes)
        
        # Delete selected item
        result = self.graph.delete_selected()
        
        assert result == True
        assert len(self.graph.nets) == initial_net_count - 1
        assert len(self.graph.nodes) == initial_node_count  # Nodes unaffected
        assert self.graph.selected_net_id is None
    
    def test_delete_selected_nothing(self):
        """Test delete_selected when nothing is selected"""
        # Ensure nothing is selected
        assert self.graph.selected_node_id is None
        assert self.graph.selected_net_id is None
        
        initial_node_count = len(self.graph.nodes)
        initial_net_count = len(self.graph.nets)
        
        # Attempt to delete
        result = self.graph.delete_selected()
        
        assert result == False
        assert len(self.graph.nodes) == initial_node_count
        assert len(self.graph.nets) == initial_net_count
    
    def test_cascade_deletion_complex_graph(self):
        """Test cascade deletion with node having multiple nets"""
        # input1 has nets c1 and c3, so deleting it should remove both
        assert "c1" in self.graph.nets  # input1 -> and1
        assert "c3" in self.graph.nets  # input1 -> or1
        assert len(self.graph.nets) == 3
        
        # Delete input1
        result = self.graph.delete_node("input1")
        
        assert result == True
        assert "input1" not in self.graph.nodes
        assert "c1" not in self.graph.nets  # Cascade deleted
        assert "c3" not in self.graph.nets  # Cascade deleted
        assert "c2" in self.graph.nets     # Unrelated net preserved
        assert len(self.graph.nets) == 1
    
    def test_delete_integration_scenario(self):
        """Test a complete deletion workflow scenario"""
        # Start with full graph
        assert len(self.graph.nodes) == 4
        assert len(self.graph.nets) == 3
        
        # Select and delete a net
        self.graph.select_net("c2")
        result1 = self.graph.delete_selected()
        assert result1 == True
        assert len(self.graph.nets) == 2
        assert self.graph.selected_net_id is None
        
        # Select and delete a node (with cascade)
        self.graph.select_node("input1")
        result2 = self.graph.delete_selected()
        assert result2 == True
        assert len(self.graph.nodes) == 3
        assert len(self.graph.nets) == 0  # c1 and c3 cascade deleted
        assert self.graph.selected_node_id is None
        
        # Try to delete when nothing selected
        result3 = self.graph.delete_selected()
        assert result3 == False
        
        # Graph should be stable
        assert len(self.graph.nodes) == 3
        assert len(self.graph.nets) == 0
    
    def test_delete_preserves_other_selections(self):
        """Test that deletion of unrelated items preserves current selection"""
        # Test 1: Select node1, but delete node2 (unrelated)
        self.graph.select_node("input1")
        result = self.graph.delete_node("or1")  # Delete unrelated node
        
        assert result == True
        assert self.graph.selected_node_id == "input1"  # Selection preserved
        
        # Test 2: Select one net, delete a different one
        # After deleting or1, we still have c1 (input1->and1) and c2 (and1->output1)
        # c3 (input1->or1) was cascade deleted with or1
        self.graph.select_net("c1")
        result = self.graph.delete_net("c2")  # Delete different net
        
        assert result == True
        assert self.graph.selected_net_id == "c1"  # Selection preserved


class TestGraphDataNetCreation:
    """Test cases for net creation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
        
        # Create test nodes with different types
        self.input_node = Node.create("input1", NODE_REGISTRY.get_definition("input"), 50, 50)
        self.and_gate = Node.create("and1", NODE_REGISTRY.get_definition("and"), 200, 80)
        self.output_node = Node.create("output1", NODE_REGISTRY.get_definition("output"), 350, 85)
        
        self.graph.add_node(self.input_node)
        self.graph.add_node(self.and_gate)
        self.graph.add_node(self.output_node)
    
    def test_get_connector_at_position(self):
        """Test connector hit testing"""
        # Calculate absolute position for input node "out" connector
        abs_x, abs_y = self.graph.get_connector_absolute_position("input1", "out")
        
        # Test hitting the connector
        result = self.graph.get_connector_at_position(abs_x, abs_y)
        assert result == ("input1", "out")
        
        # Test hitting near the connector (within tolerance)
        result = self.graph.get_connector_at_position(abs_x + 5, abs_y + 5)
        assert result == ("input1", "out")
        
        # Test missing the connector (outside tolerance)
        result = self.graph.get_connector_at_position(abs_x + 20, abs_y + 20)
        assert result is None
        
        # Test hitting empty space
        result = self.graph.get_connector_at_position(10, 10)
        assert result is None
    
    def test_get_connector_at_position_multiple_connectors(self):
        """Test connector hit testing with multiple connectors on same node"""
        # AND gate has three connectors: in1, in2, out
        and_node = self.graph.nodes["and1"]
        
        # Test hitting each connector
        for connector in and_node.connectors:
            abs_x, abs_y = self.graph.get_connector_absolute_position("and1", connector.id)
            result = self.graph.get_connector_at_position(abs_x, abs_y)
            assert result == ("and1", connector.id)
    
    def test_get_connector_at_position_overlapping_priority(self):
        """Test connector priority when nodes overlap"""
        # Create overlapping nodes
        node1 = Node.create("node1", NODE_REGISTRY.get_definition("input"), 100, 100)
        node2 = Node.create("node2", NODE_REGISTRY.get_definition("input"), 110, 110)
        
        self.graph.add_node(node1)
        self.graph.add_node(node2)
        
        # Get overlapping position where both connectors might be hit
        # Since we iterate in reverse order, later-added nodes take priority
        abs_x, abs_y = self.graph.get_connector_absolute_position("node2", "out")
        result = self.graph.get_connector_at_position(abs_x, abs_y)
        assert result == ("node2", "out")
    
    def test_start_net_creation_valid(self):
        """Test starting net creation from valid connector"""
        result = self.graph.start_net_creation("input1", "out")
        
        assert result == True
        assert self.graph.creating_net == True
        assert self.graph.net_start_node_id == "input1"
        assert self.graph.net_start_connector_id == "out"
        
        # Pending end position should be initialized to connector position
        start_x, start_y = self.graph.get_connector_absolute_position("input1", "out")
        assert self.graph.pending_net_end_x == start_x
        assert self.graph.pending_net_end_y == start_y
    
    def test_start_net_creation_invalid_node(self):
        """Test starting net creation from non-existent node"""
        result = self.graph.start_net_creation("nonexistent", "out")
        
        assert result == False
        assert self.graph.creating_net == False
    
    def test_start_net_creation_invalid_connector(self):
        """Test starting net creation from non-existent connector"""
        result = self.graph.start_net_creation("input1", "nonexistent")
        
        assert result == False
        assert self.graph.creating_net == False
    
    def test_start_net_creation_clears_existing_state(self):
        """Test that starting net creation clears other states"""
        # Set up existing states
        self.graph.select_node("input1")
        self.graph.start_label_edit("input1")
        
        # Start net creation
        result = self.graph.start_net_creation("and1", "out")
        
        assert result == True
        assert self.graph.creating_net == True
        assert self.graph.selected_node_id is None  # Selection cleared
        assert self.graph.editing_node_id is None   # Editing cleared
    
    def test_update_pending_net(self):
        """Test updating pending net position"""
        # Start net creation
        self.graph.start_net_creation("input1", "out")
        
        # Update pending position
        result = self.graph.update_pending_net(300.0, 250.0)
        
        assert result == True
        assert self.graph.pending_net_end_x == 300.0
        assert self.graph.pending_net_end_y == 250.0
    
    def test_update_pending_net_not_creating(self):
        """Test updating pending net when not in creation mode"""
        result = self.graph.update_pending_net(300.0, 250.0)
        
        assert result == False
    
    def test_cancel_net_creation(self):
        """Test cancelling net creation"""
        # Start net creation
        self.graph.start_net_creation("input1", "out")
        assert self.graph.creating_net == True
        
        # Cancel creation
        result = self.graph.cancel_net_creation()
        
        assert result == True
        assert self.graph.creating_net == False
        assert self.graph.net_start_node_id is None
        assert self.graph.net_start_connector_id is None
        assert self.graph.pending_net_end_x == 0.0
        assert self.graph.pending_net_end_y == 0.0
    
    def test_cancel_net_creation_not_creating(self):
        """Test cancelling when not in creation mode"""
        result = self.graph.cancel_net_creation()
        
        assert result == False
    
    def test_complete_net_creation_valid(self):
        """Test completing valid net creation"""
        # Start net from input to and gate
        self.graph.start_net_creation("input1", "out")
        
        # Complete net to and gate input
        result = self.graph.complete_net_creation("and1", "in1")
        
        assert result == True
        assert self.graph.creating_net == False
        assert len(self.graph.nets) == 1
        
        # Check the created net
        net = list(self.graph.nets.values())[0]
        assert len(net.connectors) == 2
        connector_pairs = {(conn.node_id, conn.connector_id) for conn in net.connectors}
        expected_pairs = {("input1", "out"), ("and1", "in1")}
        assert connector_pairs == expected_pairs
    
    def test_complete_net_creation_not_creating(self):
        """Test completing when not in creation mode"""
        result = self.graph.complete_net_creation("and1", "in1")
        
        assert result == False
    
    def test_complete_net_creation_invalid_target(self):
        """Test completing with invalid target connector"""
        self.graph.start_net_creation("input1", "out")
        
        # Try to complete with non-existent node
        result = self.graph.complete_net_creation("nonexistent", "in1")
        
        assert result == True  # Returns True because it cancels creation
        assert self.graph.creating_net == False
        assert len(self.graph.nets) == 0
    
    def test_can_create_net_valid_output_to_input(self):
        """Test valid net from output to input"""
        result = self.graph.can_create_net("input1", "out", "and1", "in1")
        assert result == True
    
    def test_can_create_net_valid_input_to_output(self):
        """Test valid net from input to output (reverse direction)"""
        result = self.graph.can_create_net("and1", "in1", "input1", "out")
        assert result == True
    
    def test_can_create_net_same_node(self):
        """Test invalid net to same node"""
        result = self.graph.can_create_net("input1", "out", "input1", "out")
        assert result == False
    
    def test_can_create_net_output_to_output(self):
        """Test that output to output nets are now allowed (directionality removed)"""
        result = self.graph.can_create_net("input1", "out", "and1", "out")
        assert result == True  # Multi-connector nets allow any connector combinations
    
    def test_can_create_net_input_to_input(self):
        """Test invalid net from input to input"""
        result = self.graph.can_create_net("and1", "in1", "and1", "in2")
        assert result == False
    
    def test_can_create_net_duplicate(self):
        """Test preventing duplicate nets"""
        # Create initial net
        net = Net.create_two_point("c1", "input1", "out", "and1", "in1")
        self.graph.add_net(net)
        
        # Try to create same net again
        result = self.graph.can_create_net("input1", "out", "and1", "in1")
        assert result == False
        
        # Try reverse direction
        result = self.graph.can_create_net("and1", "in1", "input1", "out")
        assert result == False
    
    def test_can_create_net_input_already_connected(self):
        """Test preventing multiple nets to same input"""
        # Create initial net to and gate input
        net = Net.create_two_point("c1", "input1", "out", "and1", "in1")
        self.graph.add_net(net)
        
        # Try to connect another output to same input
        input2 = Node.create("input2", NODE_REGISTRY.get_definition("input"), 50, 150)
        self.graph.add_node(input2)
        
        result = self.graph.can_create_net("input2", "out", "and1", "in1")
        assert result == False
    
    def test_can_create_net_nonexistent_nodes(self):
        """Test net validation with non-existent nodes"""
        result = self.graph.can_create_net("nonexistent1", "out", "and1", "in1")
        assert result == False
        
        result = self.graph.can_create_net("input1", "out", "nonexistent2", "in1")
        assert result == False
    
    def test_can_create_net_nonexistent_connectors(self):
        """Test net validation with non-existent connectors"""
        result = self.graph.can_create_net("input1", "nonexistent", "and1", "in1")
        assert result == False
        
        result = self.graph.can_create_net("input1", "out", "and1", "nonexistent")
        assert result == False
    
    def test_net_creation_workflow(self):
        """Test complete net creation workflow"""
        initial_nets = len(self.graph.nets)
        
        # Start net creation
        result1 = self.graph.start_net_creation("input1", "out")
        assert result1 == True
        assert self.graph.creating_net == True
        
        # Update pending net a few times (simulating mouse movement)
        result2 = self.graph.update_pending_net(150.0, 90.0)
        assert result2 == True
        
        result3 = self.graph.update_pending_net(180.0, 95.0)
        assert result3 == True
        
        # Complete net
        result4 = self.graph.complete_net_creation("and1", "in1")
        assert result4 == True
        assert self.graph.creating_net == False
        assert len(self.graph.nets) == initial_nets + 1
    
    def test_net_creation_cancel_workflow(self):
        """Test net creation with cancellation"""
        initial_nets = len(self.graph.nets)
        
        # Start net creation
        self.graph.start_net_creation("input1", "out")
        
        # Update pending net
        self.graph.update_pending_net(200.0, 100.0)
        
        # Cancel instead of completing
        result = self.graph.cancel_net_creation()
        
        assert result == True
        assert self.graph.creating_net == False
        assert len(self.graph.nets) == initial_nets  # No new nets
    
    def test_to_slint_format_with_pending_net(self):
        """Test Slint format includes pending net data"""
        # Start net creation
        self.graph.start_net_creation("input1", "out")
        self.graph.update_pending_net(300.0, 250.0)
        
        result = self.graph.to_slint_format()
        
        assert "creating_net" in result
        assert result["creating_net"] == True
        assert "pending_start_x" in result
        assert "pending_start_y" in result
        assert "pending_end_x" in result
        assert "pending_end_y" in result
        assert result["pending_end_x"] == 300.0
        assert result["pending_end_y"] == 250.0
    
    def test_to_slint_format_no_pending_net(self):
        """Test Slint format when not creating net"""
        result = self.graph.to_slint_format()
        
        assert "creating_net" in result
        assert result["creating_net"] == False
        assert result["pending_start_x"] == 0.0
        assert result["pending_start_y"] == 0.0
        assert result["pending_end_x"] == 0.0
        assert result["pending_end_y"] == 0.0
    
    def test_pointer_events_net_creation(self):
        """Test pointer event handling during net creation"""
        # Test starting net by clicking on connector
        input_connector_x, input_connector_y = self.graph.get_connector_absolute_position("input1", "out")
        
        result1 = self.graph.handle_pointer_down(input_connector_x, input_connector_y)
        assert result1 == True
        assert self.graph.creating_net == True
        
        # Test updating pending net during mouse move
        result2 = self.graph.handle_pointer_move(200.0, 100.0)
        assert result2 == True
        assert self.graph.pending_net_end_x == 200.0
        assert self.graph.pending_net_end_y == 100.0
        
        # Test completing net by clicking on target connector
        and_connector_x, and_connector_y = self.graph.get_connector_absolute_position("and1", "in1")
        result3 = self.graph.handle_pointer_down(and_connector_x, and_connector_y)
        assert result3 == True
        assert self.graph.creating_net == False
        assert len(self.graph.nets) == 1
    
    def test_pointer_events_cancel_net_creation(self):
        """Test cancelling net creation by clicking empty area"""
        # Start net creation
        input_connector_x, input_connector_y = self.graph.get_connector_absolute_position("input1", "out")
        self.graph.handle_pointer_down(input_connector_x, input_connector_y)
        assert self.graph.creating_net == True
        
        # Click on empty area to cancel
        result = self.graph.handle_pointer_down(500.0, 500.0)  # Empty area
        assert result == True
        assert self.graph.creating_net == False
        assert len(self.graph.nets) == 0


class TestGraphDataToolbox:
    """Test cases for toolbox functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
    
    def test_select_toolbox_node_type_valid(self):
        """Test selecting a valid node type from toolbox"""
        result = self.graph.select_toolbox_node_type("input")
        
        assert result == True
        assert self.graph.selected_node_type == "input"
        assert self.graph.toolbox_creation_mode == True
    
    def test_select_toolbox_node_type_invalid(self):
        """Test selecting an invalid node type from toolbox"""
        result = self.graph.select_toolbox_node_type("nonexistent")
        
        assert result == False
        assert self.graph.selected_node_type is None
        assert self.graph.toolbox_creation_mode == False
    
    def test_select_toolbox_clears_other_states(self):
        """Test that selecting toolbox item clears other interaction states"""
        # Set up existing states
        input_node = Node.create("input1", NODE_REGISTRY.get_definition("input"), 50, 50)
        self.graph.add_node(input_node)
        
        self.graph.select_node("input1")
        self.graph.start_label_edit("input1")
        self.graph.start_net_creation("input1", "out")
        
        # Select toolbox item
        result = self.graph.select_toolbox_node_type("and")
        
        assert result == True
        assert self.graph.selected_node_type == "and"
        assert self.graph.toolbox_creation_mode == True
        assert self.graph.selected_node_id is None          # Node selection cleared
        assert self.graph.editing_node_id is None           # Label editing cleared
        assert self.graph.creating_net == False      # Net creation cleared
    
    def test_deselect_toolbox_node_type(self):
        """Test deselecting toolbox node type"""
        # First select a node type
        self.graph.select_toolbox_node_type("input")
        assert self.graph.selected_node_type == "input"
        
        # Then deselect
        result = self.graph.deselect_toolbox_node_type()
        
        assert result == True
        assert self.graph.selected_node_type is None
        assert self.graph.toolbox_creation_mode == False
    
    def test_deselect_toolbox_when_none_selected(self):
        """Test deselecting when no toolbox item is selected"""
        result = self.graph.deselect_toolbox_node_type()
        
        assert result == False
        assert self.graph.selected_node_type is None
        assert self.graph.toolbox_creation_mode == False
    
    def test_create_node_at_position_valid(self):
        """Test creating a node at specified position"""
        initial_node_count = len(self.graph.nodes)
        
        result = self.graph.create_node_at_position("input", 100.0, 200.0)
        
        assert result == True
        assert len(self.graph.nodes) == initial_node_count + 1
        
        # Find the created node
        created_node = None
        for node in self.graph.nodes.values():
            if node.node_type == "input" and node.x == 100.0 and node.y == 200.0:
                created_node = node
                break
        
        assert created_node is not None
        assert created_node.node_type == "input"
        assert created_node.x == 100.0
        assert created_node.y == 200.0
    
    def test_create_node_at_position_invalid_type(self):
        """Test creating a node with invalid type"""
        initial_node_count = len(self.graph.nodes)
        
        result = self.graph.create_node_at_position("nonexistent", 100.0, 200.0)
        
        assert result == False
        assert len(self.graph.nodes) == initial_node_count
    
    def test_create_node_generates_unique_id(self):
        """Test that created nodes get unique IDs"""
        # Create multiple nodes of same type
        self.graph.create_node_at_position("input", 50.0, 50.0)
        self.graph.create_node_at_position("input", 100.0, 100.0)
        self.graph.create_node_at_position("input", 150.0, 150.0)
        
        # Check that all nodes have unique IDs
        node_ids = list(self.graph.nodes.keys())
        assert len(node_ids) == len(set(node_ids))  # No duplicates
        
        # Check that all are input nodes
        input_nodes = [n for n in self.graph.nodes.values() if n.node_type == "input"]
        assert len(input_nodes) == 3
    
    def test_create_node_resets_toolbox_state(self):
        """Test that creating a node resets toolbox to idle state"""
        # Select a toolbox item
        self.graph.select_toolbox_node_type("input")
        assert self.graph.toolbox_creation_mode == True
        
        # Create a node
        result = self.graph.create_node_at_position("input", 100.0, 200.0)
        
        assert result == True
        assert self.graph.selected_node_type is None
        assert self.graph.toolbox_creation_mode == False
    
    def test_get_toolbox_data_empty(self):
        """Test getting toolbox data with no selection"""
        toolbox_data = self.graph.get_toolbox_data()
        
        # Should return all available node types
        assert len(toolbox_data) == len(NODE_REGISTRY.list_definitions())
        
        # Check structure of returned data
        for item in toolbox_data:
            assert "node_type" in item
            assert "label" in item
            assert "color" in item
            assert "is_selected" in item
            assert item["is_selected"] == False  # Nothing selected
    
    def test_get_toolbox_data_with_selection(self):
        """Test getting toolbox data with a selection"""
        # Select a node type
        self.graph.select_toolbox_node_type("and")
        
        toolbox_data = self.graph.get_toolbox_data()
        
        # Check that the selected item is marked as selected
        and_item = next(item for item in toolbox_data if item["node_type"] == "and")
        assert and_item["is_selected"] == True
        
        # Check that other items are not selected
        other_items = [item for item in toolbox_data if item["node_type"] != "and"]
        for item in other_items:
            assert item["is_selected"] == False
    
    def test_get_toolbox_data_structure(self):
        """Test the structure of toolbox data"""
        toolbox_data = self.graph.get_toolbox_data()
        
        # Should include all standard node types
        node_types = [item["node_type"] for item in toolbox_data]
        assert "input" in node_types
        assert "output" in node_types
        assert "and" in node_types
        assert "or" in node_types
        assert "not" in node_types
        
        # Check data structure for input node
        input_item = next(item for item in toolbox_data if item["node_type"] == "input")
        assert input_item["label"] == "INPUT"
        assert input_item["color"] == "rgb(144, 238, 144)"
        assert isinstance(input_item["is_selected"], bool)
    
    def test_to_slint_format_includes_toolbox_data(self):
        """Test that Slint format includes toolbox data"""
        # Select a toolbox item
        self.graph.select_toolbox_node_type("or")
        
        result = self.graph.to_slint_format()
        
        assert "toolbox_items" in result
        assert "toolbox_creation_mode" in result
        assert result["toolbox_creation_mode"] == True
        
        # Check toolbox items structure
        toolbox_items = result["toolbox_items"]
        assert len(toolbox_items) > 0
        
        # Check that OR is selected
        or_item = next(item for item in toolbox_items if item["node_type"] == "or")
        assert or_item["is_selected"] == True
    
    def test_to_slint_format_no_toolbox_selection(self):
        """Test Slint format when no toolbox item is selected"""
        result = self.graph.to_slint_format()
        
        assert "toolbox_items" in result
        assert "toolbox_creation_mode" in result
        assert result["toolbox_creation_mode"] == False
        
        # Check that no items are selected
        toolbox_items = result["toolbox_items"]
        for item in toolbox_items:
            assert item["is_selected"] == False
    
    def test_toolbox_workflow_complete(self):
        """Test complete toolbox workflow"""
        initial_node_count = len(self.graph.nodes)
        
        # 1. Select node type from toolbox
        result1 = self.graph.select_toolbox_node_type("and")
        assert result1 == True
        assert self.graph.toolbox_creation_mode == True
        
        # 2. Create node at position
        result2 = self.graph.create_node_at_position("and", 200.0, 150.0)
        assert result2 == True
        assert len(self.graph.nodes) == initial_node_count + 1
        assert self.graph.toolbox_creation_mode == False  # Back to idle
        
        # 3. Verify created node
        and_nodes = [n for n in self.graph.nodes.values() if n.node_type == "and"]
        assert len(and_nodes) == 1
        
        created_node = and_nodes[0]
        assert created_node.x == 200.0
        assert created_node.y == 150.0
        assert created_node.label == "AND"
    
    def test_toolbox_toggle_selection(self):
        """Test toggling toolbox selection (select same item twice)"""
        # First selection
        result1 = self.graph.select_toolbox_node_type("not")
        assert result1 == True
        assert self.graph.selected_node_type == "not"
        
        # Second selection of same type should work (selection logic is in main.py)
        result2 = self.graph.select_toolbox_node_type("not")
        assert result2 == True
        assert self.graph.selected_node_type == "not"  # Still selected
    
    def test_toolbox_switch_selection(self):
        """Test switching between different toolbox selections"""
        # Select first type
        self.graph.select_toolbox_node_type("input")
        assert self.graph.selected_node_type == "input"
        
        # Switch to different type
        result = self.graph.select_toolbox_node_type("output")
        assert result == True
        assert self.graph.selected_node_type == "output"
        assert self.graph.toolbox_creation_mode == True
    
    def test_node_creation_with_existing_nodes(self):
        """Test creating nodes when graph already has nodes"""
        # Add existing node
        existing_node = Node.create("existing", NODE_REGISTRY.get_definition("input"), 0, 0)
        self.graph.add_node(existing_node)
        
        initial_count = len(self.graph.nodes)
        
        # Create new node
        result = self.graph.create_node_at_position("and", 300.0, 300.0)
        
        assert result == True
        assert len(self.graph.nodes) == initial_count + 1
        
        # Verify both nodes exist
        node_types = [n.node_type for n in self.graph.nodes.values()]
        assert "input" in node_types
        assert "and" in node_types


class TestCircuitSimulation:
    """Test cases for circuit simulation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
        
        # Create a simple test circuit: Input A -> NOT gate -> Output
        self.input_a = Node.create("input_a", NODE_REGISTRY.get_definition("input"), 50, 50, label="A")
        self.not_gate = Node.create("not_gate", NODE_REGISTRY.get_definition("not"), 200, 50, label="NOT")
        self.output = Node.create("output", NODE_REGISTRY.get_definition("output"), 350, 50, label="OUT")
        
        self.graph.add_node(self.input_a)
        self.graph.add_node(self.not_gate)
        self.graph.add_node(self.output)
        
        # Connect: A -> NOT -> OUT
        self.conn1 = Net.create_two_point("c1", "input_a", "out", "not_gate", "in")
        self.conn2 = Net.create_two_point("c2", "not_gate", "out", "output", "in")
        
        self.graph.add_net(self.conn1)
        self.graph.add_net(self.conn2)
    
    def test_simulate_simple_not_gate_true_input(self):
        """Test simulation of NOT gate with true input"""
        # Set input to True
        self.graph.set_input_value("input_a", True)
        
        # Run simulation
        result = self.graph.simulate()
        
        assert result == True
        assert self.graph.get_node_value("input_a") == True
        assert self.graph.get_node_value("not_gate") == False  # NOT True = False
        assert self.graph.get_node_value("output") == False   # Pass through from NOT gate
    
    def test_simulate_simple_not_gate_false_input(self):
        """Test simulation of NOT gate with false input"""
        # Set input to False
        self.graph.set_input_value("input_a", False)
        
        # Run simulation
        result = self.graph.simulate()
        
        assert result == True
        assert self.graph.get_node_value("input_a") == False
        assert self.graph.get_node_value("not_gate") == True   # NOT False = True
        assert self.graph.get_node_value("output") == True    # Pass through from NOT gate
    
    def test_simulate_without_input_values_raises_error(self):
        """Test that simulation fails if input values are not set"""
        # Don't set any input values
        
        with pytest.raises(ValueError, match="Input nodes must have values set before simulation"):
            self.graph.simulate()
    
    def test_simulate_complex_circuit(self):
        """Test simulation of more complex circuit with AND and OR gates"""
        # Create a more complex circuit: (A AND B) OR C -> Output
        graph = GraphData()
        
        # Create nodes
        input_a = Node.create("a", NODE_REGISTRY.get_definition("input"), 50, 50, label="A")
        input_b = Node.create("b", NODE_REGISTRY.get_definition("input"), 50, 150, label="B")
        input_c = Node.create("c", NODE_REGISTRY.get_definition("input"), 50, 250, label="C")
        and_gate = Node.create("and", NODE_REGISTRY.get_definition("and"), 200, 100, label="AND")
        or_gate = Node.create("or", NODE_REGISTRY.get_definition("or"), 350, 175, label="OR")
        output = Node.create("out", NODE_REGISTRY.get_definition("output"), 500, 175, label="OUT")
        
        graph.add_node(input_a)
        graph.add_node(input_b)
        graph.add_node(input_c)
        graph.add_node(and_gate)
        graph.add_node(or_gate)
        graph.add_node(output)
        
        # Create nets: A -> AND, B -> AND, AND -> OR, C -> OR, OR -> OUT
        nets = [
            Net.create_two_point("c1", "a", "out", "and", "in1"),
            Net.create_two_point("c2", "b", "out", "and", "in2"),
            Net.create_two_point("c3", "and", "out", "or", "in1"),
            Net.create_two_point("c4", "c", "out", "or", "in2"),
            Net.create_two_point("c5", "or", "out", "out", "in")
        ]
        
        for conn in nets:
            graph.add_net(conn)
        
        # Test case: A=True, B=False, C=False -> (True AND False) OR False = False OR False = False
        graph.set_input_value("a", True)
        graph.set_input_value("b", False)
        graph.set_input_value("c", False)
        
        result = graph.simulate()
        
        assert result == True
        assert graph.get_node_value("a") == True
        assert graph.get_node_value("b") == False
        assert graph.get_node_value("c") == False
        assert graph.get_node_value("and") == False   # True AND False = False
        assert graph.get_node_value("or") == False    # False OR False = False
        assert graph.get_node_value("out") == False   # Pass through
        
        # Test case: A=True, B=True, C=False -> (True AND True) OR False = True OR False = True
        graph.set_input_value("a", True)
        graph.set_input_value("b", True)
        graph.set_input_value("c", False)
        
        result = graph.simulate()
        
        assert result == True
        assert graph.get_node_value("and") == True    # True AND True = True
        assert graph.get_node_value("or") == True     # True OR False = True
        assert graph.get_node_value("out") == True    # Pass through
        
        # Test case: A=False, B=False, C=True -> (False AND False) OR True = False OR True = True
        graph.set_input_value("a", False)
        graph.set_input_value("b", False)
        graph.set_input_value("c", True)
        
        result = graph.simulate()
        
        assert result == True
        assert graph.get_node_value("and") == False   # False AND False = False
        assert graph.get_node_value("or") == True     # False OR True = True
        assert graph.get_node_value("out") == True    # Pass through
    
    def test_topological_sorting_simple_chain(self):
        """Test topological sorting for a simple chain of nodes"""
        # Using the simple setup: Input -> NOT -> Output
        order = self.graph._get_evaluation_order()
        
        # Input should come first, then NOT, then Output
        input_index = order.index("input_a")
        not_index = order.index("not_gate")
        output_index = order.index("output")
        
        assert input_index < not_index
        assert not_index < output_index
    
    def test_topological_sorting_complex_dependencies(self):
        """Test topological sorting for complex dependency graph"""
        # Create circuit with complex dependencies
        graph = GraphData()
        
        # Create nodes: A, B -> AND -> NOT -> C, D -> OR -> Output
        nodes = [
            Node.create("a", NODE_REGISTRY.get_definition("input"), 0, 0),
            Node.create("b", NODE_REGISTRY.get_definition("input"), 0, 50),
            Node.create("c", NODE_REGISTRY.get_definition("input"), 0, 200),
            Node.create("d", NODE_REGISTRY.get_definition("input"), 0, 250),
            Node.create("and", NODE_REGISTRY.get_definition("and"), 100, 25),
            Node.create("not", NODE_REGISTRY.get_definition("not"), 200, 25),
            Node.create("or", NODE_REGISTRY.get_definition("or"), 100, 225),
            Node.create("final_or", NODE_REGISTRY.get_definition("or"), 300, 125),
            Node.create("out", NODE_REGISTRY.get_definition("output"), 400, 125)
        ]
        
        for node in nodes:
            graph.add_node(node)
        
        # Nets: A,B -> AND -> NOT, C,D -> OR, NOT,OR -> final_or -> out
        nets = [
            Net.create_two_point("c1", "a", "out", "and", "in1"),
            Net.create_two_point("c2", "b", "out", "and", "in2"),
            Net.create_two_point("c3", "and", "out", "not", "in"),
            Net.create_two_point("c4", "c", "out", "or", "in1"),
            Net.create_two_point("c5", "d", "out", "or", "in2"),
            Net.create_two_point("c6", "not", "out", "final_or", "in1"),
            Net.create_two_point("c7", "or", "out", "final_or", "in2"),
            Net.create_two_point("c8", "final_or", "out", "out", "in")
        ]
        
        for conn in nets:
            graph.add_net(conn)
        
        order = graph._get_evaluation_order()
        
        # Verify dependencies are respected
        def get_index(node_id):
            return order.index(node_id)
        
        # Inputs should come before their dependents
        assert get_index("a") < get_index("and")
        assert get_index("b") < get_index("and")
        assert get_index("c") < get_index("or")
        assert get_index("d") < get_index("or")
        
        # Gates should come before their dependents
        assert get_index("and") < get_index("not")
        assert get_index("not") < get_index("final_or")
        assert get_index("or") < get_index("final_or")
        assert get_index("final_or") < get_index("out")
    
    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected and raise an error"""
        # Create a circular dependency: A -> B -> C -> A
        graph = GraphData()
        
        nodes = [
            Node.create("a", NODE_REGISTRY.get_definition("not"), 0, 0),
            Node.create("b", NODE_REGISTRY.get_definition("not"), 100, 0),
            Node.create("c", NODE_REGISTRY.get_definition("not"), 200, 0)
        ]
        
        for node in nodes:
            graph.add_node(node)
        
        # Create circular nets
        nets = [
            Net.create_two_point("c1", "a", "out", "b", "in"),
            Net.create_two_point("c2", "b", "out", "c", "in"),
            Net.create_two_point("c3", "c", "out", "a", "in")  # This creates the cycle
        ]
        
        for conn in nets:
            graph.add_net(conn)
        
        with pytest.raises(ValueError, match="Circular dependency detected involving nodes"):
            graph._get_evaluation_order()
    
    def test_simulate_demo_graph(self):
        """Test simulation using the demo graph"""
        demo_graph = create_demo_graph()
        
        # Set input values
        demo_graph.set_input_value("input_a", True)
        demo_graph.set_input_value("input_b", False)
        demo_graph.set_input_value("input_c", True)
        
        # Run simulation
        result = demo_graph.simulate()
        
        assert result == True
        
        # Verify expected results based on demo graph structure
        # A=True, B=False, C=True
        # AND gate: A AND B = True AND False = False
        # OR gate: B OR C = False OR True = True
        # NOT gate: NOT(AND result) = NOT(False) = True
        # Output A: NOT result = True
        # Output B: OR result = True
        
        assert demo_graph.get_node_value("input_a") == True
        assert demo_graph.get_node_value("input_b") == False
        assert demo_graph.get_node_value("input_c") == True
        assert demo_graph.get_node_value("and_gate") == False  # True AND False
        assert demo_graph.get_node_value("or_gate") == True    # False OR True
        assert demo_graph.get_node_value("not_gate") == True   # NOT False
        assert demo_graph.get_node_value("output_a") == True   # Pass through from NOT
        assert demo_graph.get_node_value("output_b") == True   # Pass through from OR
    
    def test_simulate_clears_previous_values(self):
        """Test that simulation clears previous non-input node values"""
        # Set input and run simulation
        self.graph.set_input_value("input_a", True)
        self.graph.simulate()
        
        # Verify initial results
        assert self.graph.get_node_value("not_gate") == False
        assert self.graph.get_node_value("output") == False
        
        # Change input and run simulation again
        self.graph.set_input_value("input_a", False)
        self.graph.simulate()
        
        # Verify that values were properly updated
        assert self.graph.get_node_value("not_gate") == True
        assert self.graph.get_node_value("output") == True
    
    def test_simulate_with_disconnected_nodes(self):
        """Test simulation with disconnected nodes (nodes with no inputs)"""
        # Add a disconnected input node
        disconnected = Node.create("disconnected", NODE_REGISTRY.get_definition("input"), 500, 50, label="DISC")
        self.graph.add_node(disconnected)
        
        # Set values for all input nodes
        self.graph.set_input_value("input_a", True)
        self.graph.set_input_value("disconnected", False)
        
        # Simulation should still work
        result = self.graph.simulate()
        assert result == True
        
        # Both input nodes should maintain their values
        assert self.graph.get_node_value("input_a") == True
        assert self.graph.get_node_value("disconnected") == False
    
    def test_simulate_logging(self):
        """Test that simulation produces appropriate log messages"""
        with patch.object(self.graph, 'logger') as mock_logger:
            self.graph.set_input_value("input_a", True)
            
            result = self.graph.simulate()
            
            assert result == True
            
            # Verify logging calls were made
            mock_logger.info.assert_any_call("Starting circuit simulation")
            mock_logger.info.assert_any_call("Combinational simulation completed: evaluated 2 nodes")
            
            # Verify debug logging for evaluation
            mock_logger.debug.assert_any_call("Node evaluation order: ['input_a', 'not_gate', 'output']")
    
    def test_partial_input_setting_error(self):
        """Test error when only some inputs are set"""
        # Create circuit with multiple inputs
        graph = GraphData()
        
        input_a = Node.create("a", NODE_REGISTRY.get_definition("input"), 0, 0)
        input_b = Node.create("b", NODE_REGISTRY.get_definition("input"), 0, 50)
        and_gate = Node.create("and", NODE_REGISTRY.get_definition("and"), 100, 25)
        
        graph.add_node(input_a)
        graph.add_node(input_b) 
        graph.add_node(and_gate)
        
        graph.add_net(Net.create_two_point("c1", "a", "out", "and", "in1"))
        graph.add_net(Net.create_two_point("c2", "b", "out", "and", "in2"))
        
        # Set only one input
        graph.set_input_value("a", True)
        # Don't set input_b
        
        with pytest.raises(ValueError, match="Input nodes must have values set before simulation: \\['b'\\]"):
            graph.simulate()


class TestUIInteraction:
    """Test cases for UI interaction functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
        
        # Create a simple test circuit with input and output nodes
        self.input_a = Node.create("input_a", NODE_REGISTRY.get_definition("input"), 50, 50, label="A")
        self.input_b = Node.create("input_b", NODE_REGISTRY.get_definition("input"), 50, 150, label="B")
        self.and_gate = Node.create("and_gate", NODE_REGISTRY.get_definition("and"), 200, 100, label="AND")
        self.output = Node.create("output", NODE_REGISTRY.get_definition("output"), 350, 100, label="OUT")
        
        self.graph.add_node(self.input_a)
        self.graph.add_node(self.input_b)
        self.graph.add_node(self.and_gate)
        self.graph.add_node(self.output)
        
        # Connect: A,B -> AND -> OUT
        self.graph.add_net(Net.create_two_point("c1", "input_a", "out", "and_gate", "in1"))
        self.graph.add_net(Net.create_two_point("c2", "input_b", "out", "and_gate", "in2"))
        self.graph.add_net(Net.create_two_point("c3", "and_gate", "out", "output", "in"))
    
    def test_toggle_input_node_cycle(self):
        """Test that input node toggling cycles through None -> True -> False -> None"""
        # Initially None
        assert self.graph.get_node_value("input_a") == None
        
        # First toggle: None -> True
        result = self.graph.toggle_input_node("input_a")
        assert result == True
        assert self.graph.get_node_value("input_a") == True
        
        # Second toggle: True -> False
        result = self.graph.toggle_input_node("input_a")
        assert result == True
        assert self.graph.get_node_value("input_a") == False
        
        # Third toggle: False -> None
        result = self.graph.toggle_input_node("input_a")
        assert result == True
        assert self.graph.get_node_value("input_a") == None
        
        # Fourth toggle: None -> True (cycle complete)
        result = self.graph.toggle_input_node("input_a")
        assert result == True
        assert self.graph.get_node_value("input_a") == True
    
    def test_toggle_input_node_invalid_node(self):
        """Test that toggling non-existent node raises error"""
        with pytest.raises(ValueError, match="Node with ID 'nonexistent' does not exist"):
            self.graph.toggle_input_node("nonexistent")
    
    def test_toggle_input_node_non_input(self):
        """Test that toggling non-input node raises error"""
        with pytest.raises(ValueError, match="Node 'and_gate' is not an input node"):
            self.graph.toggle_input_node("and_gate")
    
    def test_reset_simulation_clears_all_values(self):
        """Test that reset simulation clears all node values"""
        # Set some values
        self.graph.set_input_value("input_a", True)
        self.graph.set_input_value("input_b", False)
        
        # Run simulation to populate other nodes
        self.graph.simulate()
        
        # Verify values are set
        assert self.graph.get_node_value("input_a") == True
        assert self.graph.get_node_value("input_b") == False
        assert self.graph.get_node_value("and_gate") == False
        assert self.graph.get_node_value("output") == False
        
        # Reset simulation
        result = self.graph.reset_simulation()
        
        # Verify all values are cleared
        assert result == True  # UI should refresh
        assert self.graph.get_node_value("input_a") == None
        assert self.graph.get_node_value("input_b") == None
        assert self.graph.get_node_value("and_gate") == None
        assert self.graph.get_node_value("output") == None
    
    def test_reset_simulation_no_values_to_clear(self):
        """Test that reset simulation returns False when no values to clear"""
        # Don't set any values
        result = self.graph.reset_simulation()
        
        # Should return False since nothing was cleared
        assert result == False
    
    def test_handle_double_click_input_node_toggleing(self):
        """Test that double-clicking input node toggles its value"""
        # Click on input node position (50, 50 is center, but we need to click within bounds)
        # Input node is 50x50 at position (50,50), so clicking at (60,60) should hit it
        
        # Initial value is None
        assert self.graph.get_node_value("input_a") == None
        
        # Double-click on input node
        result = self.graph.handle_double_click(60.0, 60.0)
        
        assert result == True  # UI should refresh
        assert self.graph.get_node_value("input_a") == True  # Should be toggled to True
        
        # Double-click again
        result = self.graph.handle_double_click(60.0, 60.0)
        
        assert result == True
        assert self.graph.get_node_value("input_a") == False  # Should be toggled to False
    
    def test_handle_double_click_non_input_node_starts_editing(self):
        """Test that double-clicking non-input node starts label editing"""
        # Double-click on AND gate (200x60 at position 200,100, so click at 220,110)
        result = self.graph.handle_double_click(220.0, 110.0)
        
        assert result == True  # UI should refresh
        assert self.graph.editing_node_id == "and_gate"
        assert self.graph.editing_text == "AND"  # Should be initialized with current label
    
    def test_handle_double_click_empty_area(self):
        """Test that double-clicking empty area does nothing"""
        result = self.graph.handle_double_click(1000.0, 1000.0)  # Far away from any nodes
        
        assert result == False  # No UI refresh needed
        assert self.graph.editing_node_id == None
    
    def test_complete_simulation_workflow(self):
        """Test complete user workflow: toggle inputs -> simulate -> reset"""
        # Step 1: Toggle input values
        self.graph.toggle_input_node("input_a")  # True
        self.graph.toggle_input_node("input_b")  # True
        
        assert self.graph.get_node_value("input_a") == True
        assert self.graph.get_node_value("input_b") == True
        
        # Step 2: Run simulation
        result = self.graph.simulate()
        assert result == True
        
        # Verify results: True AND True = True
        assert self.graph.get_node_value("and_gate") == True
        assert self.graph.get_node_value("output") == True
        
        # Step 3: Reset simulation
        reset_result = self.graph.reset_simulation()
        assert reset_result == True
        
        # Verify all values cleared
        assert self.graph.get_node_value("input_a") == None
        assert self.graph.get_node_value("input_b") == None
        assert self.graph.get_node_value("and_gate") == None
        assert self.graph.get_node_value("output") == None
    
    def test_simulation_with_mixed_input_states(self):
        """Test simulation with inputs in different states"""
        # Set one input to True, leave other as None
        self.graph.toggle_input_node("input_a")  # True
        # input_b remains None
        
        # Simulation should fail because not all inputs are set
        with pytest.raises(ValueError, match="Input nodes must have values set before simulation"):
            self.graph.simulate()
        
        # Set second input
        self.graph.toggle_input_node("input_b")  # True
        
        # Now simulation should work
        result = self.graph.simulate()
        assert result == True
        assert self.graph.get_node_value("and_gate") == True  # True AND True
    
    def test_toggle_input_node_logging(self):
        """Test that input node toggling produces appropriate log messages"""
        with patch.object(self.graph, 'logger') as mock_logger:
            self.graph.toggle_input_node("input_a")
            
            mock_logger.debug.assert_called_with("Toggled input node 'input_a' to: True")
    
    def test_reset_simulation_logging(self):
        """Test that reset simulation produces appropriate log messages"""
        with patch.object(self.graph, 'logger') as mock_logger:
            # Set a value first
            self.graph.set_input_value("input_a", True)
            
            # Reset
            self.graph.reset_simulation()
            
            mock_logger.info.assert_called_with("Resetting circuit simulation")
            mock_logger.debug.assert_called_with("Cleared all node values")
            
            # Test when no values to clear
            mock_logger.reset_mock()
            self.graph.reset_simulation()
            
            mock_logger.debug.assert_called_with("No node values to clear")
    
    def test_input_node_visual_state_after_toggle(self):
        """Test that input node shows correct visual state after toggling"""
        # Toggle to True
        self.graph.toggle_input_node("input_a")
        
        # Check Slint format includes correct state
        slint_data = self.graph.to_slint_format()
        input_node_data = next(node for node in slint_data['nodes'] if node['id'] == 'input_a')
        
        assert input_node_data['value'] == True
        
        # Toggle to False
        self.graph.toggle_input_node("input_a")
        
        slint_data = self.graph.to_slint_format()
        input_node_data = next(node for node in slint_data['nodes'] if node['id'] == 'input_a')
        
        assert input_node_data['value'] == False
        
        # Toggle to None
        self.graph.toggle_input_node("input_a")
        
        slint_data = self.graph.to_slint_format()
        input_node_data = next(node for node in slint_data['nodes'] if node['id'] == 'input_a')
        
        assert input_node_data['value'] == None  # None should be preserved in to_slint_format


class TestModeBasedSimulation:
    """Test cases for mode-based simulation functionality"""
    
    def setup_method(self):
        """Set up test graph for each test"""
        self.graph = GraphData()
        
        # Create test nodes
        input_node = Node.create("input1", NODE_REGISTRY.get_definition("input"), 50, 50)
        input_node2 = Node.create("input2", NODE_REGISTRY.get_definition("input"), 50, 120)
        and_node = Node.create("and1", NODE_REGISTRY.get_definition("and"), 150, 85)
        output_node = Node.create("output1", NODE_REGISTRY.get_definition("output"), 250, 85)
        
        self.graph.add_node(input_node)
        self.graph.add_node(input_node2)
        self.graph.add_node(and_node)
        self.graph.add_node(output_node)
        
        # Create nets
        conn1 = Net.create_two_point("c1", "input1", "out", "and1", "in1")
        conn2 = Net.create_two_point("c2", "input2", "out", "and1", "in2")
        conn3 = Net.create_two_point("c3", "and1", "out", "output1", "in")
        
        self.graph.add_net(conn1)
        self.graph.add_net(conn2)
        self.graph.add_net(conn3)
    
    def test_initial_mode_is_edit(self):
        """Test that the graph starts in edit mode"""
        assert self.graph.simulation_mode == False
        slint_data = self.graph.to_slint_format()
        assert slint_data['simulation_mode'] == False
    
    def test_enter_simulation_mode(self):
        """Test entering simulation mode"""
        # Initially None values
        assert self.graph.get_node_value("input1") == None
        assert self.graph.get_node_value("input2") == None
        
        result = self.graph.enter_simulation_mode()
        
        assert result == True
        assert self.graph.simulation_mode == True
        
        # Input nodes should be initialized to False
        assert self.graph.get_node_value("input1") == False
        assert self.graph.get_node_value("input2") == False
        
        # Should be included in slint format
        slint_data = self.graph.to_slint_format()
        assert slint_data['simulation_mode'] == True
    
    def test_enter_simulation_mode_already_in_simulation(self):
        """Test entering simulation mode when already in simulation mode"""
        self.graph.enter_simulation_mode()
        
        result = self.graph.enter_simulation_mode()
        
        assert result == False  # Should return False as no change was made
        assert self.graph.simulation_mode == True
    
    def test_enter_edit_mode(self):
        """Test entering edit mode from simulation mode"""
        # Enter simulation mode first
        self.graph.enter_simulation_mode()
        self.graph.set_input_value("input1", True)
        self.graph.set_input_value("input2", False)
        
        result = self.graph.enter_edit_mode()
        
        assert result == True
        assert self.graph.simulation_mode == False
        
        # All node values should be reset to None
        assert self.graph.get_node_value("input1") == None
        assert self.graph.get_node_value("input2") == None
        assert self.graph.get_node_value("and1") == None
        assert self.graph.get_node_value("output1") == None
        
        # Should be included in slint format
        slint_data = self.graph.to_slint_format()
        assert slint_data['simulation_mode'] == False
    
    def test_enter_edit_mode_already_in_edit(self):
        """Test entering edit mode when already in edit mode"""
        result = self.graph.enter_edit_mode()
        
        assert result == False  # Should return False as no change was made
        assert self.graph.simulation_mode == False
    
    def test_toggle_input_node_in_simulation_mode(self):
        """Test toggling input nodes in simulation mode"""
        self.graph.enter_simulation_mode()
        
        # Initially False
        assert self.graph.get_node_value("input1") == False
        
        # Toggle to True
        with patch.object(self.graph, 'logger') as mock_logger:
            result = self.graph.toggle_input_node("input1")
        
        assert result == True
        assert self.graph.get_node_value("input1") == True
        
        # Toggle back to False
        with patch.object(self.graph, 'logger') as mock_logger:
            result = self.graph.toggle_input_node("input1")
        
        assert result == True
        assert self.graph.get_node_value("input1") == False
    
    def test_toggle_input_node_in_edit_mode(self):
        """Test toggling input nodes in edit mode (original behavior)"""
        # Initially None
        assert self.graph.get_node_value("input1") == None
        
        # Toggle to True
        self.graph.toggle_input_node("input1")
        assert self.graph.get_node_value("input1") == True
        
        # Toggle to False
        self.graph.toggle_input_node("input1")
        assert self.graph.get_node_value("input1") == False
        
        # Toggle back to None
        self.graph.toggle_input_node("input1")
        assert self.graph.get_node_value("input1") == None
    
    def test_auto_simulation_on_input_toggle(self):
        """Test that circuit is automatically simulated when input is toggled in simulation mode"""
        self.graph.enter_simulation_mode()
        
        with patch.object(self.graph, 'simulate') as mock_simulate:
            mock_simulate.return_value = True
            
            self.graph.toggle_input_node("input1")
            
            mock_simulate.assert_called_once()
    
    def test_handle_double_click_simulation_mode_input_node(self):
        """Test double-clicking input nodes in simulation mode"""
        self.graph.enter_simulation_mode()
        
        # Double-click on input node should toggle its value
        result = self.graph.handle_double_click(60.0, 60.0)  # Click on input1
        
        assert result == True
        assert self.graph.get_node_value("input1") == True  # Should toggle from False to True
    
    def test_handle_double_click_simulation_mode_non_input_node(self):
        """Test double-clicking non-input nodes in simulation mode"""
        self.graph.enter_simulation_mode()
        
        with patch.object(self.graph, 'logger') as mock_logger:
            # Double-click on non-input node should do nothing
            result = self.graph.handle_double_click(160.0, 95.0)  # Click on and1
        
        assert result == False
        assert self.graph.editing_node_id == None  # Should not start editing
        mock_logger.debug.assert_called_with("Ignoring double-click on and node in simulation mode")
    
    def test_handle_double_click_edit_mode_non_input_node(self):
        """Test double-clicking non-input nodes in edit mode"""
        # Should start label editing as before
        result = self.graph.handle_double_click(160.0, 95.0)  # Click on and1
        
        assert result == True
        assert self.graph.editing_node_id == "and1"  # Should start editing
    
    def test_pointer_down_simulation_mode_blocks_editing(self):
        """Test that pointer down in simulation mode blocks most editing operations"""
        self.graph.enter_simulation_mode()
        
        # Should allow node selection but not dragging or net creation
        result = self.graph.handle_pointer_down(60.0, 60.0)  # Click on input1
        
        assert result == True
        assert self.graph.selected_node_id == "input1"  # Should select node
        assert self.graph.pointer_state == PointerState.IDLE  # Should not enter pressed state
    
    def test_pointer_move_simulation_mode_blocked(self):
        """Test that pointer move in simulation mode is blocked"""
        self.graph.enter_simulation_mode()
        
        result = self.graph.handle_pointer_move(100.0, 100.0)
        
        assert result == False  # Should be blocked
    
    def test_pointer_up_simulation_mode_resets_state(self):
        """Test that pointer up in simulation mode resets state properly"""
        self.graph.enter_simulation_mode()
        self.graph.pointer_state = PointerState.PRESSED  # Simulate pressed state
        
        result = self.graph.handle_pointer_up(100.0, 100.0)
        
        assert result == False
        assert self.graph.pointer_state == PointerState.IDLE
    
    def test_create_node_blocked_in_simulation_mode(self):
        """Test that node creation is blocked in simulation mode"""
        self.graph.enter_simulation_mode()
        
        with patch.object(self.graph, 'logger') as mock_logger:
            result = self.graph.create_node_at_position("input", 100.0, 100.0)
        
        assert result == False
        mock_logger.debug.assert_called_with("Node creation is blocked in simulation mode")
    
    def test_delete_selected_blocked_in_simulation_mode(self):
        """Test that deletion is blocked in simulation mode"""
        self.graph.enter_simulation_mode()
        self.graph.select_node("input1")
        
        with patch.object(self.graph, 'logger') as mock_logger:
            result = self.graph.delete_selected()
        
        assert result == False
        mock_logger.debug.assert_called_with("Deletion is blocked in simulation mode")
        assert "input1" in self.graph.nodes  # Node should still exist
    
    def test_toolbox_selection_blocked_in_simulation_mode(self):
        """Test that toolbox node type selection is blocked in simulation mode"""
        self.graph.enter_simulation_mode()
        
        with patch.object(self.graph, 'logger') as mock_logger:
            result = self.graph.select_toolbox_node_type("input")
        
        assert result == False
        mock_logger.debug.assert_called_with("Toolbox selection is blocked in simulation mode")
        assert self.graph.selected_node_type == None
        assert self.graph.toolbox_creation_mode == False
    
    def test_mode_transition_clears_editing_states(self):
        """Test that mode transitions clear active editing states"""
        # Start label editing (this should work in edit mode)
        result = self.graph.start_label_edit("input1")
        assert result == True
        assert self.graph.editing_node_id == "input1"
        
        # Select toolbox node type separately (this clears other states)
        self.graph.cancel_label_edit()
        result = self.graph.select_toolbox_node_type("and")
        assert result == True
        assert self.graph.selected_node_type == "and"
        assert self.graph.toolbox_creation_mode == True
        
        # Enter simulation mode should clear these states
        self.graph.enter_simulation_mode()
        
        assert self.graph.editing_node_id == None
        assert self.graph.editing_text == ""
        assert self.graph.selected_node_type == None
        assert self.graph.toolbox_creation_mode == False
    
    def test_input_nodes_default_false_in_simulation_mode(self):
        """Test that input nodes get default False value when entering simulation mode"""
        # Set some input values in edit mode
        self.graph.set_input_value("input1", True)
        self.graph.set_input_value("input2", None)  # Explicitly set to None
        
        # Enter simulation mode
        self.graph.enter_simulation_mode()
        
        # Both should be False now
        assert self.graph.get_node_value("input1") == False  # Changed from True
        assert self.graph.get_node_value("input2") == False  # Changed from None
    
    def test_complete_workflow_edit_simulate_edit(self):
        """Test complete workflow: edit -> simulation -> edit"""
        # Start in edit mode
        assert self.graph.simulation_mode == False
        
        # Set up circuit in edit mode
        self.graph.set_input_value("input1", True)
        self.graph.set_input_value("input2", True)
        
        # Enter simulation mode
        self.graph.enter_simulation_mode()
        assert self.graph.simulation_mode == True
        assert self.graph.get_node_value("input1") == False  # Reset to default
        assert self.graph.get_node_value("input2") == False  # Reset to default
        
        # Toggle inputs in simulation mode
        self.graph.toggle_input_node("input1")  # False -> True
        self.graph.toggle_input_node("input2")  # False -> True
        assert self.graph.get_node_value("input1") == True
        assert self.graph.get_node_value("input2") == True
        
        # Exit to edit mode
        self.graph.enter_edit_mode()
        assert self.graph.simulation_mode == False
        
        # All values should be None again
        assert self.graph.get_node_value("input1") == None
        assert self.graph.get_node_value("input2") == None
        assert self.graph.get_node_value("and1") == None
        assert self.graph.get_node_value("output1") == None


class TestFeedbackLoops:
    """Test cases for feedback loop detection and sequential circuit simulation"""
    
    def setup_method(self):
        """Set up test SR NOR latch for each test"""
        from logicsim.graph_data import create_sr_nor_latch_demo
        self.graph = create_sr_nor_latch_demo()
    
    def test_tarjan_scc_detection(self):
        """Test Tarjan's strongly connected components algorithm"""
        sccs = self.graph._tarjan_scc()
        
        # Should find strongly connected components
        assert len(sccs) > 0
        
        # The feedback between nor1 and nor2 should create an SCC
        feedback_scc = None
        for scc in sccs:
            if "nor1" in scc and "nor2" in scc:
                feedback_scc = scc
                break
        
        assert feedback_scc is not None
        assert "nor1" in feedback_scc
        assert "nor2" in feedback_scc
    
    def test_feedback_loop_detection(self):
        """Test detection of feedback loops in circuit"""
        feedback_components, has_feedback = self.graph._detect_feedback_loops()
        
        assert has_feedback == True
        assert len(feedback_components) > 0
        
        # Should detect the NOR gate feedback loop
        nor_feedback = None
        for component in feedback_components:
            if "nor1" in component and "nor2" in component:
                nor_feedback = component
                break
        
        assert nor_feedback is not None
        assert "nor1" in nor_feedback
        assert "nor2" in nor_feedback
    
    def test_initialize_feedback_nodes(self):
        """Test initialization of feedback nodes to power-on reset state"""
        # Clear all values first
        for node in self.graph.nodes.values():
            node.value = None
        
        feedback_components, _ = self.graph._detect_feedback_loops()
        self.graph._initialize_feedback_nodes(feedback_components)
        
        # Feedback nodes should be initialized to False
        assert self.graph.get_node_value("nor1") == False
        assert self.graph.get_node_value("nor2") == False
        
        # Input and output nodes should remain None
        assert self.graph.get_node_value("set") == None
        assert self.graph.get_node_value("reset") == None
        assert self.graph.get_node_value("q") == None
        assert self.graph.get_node_value("q_not") == None
    
    def test_sr_latch_simulation_iterative(self):
        """Test iterative simulation of SR NOR latch"""
        # Enter simulation mode to initialize
        self.graph.enter_simulation_mode()
        
        # Verify feedback loops are detected
        feedback_components, has_feedback = self.graph._detect_feedback_loops()
        assert has_feedback == True
        
        # Set S=0, R=0 (hold state)
        self.graph.set_input_value("set", False)
        self.graph.set_input_value("reset", False)
        
        # Run simulation
        result = self.graph.simulate()
        assert result == True
        
        # Should converge to a stable state
        # With both inputs False and power-on reset (both NORs start False)
        # The circuit should settle to Q=True, Q̅=False
        assert self.graph.get_node_value("q") == True
        assert self.graph.get_node_value("q_not") == False
    
    def test_sr_latch_set_operation(self):
        """Test SET operation of SR NOR latch"""
        self.graph.enter_simulation_mode()
        
        # Set S=1, R=0 (SET operation)
        self.graph.set_input_value("set", True)
        self.graph.set_input_value("reset", False)
        
        result = self.graph.simulate()
        assert result == True
        
        # Should set Q=1, Q̅=0
        assert self.graph.get_node_value("q") == True
        assert self.graph.get_node_value("q_not") == False
    
    def test_sr_latch_reset_operation(self):
        """Test RESET operation of SR NOR latch"""
        self.graph.enter_simulation_mode()
        
        # First set the latch
        self.graph.set_input_value("set", True)
        self.graph.set_input_value("reset", False)
        self.graph.simulate()
        
        # Now reset: S=0, R=1
        self.graph.set_input_value("set", False)
        self.graph.set_input_value("reset", True)
        
        result = self.graph.simulate()
        assert result == True
        
        # Should reset Q=0, Q̅=1
        assert self.graph.get_node_value("q") == False
        assert self.graph.get_node_value("q_not") == True
    
    def test_sr_latch_hold_state(self):
        """Test HOLD state of SR NOR latch"""
        self.graph.enter_simulation_mode()
        
        # Set the latch first
        self.graph.set_input_value("set", True)
        self.graph.set_input_value("reset", False)
        self.graph.simulate()
        
        # Verify it's set
        assert self.graph.get_node_value("q") == True
        assert self.graph.get_node_value("q_not") == False
        
        # Now go to hold state: S=0, R=0
        self.graph.set_input_value("set", False)
        self.graph.set_input_value("reset", False)
        
        result = self.graph.simulate()
        assert result == True
        
        # Should maintain previous state
        assert self.graph.get_node_value("q") == True
        assert self.graph.get_node_value("q_not") == False
    
    def test_sr_latch_invalid_state(self):
        """Test invalid state S=1, R=1 of SR NOR latch"""
        self.graph.enter_simulation_mode()
        
        # Set S=1, R=1 (invalid/forbidden state)
        self.graph.set_input_value("set", True)
        self.graph.set_input_value("reset", True)
        
        result = self.graph.simulate()
        assert result == True
        
        # Should result in Q=0, Q̅=0 (both outputs low)
        assert self.graph.get_node_value("q") == False
        assert self.graph.get_node_value("q_not") == False
    
    def test_iterative_simulation_convergence(self):
        """Test that iterative simulation properly converges"""
        self.graph.enter_simulation_mode()
        
        # Test multiple input changes to ensure convergence works
        for set_val in [False, True, False]:
            for reset_val in [False, True, False]:
                if not (set_val and reset_val):  # Skip invalid state
                    self.graph.set_input_value("set", set_val)
                    self.graph.set_input_value("reset", reset_val)
                    
                    result = self.graph.simulate()
                    assert result == True  # Should always converge
    
    def test_combinational_circuit_still_works(self):
        """Test that combinational circuits still work with new system"""
        # Create a simple combinational circuit (no feedback)
        from logicsim.graph_data import GraphData, Node, Net, NetConnector, NODE_REGISTRY
        
        graph = GraphData()
        
        # Create nodes: Input -> NOT -> Output
        input_node = Node.create("input1", NODE_REGISTRY.get_definition("input"), 50, 50)
        not_node = Node.create("not1", NODE_REGISTRY.get_definition("not"), 150, 50)
        output_node = Node.create("output1", NODE_REGISTRY.get_definition("output"), 250, 50)
        
        graph.add_node(input_node)
        graph.add_node(not_node)
        graph.add_node(output_node)
        
        # Add net
        conn = Net.create_two_point("c1", "input1", "out", "not1", "in")
        graph.add_net(conn)
        conn2 = Net.create_two_point("c2", "not1", "out", "output1", "in")
        graph.add_net(conn2)
        
        # Test simulation
        graph.enter_simulation_mode()
        graph.set_input_value("input1", True)
        
        result = graph.simulate()
        assert result == True
        
        # Should use single-pass combinational simulation
        assert graph.get_node_value("output1") == False  # NOT True = False


class TestNetValueVisualization:
    """Test cases for net-based value visualization system"""
    
    def setup_method(self):
        """Set up test circuit for each test"""
        self.graph = GraphData()
        
        # Create a simple circuit: Input -> NOT -> Output
        input_node = Node.create("input1", NODE_REGISTRY.get_definition("input"), 50, 50)
        not_node = Node.create("not1", NODE_REGISTRY.get_definition("not"), 150, 50)
        output_node = Node.create("output1", NODE_REGISTRY.get_definition("output"), 250, 50)
        
        self.graph.add_node(input_node)
        self.graph.add_node(not_node)
        self.graph.add_node(output_node)
        
        # Add nets
        conn1 = Net.create_two_point("c1", "input1", "out", "not1", "in")
        conn2 = Net.create_two_point("c2", "not1", "out", "output1", "in")
        self.graph.add_net(conn1)
        self.graph.add_net(conn2)
    
    def test_get_net_value_input_node(self):
        """Test _get_net_value for input node nets"""
        # Set input value
        self.graph.nodes["input1"].value = True
        
        # Test net from input node
        net = self.graph.nets["c1"]
        value, has_value = self.graph._get_net_value(net)
        
        assert value == True
        assert has_value == True
    
    def test_get_net_value_input_node_no_value(self):
        """Test _get_net_value for input node with no value"""
        # Input node has no value set
        assert self.graph.nodes["input1"].value is None
        
        net = self.graph.nets["c1"]
        value, has_value = self.graph._get_net_value(net)
        
        assert value == False
        assert has_value == False
    
    def test_get_net_value_logic_gate(self):
        """Test _get_net_value for logic gate nets"""
        # Set computed value on NOT gate
        self.graph.nodes["not1"].value = False
        
        # Test net from NOT gate
        net = self.graph.nets["c2"]
        value, has_value = self.graph._get_net_value(net)
        
        assert value == False
        assert has_value == True
    
    def test_get_net_value_logic_gate_no_value(self):
        """Test _get_net_value for logic gate with no computed value"""
        # NOT gate has no computed value
        assert self.graph.nodes["not1"].value is None
        
        net = self.graph.nets["c2"]
        value, has_value = self.graph._get_net_value(net)
        
        assert value == False
        assert has_value == False
    
    def test_get_net_value_net_to_output_node(self):
        """Test _get_net_value for net that connects TO an output node"""
        # Set value on NOT gate (which drives the output)
        self.graph.nodes["not1"].value = False
        
        # Test the existing net TO the output node (c2: not1 -> output1)
        net = self.graph.nets["c2"]
        value, has_value = self.graph._get_net_value(net)
        
        assert value == False
        assert has_value == True
    
    def test_get_net_value_nonexistent_node(self):
        """Test _get_net_value for net with nonexistent source node"""
        # Create net with invalid source node
        invalid_conn = Net.create_two_point("invalid", "nonexistent", "out", "not1", "in")
        value, has_value = self.graph._get_net_value(invalid_conn)
        
        assert value == False
        assert has_value == False
    
    def test_get_net_value_unknown_node_type(self):
        """Test _get_net_value for unknown node type"""
        # Create a node with unknown type
        unknown_node = Node("unknown1", "unknown_type", 100, 100, 80, 60, "Unknown", "gray", [], None)
        self.graph.add_node(unknown_node)
        
        # Create net from unknown node
        unknown_conn = Net.create_two_point("unknown_c", "unknown1", "out", "not1", "in")
        value, has_value = self.graph._get_net_value(unknown_conn)
        
        assert value == False
        assert has_value == False
    
    def test_get_net_value_all_logic_gate_types(self):
        """Test _get_net_value for all logic gate types"""
        # Test AND gate
        and_node = Node.create("and1", NODE_REGISTRY.get_definition("and"), 300, 50)
        and_node.value = True
        self.graph.add_node(and_node)
        and_conn = Net.create_two_point("and_c", "and1", "out", "output1", "in")
        self.graph.add_net(and_conn)
        value, has_value = self.graph._get_net_value(and_conn)
        assert value == True and has_value == True
        
        # Test OR gate
        or_node = Node.create("or1", NODE_REGISTRY.get_definition("or"), 300, 100)
        or_node.value = False
        self.graph.add_node(or_node)
        or_conn = Net.create_two_point("or_c", "or1", "out", "output1", "in")
        self.graph.add_net(or_conn)
        value, has_value = self.graph._get_net_value(or_conn)
        assert value == False and has_value == True
        
        # Test NOR gate
        nor_node = Node.create("nor1", NODE_REGISTRY.get_definition("nor"), 300, 150)
        nor_node.value = True
        self.graph.add_node(nor_node)
        nor_conn = Net.create_two_point("nor_c", "nor1", "out", "output1", "in")
        self.graph.add_net(nor_conn)
        value, has_value = self.graph._get_net_value(nor_conn)
        assert value == True and has_value == True
    
    def test_to_slint_format_includes_net_values_edit_mode(self):
        """Test that to_slint_format includes net values in edit mode"""
        # In edit mode, simulation_mode should be False
        graph_data = self.graph.to_slint_format()
        
        assert "nets" in graph_data
        assert len(graph_data["nets"]) == 2
        
        for conn in graph_data["nets"]:
            assert "value" in conn
            assert "has_value" in conn
            assert "simulation_mode" in conn
            
            # In edit mode, simulation_mode should be False
            assert conn["simulation_mode"] == False
            # Nets should have no values in edit mode
            assert conn["has_value"] == False
    
    def test_to_slint_format_includes_net_values_simulation_mode(self):
        """Test that to_slint_format includes net values in simulation mode"""
        # Enter simulation mode and run simulation
        self.graph.enter_simulation_mode()
        self.graph.set_input_value("input1", True)
        self.graph.simulate()
        
        graph_data = self.graph.to_slint_format()
        
        assert "nets" in graph_data
        assert len(graph_data["nets"]) == 2
        
        for conn in graph_data["nets"]:
            assert "value" in conn
            assert "has_value" in conn
            assert "simulation_mode" in conn
            
            # In simulation mode, simulation_mode should be True
            assert conn["simulation_mode"] == True
            # Nets should have values in simulation mode
            assert conn["has_value"] == True
        
        # Check specific net values
        c1_conn = next(c for c in graph_data["nets"] if c["id"] == "c1")
        c2_conn = next(c for c in graph_data["nets"] if c["id"] == "c2")
        
        # c1: input1(True) -> not1, so value should be True
        assert c1_conn["value"] == True
        # c2: not1(False) -> output1, so value should be False (NOT True = False)
        assert c2_conn["value"] == False
    
    def test_to_slint_format_includes_node_has_value_field(self):
        """Test that to_slint_format includes has_value field for nodes"""
        # Test in edit mode
        graph_data = self.graph.to_slint_format()
        
        for node in graph_data["nodes"]:
            assert "has_value" in node
            # In edit mode, nodes should not have values
            assert node["has_value"] == False
        
        # Test in simulation mode
        self.graph.enter_simulation_mode()
        self.graph.set_input_value("input1", True)
        self.graph.simulate()
        
        graph_data = self.graph.to_slint_format()
        
        for node in graph_data["nodes"]:
            assert "has_value" in node
            # In simulation mode, all nodes should have values
            assert node["has_value"] == True
    
    def test_to_slint_format_net_values_with_different_input_states(self):
        """Test net values with different input states"""
        self.graph.enter_simulation_mode()
        
        # Test with input = False
        self.graph.set_input_value("input1", False)
        self.graph.simulate()
        
        graph_data = self.graph.to_slint_format()
        c1_conn = next(c for c in graph_data["nets"] if c["id"] == "c1")
        c2_conn = next(c for c in graph_data["nets"] if c["id"] == "c2")
        
        # c1: input1(False) -> not1, so value should be False
        assert c1_conn["value"] == False
        # c2: not1(True) -> output1, so value should be True (NOT False = True)
        assert c2_conn["value"] == True
        
        # Test with input = True
        self.graph.set_input_value("input1", True)
        self.graph.simulate()
        
        graph_data = self.graph.to_slint_format()
        c1_conn = next(c for c in graph_data["nets"] if c["id"] == "c1")
        c2_conn = next(c for c in graph_data["nets"] if c["id"] == "c2")
        
        # c1: input1(True) -> not1, so value should be True
        assert c1_conn["value"] == True
        # c2: not1(False) -> output1, so value should be False (NOT True = False)
        assert c2_conn["value"] == False
    
    def test_visual_behavior_edit_mode(self):
        """Test visual behavior expectations in edit mode"""
        graph_data = self.graph.to_slint_format()
        
        # All nets should have simulation_mode = False
        for conn in graph_data["nets"]:
            assert conn["simulation_mode"] == False
            assert conn["has_value"] == False
            # This will result in black lines (2px) in the UI
        
        # Nodes should not have simulation values
        for node in graph_data["nodes"]:
            assert node["has_value"] == False
            # Input/output nodes will show default colors, logic gates will show gray
    
    def test_visual_behavior_simulation_mode(self):
        """Test visual behavior expectations in simulation mode"""
        self.graph.enter_simulation_mode()
        self.graph.set_input_value("input1", True)
        self.graph.simulate()
        
        graph_data = self.graph.to_slint_format()
        
        # All nets should have simulation_mode = True
        for conn in graph_data["nets"]:
            assert conn["simulation_mode"] == True
            assert conn["has_value"] == True
            # This will result in colored lines (4px) in the UI
        
        # Nodes should have simulation values
        for node in graph_data["nodes"]:
            assert node["has_value"] == True
            # Input/output nodes will show green/red, logic gates will show gray
    
    def test_different_node_types_net_values(self):
        """Test net values for different node types"""
        # Create a more complex circuit with different node types
        graph = GraphData()
        
        # Input nodes
        input_a = Node.create("input_a", NODE_REGISTRY.get_definition("input"), 50, 50)
        input_b = Node.create("input_b", NODE_REGISTRY.get_definition("input"), 50, 100)
        
        # Logic gates
        and_gate = Node.create("and1", NODE_REGISTRY.get_definition("and"), 150, 75)
        or_gate = Node.create("or1", NODE_REGISTRY.get_definition("or"), 250, 75)
        not_gate = Node.create("not1", NODE_REGISTRY.get_definition("not"), 350, 75)
        nor_gate = Node.create("nor1", NODE_REGISTRY.get_definition("nor"), 450, 75)
        
        # Output node
        output = Node.create("output1", NODE_REGISTRY.get_definition("output"), 550, 75)
        
        # Add all nodes
        for node in [input_a, input_b, and_gate, or_gate, not_gate, nor_gate, output]:
            graph.add_node(node)
        
        # Add nets
        nets = [
            Net.create_two_point("c1", "input_a", "out", "and1", "in1"),
            Net.create_two_point("c2", "input_b", "out", "and1", "in2"),
            Net.create_two_point("c3", "and1", "out", "or1", "in1"),
            Net.create_two_point("c4", "input_a", "out", "or1", "in2"),
            Net.create_two_point("c5", "or1", "out", "not1", "in"),
            Net.create_two_point("c6", "not1", "out", "nor1", "in1"),
            Net.create_two_point("c7", "input_b", "out", "nor1", "in2"),
            Net.create_two_point("c8", "nor1", "out", "output1", "in")
        ]
        
        for conn in nets:
            graph.add_net(conn)
        
        # Test in simulation mode
        graph.enter_simulation_mode()
        graph.set_input_value("input_a", True)
        graph.set_input_value("input_b", False)
        graph.simulate()
        
        graph_data = graph.to_slint_format()
        
        # Verify all nets have values
        for conn in graph_data["nets"]:
            assert conn["simulation_mode"] == True
            assert conn["has_value"] == True
            assert isinstance(conn["value"], bool)
        
        # Verify specific net values based on logic
        nets_by_id = {c["id"]: c for c in graph_data["nets"]}
        
        # c1: input_a(True) -> and1
        assert nets_by_id["c1"]["value"] == True
        # c2: input_b(False) -> and1
        assert nets_by_id["c2"]["value"] == False
        # c3: and1(True AND False = False) -> or1
        assert nets_by_id["c3"]["value"] == False
        # c4: input_a(True) -> or1
        assert nets_by_id["c4"]["value"] == True
        # c5: or1(False OR True = True) -> not1
        assert nets_by_id["c5"]["value"] == True
        # c6: not1(NOT True = False) -> nor1
        assert nets_by_id["c6"]["value"] == False
        # c7: input_b(False) -> nor1
        assert nets_by_id["c7"]["value"] == False
        # c8: nor1(False NOR False = True) -> output1
        assert nets_by_id["c8"]["value"] == True
    
    def test_edge_case_no_simulation_run(self):
        """Test edge case where simulation mode is entered but simulation not run"""
        self.graph.enter_simulation_mode()
        # Don't run simulation
        
        graph_data = self.graph.to_slint_format()
        
        # Should be in simulation mode but nets may not have values
        for conn in graph_data["nets"]:
            assert conn["simulation_mode"] == True
            # Nets may or may not have values depending on node state
    
    def test_edge_case_partial_simulation_values(self):
        """Test edge case where only some nodes have simulation values"""
        # Set some node values manually without full simulation
        self.graph.enter_simulation_mode()
        self.graph.nodes["input1"].value = True
        # Don't set other node values
        
        graph_data = self.graph.to_slint_format()
        
        # Should handle mixed value states gracefully
        for conn in graph_data["nets"]:
            assert conn["simulation_mode"] == True
            assert "value" in conn
            assert "has_value" in conn
            assert isinstance(conn["value"], bool)
            assert isinstance(conn["has_value"], bool)
    
    def test_edge_case_mode_transitions(self):
        """Test edge case of transitioning between edit and simulation modes"""
        # Start in edit mode
        graph_data = self.graph.to_slint_format()
        assert all(not conn["simulation_mode"] for conn in graph_data["nets"])
        
        # Enter simulation mode
        self.graph.enter_simulation_mode()
        self.graph.set_input_value("input1", True)
        self.graph.simulate()
        
        graph_data = self.graph.to_slint_format()
        assert all(conn["simulation_mode"] for conn in graph_data["nets"])
        assert all(conn["has_value"] for conn in graph_data["nets"])
        
        # Exit simulation mode (back to edit)
        self.graph.enter_edit_mode()
        
        graph_data = self.graph.to_slint_format()
        assert all(not conn["simulation_mode"] for conn in graph_data["nets"])
        assert all(not conn["has_value"] for conn in graph_data["nets"])
    
    def test_sr_nor_latch_net_values(self):
        """Test net values in SR NOR latch (feedback circuit)"""
        from logicsim.graph_data import create_sr_nor_latch_demo
        
        latch_graph = create_sr_nor_latch_demo()
        latch_graph.enter_simulation_mode()
        
        # Test SET operation
        latch_graph.set_input_value("set", True)
        latch_graph.set_input_value("reset", False)
        latch_graph.simulate()
        
        graph_data = latch_graph.to_slint_format()
        
        # All nets should have values in simulation mode
        for conn in graph_data["nets"]:
            assert conn["simulation_mode"] == True
            assert conn["has_value"] == True
            assert isinstance(conn["value"], bool)
        
        # Verify feedback loop nets have proper values
        nets_by_id = {c["id"]: c for c in graph_data["nets"]}
        
        # Should have feedback nets c3 and c4
        assert "c3" in nets_by_id  # nor1 -> nor2 (Q -> NOR2)
        assert "c4" in nets_by_id  # nor2 -> nor1 (Q̅ -> NOR1)
        
        # In SET state, Q=1, Q̅=0
        q_feedback = nets_by_id["c3"]["value"]    # Q feedback
        q_not_feedback = nets_by_id["c4"]["value"] # Q̅ feedback
        
        # These should be opposite values
        assert q_feedback != q_not_feedback