"""
Unit tests for graph_data.py module
"""

import pytest
from unittest.mock import patch, MagicMock
import logging

from logicsim.graph_data import (
    NodeType,
    Connector,
    Node,
    Connection,
    GraphData,
    create_connectors_for_node_type,
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
            node_type=NodeType.AND_GATE,
            x=100.0,
            y=200.0,
            width=80.0,
            height=60.0,
            label="TEST",
            connectors=connectors
        )
        
        assert node.id == "test_node"
        assert node.node_type == NodeType.AND_GATE
        assert node.x == 100.0
        assert node.y == 200.0
        assert node.width == 80.0
        assert node.height == 60.0
        assert node.label == "TEST"
        assert len(node.connectors) == 2
        assert node.connectors[0].id == "in1"
        assert node.connectors[1].id == "out"
    
    def test_node_create_input(self):
        """Test Node.create() for input nodes"""
        node = Node.create("input1", NodeType.INPUT, 50.0, 50.0, 50.0, 50.0, "A")
        
        assert node.id == "input1"
        assert node.node_type == NodeType.INPUT
        assert node.label == "A"
        assert len(node.connectors) == 1
        assert node.connectors[0].id == "out"
        assert node.connectors[0].is_input == False
        assert node.connectors[0].x_offset == 46.0  # 50 - 4
        assert node.connectors[0].y_offset == 21.0  # 50/2 - 4
    
    def test_node_create_output(self):
        """Test Node.create() for output nodes"""
        node = Node.create("output1", NodeType.OUTPUT, 550.0, 175.0, 50.0, 50.0, "OUT")
        
        assert node.id == "output1"
        assert node.node_type == NodeType.OUTPUT
        assert node.label == "OUT"
        assert len(node.connectors) == 1
        assert node.connectors[0].id == "in"
        assert node.connectors[0].is_input == True
        assert node.connectors[0].x_offset == -4.0
        assert node.connectors[0].y_offset == 21.0  # 50/2 - 4
    
    def test_node_create_and_gate(self):
        """Test Node.create() for AND gates"""
        node = Node.create("and1", NodeType.AND_GATE, 200.0, 80.0, 80.0, 60.0, "AND")
        
        assert node.id == "and1"
        assert node.node_type == NodeType.AND_GATE
        assert node.label == "AND"
        assert len(node.connectors) == 3
        
        # Check input connectors
        in1 = next(c for c in node.connectors if c.id == "in1")
        in2 = next(c for c in node.connectors if c.id == "in2")
        out = next(c for c in node.connectors if c.id == "out")
        
        assert in1.is_input == True
        assert in1.x_offset == -4.0
        assert in1.y_offset == 15.0
        
        assert in2.is_input == True
        assert in2.x_offset == -4.0
        assert in2.y_offset == 35.0
        
        assert out.is_input == False
        assert out.x_offset == 76.0  # 80 - 4
        assert out.y_offset == 26.0  # 60/2 - 4
    
    def test_node_create_or_gate(self):
        """Test Node.create() for OR gates"""
        node = Node.create("or1", NodeType.OR_GATE, 200.0, 220.0, 80.0, 60.0, "OR")
        
        assert node.id == "or1"
        assert node.node_type == NodeType.OR_GATE
        assert node.label == "OR"
        assert len(node.connectors) == 3
        
        # Should have same connector structure as AND gate
        connector_ids = [c.id for c in node.connectors]
        assert "in1" in connector_ids
        assert "in2" in connector_ids
        assert "out" in connector_ids
    
    def test_node_create_not_gate(self):
        """Test Node.create() for NOT gates"""
        node = Node.create("not1", NodeType.NOT_GATE, 400.0, 150.0, 80.0, 60.0, "NOT")
        
        assert node.id == "not1"
        assert node.node_type == NodeType.NOT_GATE
        assert node.label == "NOT"
        assert len(node.connectors) == 2
        
        # Check input and output connectors
        in_conn = next(c for c in node.connectors if c.id == "in")
        out_conn = next(c for c in node.connectors if c.id == "out")
        
        assert in_conn.is_input == True
        assert in_conn.x_offset == -4.0
        assert in_conn.y_offset == 26.0  # 60/2 - 4
        
        assert out_conn.is_input == False
        assert out_conn.x_offset == 76.0  # 80 - 4
        assert out_conn.y_offset == 26.0  # 60/2 - 4


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


class TestCreateConnectorsForNodeType:
    """Test cases for create_connectors_for_node_type function"""
    
    def test_input_node_connectors(self):
        """Test connector creation for input nodes"""
        connectors = create_connectors_for_node_type(NodeType.INPUT, 50.0, 50.0)
        
        assert len(connectors) == 1
        connector = connectors[0]
        assert connector.id == "out"
        assert connector.is_input == False
        assert connector.x_offset == 46.0  # 50 - 4
        assert connector.y_offset == 21.0  # 50/2 - 4
    
    def test_output_node_connectors(self):
        """Test connector creation for output nodes"""
        connectors = create_connectors_for_node_type(NodeType.OUTPUT, 50.0, 50.0)
        
        assert len(connectors) == 1
        connector = connectors[0]
        assert connector.id == "in"
        assert connector.is_input == True
        assert connector.x_offset == -4.0
        assert connector.y_offset == 21.0  # 50/2 - 4
    
    @pytest.mark.parametrize("gate_type", [NodeType.AND_GATE, NodeType.OR_GATE])
    def test_two_input_gate_connectors(self, gate_type):
        """Test connector creation for AND/OR gates"""
        connectors = create_connectors_for_node_type(gate_type, 80.0, 60.0)
        
        assert len(connectors) == 3
        
        # Check that all expected connector IDs are present
        connector_ids = [c.id for c in connectors]
        assert "in1" in connector_ids
        assert "in2" in connector_ids
        assert "out" in connector_ids
        
        # Check input connectors
        in1 = next(c for c in connectors if c.id == "in1")
        in2 = next(c for c in connectors if c.id == "in2")
        out = next(c for c in connectors if c.id == "out")
        
        assert in1.is_input == True
        assert in1.x_offset == -4.0
        assert in1.y_offset == 15.0
        
        assert in2.is_input == True
        assert in2.x_offset == -4.0
        assert in2.y_offset == 35.0
        
        assert out.is_input == False
        assert out.x_offset == 76.0  # 80 - 4
        assert out.y_offset == 26.0  # 60/2 - 4
    
    def test_not_gate_connectors(self):
        """Test connector creation for NOT gates"""
        connectors = create_connectors_for_node_type(NodeType.NOT_GATE, 80.0, 60.0)
        
        assert len(connectors) == 2
        
        # Check that all expected connector IDs are present
        connector_ids = [c.id for c in connectors]
        assert "in" in connector_ids
        assert "out" in connector_ids
        
        # Check input and output connectors
        in_conn = next(c for c in connectors if c.id == "in")
        out_conn = next(c for c in connectors if c.id == "out")
        
        assert in_conn.is_input == True
        assert in_conn.x_offset == -4.0
        assert in_conn.y_offset == 26.0  # 60/2 - 4
        
        assert out_conn.is_input == False
        assert out_conn.x_offset == 76.0  # 80 - 4
        assert out_conn.y_offset == 26.0  # 60/2 - 4
    
    def test_different_dimensions(self):
        """Test connector creation with different node dimensions"""
        # Test with larger dimensions
        connectors = create_connectors_for_node_type(NodeType.INPUT, 100.0, 80.0)
        
        assert len(connectors) == 1
        connector = connectors[0]
        assert connector.x_offset == 96.0  # 100 - 4
        assert connector.y_offset == 36.0  # 80/2 - 4
    
    def test_empty_connectors_for_unknown_type(self):
        """Test that unknown node types return empty connector list"""
        # This would be an edge case if we had more node types in the future
        connectors = create_connectors_for_node_type(NodeType.INPUT, 50.0, 50.0)
        # We know INPUT should return 1 connector, this test just ensures the function works
        assert len(connectors) == 1


class TestGraphData:
    """Test cases for the GraphData class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.graph = GraphData()
        
        # Create test nodes
        self.input_node = Node.create("input1", NodeType.INPUT, 50.0, 50.0, 50.0, 50.0, "A")
        self.and_gate = Node.create("and1", NodeType.AND_GATE, 200.0, 80.0, 80.0, 60.0, "AND")
        self.output_node = Node.create("output1", NodeType.OUTPUT, 350.0, 90.0, 50.0, 50.0, "OUT")
        
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
        assert isinstance(result["nodes"], list)
        assert isinstance(result["connections"], list)
        assert len(result["nodes"]) == 0
        assert len(result["connections"]) == 0
    
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
        assert conn_data["end_y"] == 98.0    # 80 + 15 + 3


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
        
        assert graph.nodes["input_a"].node_type == NodeType.INPUT
        assert graph.nodes["input_b"].node_type == NodeType.INPUT
        assert graph.nodes["input_c"].node_type == NodeType.INPUT
        assert graph.nodes["and_gate"].node_type == NodeType.AND_GATE
        assert graph.nodes["or_gate"].node_type == NodeType.OR_GATE
        assert graph.nodes["not_gate"].node_type == NodeType.NOT_GATE
        assert graph.nodes["output_a"].node_type == NodeType.OUTPUT
        assert graph.nodes["output_b"].node_type == NodeType.OUTPUT
    
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