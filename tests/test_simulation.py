"""
Unit tests for simulation.py module
"""

import pytest
from unittest.mock import patch
import logging

from logicsim.simulation import CircuitEvaluator
from logicsim.graph_data import Node, Connector


class TestCircuitEvaluator:
    """Test cases for the CircuitEvaluator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.evaluator = CircuitEvaluator()
    
    def test_evaluator_initialization(self):
        """Test that evaluator initializes correctly"""
        evaluator = CircuitEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'logger')
    
    def test_evaluate_and_gate_all_combinations(self):
        """Test AND gate with all possible input combinations"""
        # Test all truth table combinations for AND gate
        assert self.evaluator._evaluate_and_gate([False, False]) == False
        assert self.evaluator._evaluate_and_gate([False, True]) == False
        assert self.evaluator._evaluate_and_gate([True, False]) == False
        assert self.evaluator._evaluate_and_gate([True, True]) == True
    
    def test_evaluate_and_gate_invalid_input_count(self):
        """Test AND gate with invalid number of inputs"""
        # Too few inputs
        with pytest.raises(ValueError, match="AND gate requires exactly 2 inputs, got 0"):
            self.evaluator._evaluate_and_gate([])
        
        with pytest.raises(ValueError, match="AND gate requires exactly 2 inputs, got 1"):
            self.evaluator._evaluate_and_gate([True])
        
        # Too many inputs
        with pytest.raises(ValueError, match="AND gate requires exactly 2 inputs, got 3"):
            self.evaluator._evaluate_and_gate([True, False, True])
    
    def test_evaluate_or_gate_all_combinations(self):
        """Test OR gate with all possible input combinations"""
        # Test all truth table combinations for OR gate
        assert self.evaluator._evaluate_or_gate([False, False]) == False
        assert self.evaluator._evaluate_or_gate([False, True]) == True
        assert self.evaluator._evaluate_or_gate([True, False]) == True
        assert self.evaluator._evaluate_or_gate([True, True]) == True
    
    def test_evaluate_or_gate_invalid_input_count(self):
        """Test OR gate with invalid number of inputs"""
        # Too few inputs
        with pytest.raises(ValueError, match="OR gate requires exactly 2 inputs, got 0"):
            self.evaluator._evaluate_or_gate([])
        
        with pytest.raises(ValueError, match="OR gate requires exactly 2 inputs, got 1"):
            self.evaluator._evaluate_or_gate([True])
        
        # Too many inputs
        with pytest.raises(ValueError, match="OR gate requires exactly 2 inputs, got 3"):
            self.evaluator._evaluate_or_gate([True, False, True])
    
    def test_evaluate_not_gate_both_cases(self):
        """Test NOT gate with both possible inputs"""
        assert self.evaluator._evaluate_not_gate([False]) == True
        assert self.evaluator._evaluate_not_gate([True]) == False
    
    def test_evaluate_not_gate_invalid_input_count(self):
        """Test NOT gate with invalid number of inputs"""
        # Too few inputs
        with pytest.raises(ValueError, match="NOT gate requires exactly 1 input, got 0"):
            self.evaluator._evaluate_not_gate([])
        
        # Too many inputs
        with pytest.raises(ValueError, match="NOT gate requires exactly 1 input, got 2"):
            self.evaluator._evaluate_not_gate([True, False])
    
    def test_evaluate_node_and_gate(self):
        """Test evaluate_node method with AND gate"""
        # Create an AND gate node
        connectors = [
            Connector("in1", -4.0, 11.0, True),
            Connector("in2", -4.0, 41.0, True),
            Connector("out", 76.0, 26.0, False)
        ]
        and_node = Node("and1", "and", 100.0, 100.0, 80.0, 60.0, "AND", "gray", connectors)
        
        # Test all combinations
        assert self.evaluator.evaluate_node(and_node, [False, False]) == False
        assert self.evaluator.evaluate_node(and_node, [False, True]) == False
        assert self.evaluator.evaluate_node(and_node, [True, False]) == False
        assert self.evaluator.evaluate_node(and_node, [True, True]) == True
    
    def test_evaluate_node_or_gate(self):
        """Test evaluate_node method with OR gate"""
        # Create an OR gate node
        connectors = [
            Connector("in1", -4.0, 11.0, True),
            Connector("in2", -4.0, 41.0, True),
            Connector("out", 76.0, 26.0, False)
        ]
        or_node = Node("or1", "or", 100.0, 100.0, 80.0, 60.0, "OR", "gray", connectors)
        
        # Test all combinations
        assert self.evaluator.evaluate_node(or_node, [False, False]) == False
        assert self.evaluator.evaluate_node(or_node, [False, True]) == True
        assert self.evaluator.evaluate_node(or_node, [True, False]) == True
        assert self.evaluator.evaluate_node(or_node, [True, True]) == True
    
    def test_evaluate_node_not_gate(self):
        """Test evaluate_node method with NOT gate"""
        # Create a NOT gate node
        connectors = [
            Connector("in", -4.0, 26.0, True),
            Connector("out", 76.0, 26.0, False)
        ]
        not_node = Node("not1", "not", 100.0, 100.0, 80.0, 60.0, "NOT", "gray", connectors)
        
        # Test both cases
        assert self.evaluator.evaluate_node(not_node, [False]) == True
        assert self.evaluator.evaluate_node(not_node, [True]) == False
    
    def test_evaluate_node_input_gate(self):
        """Test evaluate_node method with input node"""
        # Create an input node with value set
        connectors = [Connector("out", 46.0, 21.0, False)]
        input_node = Node("input1", "input", 50.0, 50.0, 50.0, 50.0, "A", "green", connectors, True)
        
        # Input node should return its own value, ignoring input_values
        assert self.evaluator.evaluate_node(input_node, []) == True
        
        # Test with False value
        input_node.value = False
        assert self.evaluator.evaluate_node(input_node, []) == False
    
    def test_evaluate_node_input_gate_no_value(self):
        """Test evaluate_node method with input node that has no value set"""
        # Create an input node without value
        connectors = [Connector("out", 46.0, 21.0, False)]
        input_node = Node("input1", "input", 50.0, 50.0, 50.0, 50.0, "A", "green", connectors, None)
        
        # Should raise error when value is None
        with pytest.raises(ValueError, match="Input node 'input1' has no value set"):
            self.evaluator.evaluate_node(input_node, [])
    
    def test_evaluate_node_output_gate(self):
        """Test evaluate_node method with output node"""
        # Create an output node
        connectors = [Connector("in", -4.0, 21.0, True)]
        output_node = Node("output1", "output", 300.0, 100.0, 50.0, 50.0, "OUT", "pink", connectors)
        
        # Output node should pass through input value
        assert self.evaluator.evaluate_node(output_node, [True]) == True
        assert self.evaluator.evaluate_node(output_node, [False]) == False
    
    def test_evaluate_node_output_gate_invalid_input_count(self):
        """Test evaluate_node method with output node and invalid input count"""
        # Create an output node
        connectors = [Connector("in", -4.0, 21.0, True)]
        output_node = Node("output1", "output", 300.0, 100.0, 50.0, 50.0, "OUT", "pink", connectors)
        
        # Should require exactly one input
        with pytest.raises(ValueError, match="Output node requires exactly 1 input, got 0"):
            self.evaluator.evaluate_node(output_node, [])
        
        with pytest.raises(ValueError, match="Output node requires exactly 1 input, got 2"):
            self.evaluator.evaluate_node(output_node, [True, False])
    
    def test_evaluate_node_unsupported_type(self):
        """Test evaluate_node method with unsupported node type"""
        # Create a node with unsupported type
        connectors = [Connector("out", 46.0, 21.0, False)]
        unknown_node = Node("unknown1", "xor", 100.0, 100.0, 80.0, 60.0, "XOR", "blue", connectors)
        
        # Should raise error for unsupported type
        with pytest.raises(ValueError, match="Unsupported node type: xor"):
            self.evaluator.evaluate_node(unknown_node, [True, False])
    
    def test_logging_behavior(self):
        """Test that gate evaluation logs debug information"""
        with patch.object(self.evaluator, 'logger') as mock_logger:
            # Test AND gate logging
            self.evaluator._evaluate_and_gate([True, False])
            mock_logger.debug.assert_called_with("AND gate: [True, False] -> False")
            
            # Test OR gate logging
            mock_logger.reset_mock()
            self.evaluator._evaluate_or_gate([True, False])
            mock_logger.debug.assert_called_with("OR gate: [True, False] -> True")
            
            # Test NOT gate logging
            mock_logger.reset_mock()
            self.evaluator._evaluate_not_gate([True])
            mock_logger.debug.assert_called_with("NOT gate: [True] -> False")
    
    def test_gate_evaluation_deterministic(self):
        """Test that gate evaluation is deterministic (same inputs -> same outputs)"""
        # Test multiple evaluations of same inputs produce same results
        inputs = [True, False]
        
        result1 = self.evaluator._evaluate_and_gate(inputs)
        result2 = self.evaluator._evaluate_and_gate(inputs)
        result3 = self.evaluator._evaluate_and_gate(inputs)
        
        assert result1 == result2 == result3 == False
        
        result1 = self.evaluator._evaluate_or_gate(inputs)
        result2 = self.evaluator._evaluate_or_gate(inputs)
        result3 = self.evaluator._evaluate_or_gate(inputs)
        
        assert result1 == result2 == result3 == True
    
    def test_integration_scenario(self):
        """Test a complete integration scenario with multiple gate types"""
        # Create nodes for a small circuit: A AND B -> NOT -> Output
        connectors_input = [Connector("out", 46.0, 21.0, False)]
        input_a = Node("a", "input", 0.0, 0.0, 50.0, 50.0, "A", "green", connectors_input, True)
        input_b = Node("b", "input", 0.0, 50.0, 50.0, 50.0, "B", "green", connectors_input, False)
        
        connectors_and = [
            Connector("in1", -4.0, 11.0, True),
            Connector("in2", -4.0, 41.0, True),
            Connector("out", 76.0, 26.0, False)
        ]
        and_gate = Node("and1", "and", 100.0, 25.0, 80.0, 60.0, "AND", "gray", connectors_and)
        
        connectors_not = [
            Connector("in", -4.0, 26.0, True),
            Connector("out", 76.0, 26.0, False)
        ]
        not_gate = Node("not1", "not", 200.0, 25.0, 80.0, 60.0, "NOT", "gray", connectors_not)
        
        connectors_output = [Connector("in", -4.0, 21.0, True)]
        output_node = Node("out", "output", 300.0, 25.0, 50.0, 50.0, "OUT", "pink", connectors_output)
        
        # Evaluate the circuit step by step
        # Input values: A=True, B=False
        a_value = self.evaluator.evaluate_node(input_a, [])  # Should be True
        b_value = self.evaluator.evaluate_node(input_b, [])  # Should be False
        
        assert a_value == True
        assert b_value == False
        
        # AND gate: True AND False = False
        and_result = self.evaluator.evaluate_node(and_gate, [a_value, b_value])
        assert and_result == False
        
        # NOT gate: NOT False = True
        not_result = self.evaluator.evaluate_node(not_gate, [and_result])
        assert not_result == True
        
        # Output: pass through True
        final_result = self.evaluator.evaluate_node(output_node, [not_result])
        assert final_result == True