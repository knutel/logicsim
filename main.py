#!/usr/bin/env python3
"""
LogicSim - A Python GUI application using Slint UI library
Main application entry point
"""

import sys
import os
import logging
import time
import slint
from pathlib import Path
from logicsim.graph_data import create_sr_nor_latch_demo


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def setup_logging():
    """Setup logging configuration"""
    # Get log level from environment variable, default to INFO
    log_level = os.getenv('LOGICSIM_LOG_LEVEL', 'INFO').upper()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level: {log_level}")
    return logger


def main():
    """Main application entry point"""
    logger = setup_logging()
    logger.info("Starting LogicSim application")
    
    # Graph instance and main window that will be used by the callback
    graph = None
    main_window = None
    
    def refresh_ui_data():
        """Refresh UI with current graph state"""
        if graph is None or main_window is None:
            return
            
        graph_data = graph.to_slint_format()
        logger.debug(f"Refreshing UI - Selected nodes: {graph_data['selected_nodes']}, Selected nets: {graph_data['selected_nets']}")
        
        # Convert nodes to Slint format
        slint_nodes = []
        for node in graph_data['nodes']:
            slint_connectors = []
            for connector in node['connectors']:
                slint_connector = {
                    "id": str(connector['id']),
                    "x": float(connector['x']),
                    "y": float(connector['y']),
                    "is_input": bool(connector['is_input'])
                }
                slint_connectors.append(slint_connector)
            
            slint_node = {
                "id": str(node['id']),
                "node_type": str(node['node_type']),
                "x": float(node['x']),
                "y": float(node['y']),
                "width": float(node['width']),
                "height": float(node['height']),
                "label": str(node['label']),
                "color": str(node['color']),
                "connectors": slint.ListModel(slint_connectors),
                "value": bool(node['value']) if node['value'] is not None else False,
                "has_value": node['value'] is not None
            }
            slint_nodes.append(slint_node)
        
        # Convert nets to Slint format
        slint_nets = []
        for net in graph_data['nets']:
            # Convert segments to Slint format
            slint_segments = []
            for segment in net['segments']:
                slint_segment = {
                    "start_x": float(segment['start_x']),
                    "start_y": float(segment['start_y']),
                    "end_x": float(segment['end_x']),
                    "end_y": float(segment['end_y'])
                }
                slint_segments.append(slint_segment)
            
            slint_net = {
                "id": str(net['id']),
                "segments": slint.ListModel(slint_segments),
                "value": bool(net['value']),
                "has_value": bool(net['has_value']),
                "simulation_mode": bool(net['simulation_mode'])
            }
            slint_nets.append(slint_net)
        
        # Update UI properties
        main_window.nodes = slint.ListModel(slint_nodes)
        main_window.nets = slint.ListModel(slint_nets)
        main_window.selected_nodes = slint.ListModel([str(node_id) for node_id in graph_data['selected_nodes']])
        main_window.selected_nets = slint.ListModel([str(net_id) for net_id in graph_data['selected_nets']])
        main_window.editing_node_id = graph_data['editing_node_id']
        main_window.editing_text = graph_data['editing_text']
        main_window.creating_net = graph_data['creating_net']
        main_window.pending_start_x = graph_data['pending_start_x']
        main_window.pending_start_y = graph_data['pending_start_y']
        main_window.pending_end_x = graph_data['pending_end_x']
        main_window.pending_end_y = graph_data['pending_end_y']
        
        # Convert toolbox items to Slint format
        slint_toolbox_items = []
        for item in graph_data['toolbox_items']:
            slint_item = {
                "node_type": str(item['node_type']),
                "label": str(item['label']),
                "color": str(item['color']),
                "is_selected": bool(item['is_selected'])
            }
            slint_toolbox_items.append(slint_item)
        
        main_window.toolbox_items = slint.ListModel(slint_toolbox_items)
        main_window.toolbox_creation_mode = graph_data['toolbox_creation_mode']
        main_window.simulation_mode = graph_data['simulation_mode']
    
    def handle_graph_pointer_event(kind, x, y):
        """Handle pointer event on graph area"""
        if graph is None:
            return
            
        logger.debug(f"Graph pointer event at ({x}, {y}) - kind: {kind}")
        
        # Dispatch based on event type
        ui_needs_refresh = False
        
        if kind == "down":
            # Check if we're in toolbox creation mode first
            if graph.toolbox_creation_mode and graph.selected_node_type:
                # Create new node at click position
                ui_needs_refresh = graph.create_node_at_position(graph.selected_node_type, x, y)
            else:
                # Check for double-click
                current_time = time.time()
                if graph.is_double_click(current_time):
                    # Handle double-click for label editing
                    ui_needs_refresh = graph.handle_double_click(x, y)
                else:
                    # Handle single click for selection/movement/connection creation
                    ui_needs_refresh = graph.handle_pointer_down(x, y)
        elif kind == "move":
            ui_needs_refresh = graph.handle_pointer_move(x, y)
        elif kind == "up":
            ui_needs_refresh = graph.handle_pointer_up(x, y)
        else:
            logger.warning(f"Unknown pointer event kind: {kind}")
            return
        
        # Refresh UI if needed
        if ui_needs_refresh:
            if graph.toolbox_creation_mode:
                logger.debug(f"Refreshing UI - Toolbox creation mode, selected type: {graph.selected_node_type}")
            else:
                logger.debug(f"Refreshing UI - Selected node: {graph.get_selected_node()}, Selected net: {graph.get_selected_net()}")
            refresh_ui_data()
        else:
            logger.debug("UI refresh not needed")
    
    # Get the path to the Slint UI file
    ui_path = Path(__file__).parent / "ui" / "main.slint"
    logger.debug(f"UI file path: {ui_path}")
    
    if not ui_path.exists():
        logger.error(f"UI file not found at {ui_path}")
        sys.exit(1)
    
    try:
        logger.info("Loading Slint UI file")
        # Load the Slint UI file
        ui = slint.load_file(ui_path)
        
        # Print any diagnostics for debugging
        if hasattr(ui, 'diagnostics') and ui.diagnostics:
            logger.warning("Slint diagnostics found:")
            for diagnostic in ui.diagnostics:
                logger.warning(f"  {diagnostic}")
        
        logger.info("Creating main window")
        # Create the main window
        main_window = ui.MainWindow()
        
        # Create and set up graph data
        logger.info("Creating SR NOR latch demo graph data")
        graph = create_sr_nor_latch_demo()
        graph_data = graph.to_slint_format()
        
        # Connect the pointer event callback
        main_window.graph_pointer_event = handle_graph_pointer_event
        
        # Connect label editing callbacks
        def handle_label_edit_completed(node_id, new_label):
            """Handle completion of label editing"""
            if graph is None:
                return
            
            logger.debug(f"Label edit completed for node {node_id}: '{new_label}'")
            ui_needs_refresh = graph.complete_label_edit(node_id, new_label)
            
            if ui_needs_refresh:
                refresh_ui_data()
        
        def handle_label_edit_changed(text):
            """Handle changes to editing text"""
            if graph is None:
                return
            
            graph.update_editing_text(text)
        
        main_window.label_edit_completed = handle_label_edit_completed
        main_window.label_edit_changed = handle_label_edit_changed
        
        # Connect delete callback
        def handle_delete_clicked():
            """Handle delete button click"""
            if graph is None:
                return
            
            logger.debug("Delete button clicked")
            something_deleted = graph.delete_selected()
            
            if something_deleted:
                logger.debug("Item deleted, refreshing UI")
                refresh_ui_data()
            else:
                logger.debug("No item selected for deletion")
        
        main_window.delete_clicked = handle_delete_clicked
        
        # Connect toolbox callback
        def handle_toolbox_node_type_clicked(node_type):
            """Handle toolbox node type selection"""
            if graph is None:
                return
            
            logger.debug(f"Toolbox node type clicked: {node_type}")
            
            # If the same node type is already selected, deselect it
            if graph.selected_node_type == node_type:
                ui_needs_refresh = graph.deselect_toolbox_node_type()
            else:
                # Select the new node type
                ui_needs_refresh = graph.select_toolbox_node_type(node_type)
            
            if ui_needs_refresh:
                refresh_ui_data()
        
        main_window.toolbox_node_type_clicked = handle_toolbox_node_type_clicked
        
        # Connect mode transition callbacks
        def handle_enter_simulation_clicked():
            """Handle enter simulation mode button click"""
            if graph is None:
                return
            
            logger.info("Enter simulation mode button clicked")
            ui_needs_refresh = graph.enter_simulation_mode()
            
            if ui_needs_refresh:
                logger.debug("Entered simulation mode, refreshing UI")
                refresh_ui_data()
        
        def handle_enter_edit_clicked():
            """Handle enter edit mode button click"""
            if graph is None:
                return
            
            logger.info("Enter edit mode button clicked")
            ui_needs_refresh = graph.enter_edit_mode()
            
            if ui_needs_refresh:
                logger.debug("Entered edit mode, refreshing UI")
                refresh_ui_data()
        
        main_window.enter_simulation_clicked = handle_enter_simulation_clicked
        main_window.enter_edit_clicked = handle_enter_edit_clicked
        
        logger.debug(f"Graph data: {len(graph_data['nodes'])} nodes, {len(graph_data['nets'])} nets")
        
        # Set graph data properties in Slint
        logger.debug("Graph nodes:")
        for node in graph_data['nodes']:
            logger.debug(f"  {node['id']}: {node['node_type']} at ({node['x']}, {node['y']})")
        
        logger.debug("Graph nets:")
        for net in graph_data['nets']:
            logger.debug(f"  {net['id']}: {len(net['segments'])} segments")
        
        # Use the refresh function to set initial data
        refresh_ui_data()

        # Log any callbacks or property changes
        logger.debug("Main window created successfully")
        
        logger.info("Starting application main loop")
        # Run the application
        main_window.run()
        
    except Exception as e:
        logger.error(f"Error loading or running the application: {e}")
        
        # Check if the exception contains diagnostic information
        if hasattr(e, 'args') and len(e.args) > 1 and hasattr(e.args[1], '__iter__'):
            try:
                logger.error("Slint compilation diagnostics:")
                for diagnostic in e.args[1]:
                    logger.error(f"  {diagnostic}")
            except:
                pass
        
        # If there are any Slint-specific diagnostics, print them
        try:
            if hasattr(slint, 'diagnostics'):
                logger.error("Slint diagnostics:")
                for diagnostic in slint.diagnostics:
                    logger.error(f"  {diagnostic}")
        except:
            pass
        
        sys.exit(1)
    
    logger.info("Application terminated")


if __name__ == "__main__":
    main()