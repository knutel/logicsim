#!/usr/bin/env python3
"""
LogicSim - A Python GUI application using Slint UI library
Main application entry point
"""

import sys
import os
import logging
import slint
from pathlib import Path
from logicsim.graph_data import create_demo_graph


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
        logger.debug(f"Refreshing UI - Selected nodes: {graph_data['selected_nodes']}")
        
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
                "connectors": slint.ListModel(slint_connectors)
            }
            slint_nodes.append(slint_node)
        
        # Convert connections to Slint format
        slint_connections = []
        for conn in graph_data['connections']:
            slint_conn = {
                "id": str(conn['id']),
                "start_x": float(conn['start_x']),
                "start_y": float(conn['start_y']),
                "end_x": float(conn['end_x']),
                "end_y": float(conn['end_y'])
            }
            slint_connections.append(slint_conn)
        
        # Update UI properties
        main_window.nodes = slint.ListModel(slint_nodes)
        main_window.connections = slint.ListModel(slint_connections)
        main_window.selected_nodes = slint.ListModel([str(node_id) for node_id in graph_data['selected_nodes']])
    
    def handle_graph_click(x, y):
        """Handle mouse click on graph area"""
        if graph is None:
            return
            
        logger.debug(f"Graph clicked at ({x}, {y})")
        
        # Let Python handle the click logic
        selection_changed = graph.handle_mouse_click(x, y)
        
        if selection_changed:
            logger.debug(f"Selection changed to: {graph.get_selected_node()}")
            refresh_ui_data()
        else:
            logger.debug("Selection unchanged")
    
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
        logger.info("Creating demo graph data")
        graph = create_demo_graph()
        graph_data = graph.to_slint_format()
        
        # Connect the click callback
        main_window.graph_clicked = handle_graph_click
        
        logger.debug(f"Graph data: {len(graph_data['nodes'])} nodes, {len(graph_data['connections'])} connections")
        
        # Set graph data properties in Slint
        logger.debug("Graph nodes:")
        for node in graph_data['nodes']:
            logger.debug(f"  {node['id']}: {node['node_type']} at ({node['x']}, {node['y']})")
        
        logger.debug("Graph connections:")
        for conn in graph_data['connections']:
            logger.debug(f"  {conn['id']}: ({conn['start_x']}, {conn['start_y']}) -> ({conn['end_x']}, {conn['end_y']})")
        
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