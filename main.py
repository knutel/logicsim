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
        
        logger.debug(f"Graph data: {len(graph_data['nodes'])} nodes, {len(graph_data['connections'])} connections")
        
        # Set graph data properties in Slint
        logger.debug("Graph nodes:")
        for node in graph_data['nodes']:
            logger.debug(f"  {node['id']}: {node['node_type']} at ({node['x']}, {node['y']})")
        
        logger.debug("Graph connections:")
        for conn in graph_data['connections']:
            logger.debug(f"  {conn['id']}: ({conn['start_x']}, {conn['start_y']}) -> ({conn['end_x']}, {conn['end_y']})")
        
        # Convert Python data to Slint-compatible format
        # Convert to simple dictionaries with proper types
        slint_nodes = []
        for node in graph_data['nodes']:
            # Create a properly formatted dictionary for Slint struct
            slint_node = {
                "id": str(node['id']),
                "node_type": str(node['node_type']),
                "x": float(node['x']),
                "y": float(node['y']),
                "width": float(node['width']),
                "height": float(node['height']),
                "label": str(node['label'])
            }
            slint_nodes.append(slint_node)
        
        slint_connections = []
        for conn in graph_data['connections']:
            # Create a properly formatted dictionary for Slint struct
            slint_conn = {
                "id": str(conn['id']),
                "start_x": float(conn['start_x']),
                "start_y": float(conn['start_y']),
                "end_x": float(conn['end_x']),
                "end_y": float(conn['end_y'])
            }
            slint_connections.append(slint_conn)
        
        main_window.nodes = slint.ListModel(slint_nodes)
        main_window.connections = slint.ListModel(slint_connections)

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