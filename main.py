#!/usr/bin/env python3
"""
LogicSim - A Python GUI application using Slint UI library
Main application entry point
"""

import sys
import slint
from pathlib import Path


def main():
    """Main application entry point"""
    
    # Get the path to the Slint UI file
    ui_path = Path(__file__).parent / "ui" / "main.slint"
    
    if not ui_path.exists():
        print(f"Error: UI file not found at {ui_path}")
        sys.exit(1)
    
    try:
        # Load the Slint UI file
        ui = slint.load_file(ui_path)
        
        # Print any diagnostics for debugging
        if hasattr(ui, 'diagnostics') and ui.diagnostics:
            print("Slint diagnostics:")
            for diagnostic in ui.diagnostics:
                print(f"  {diagnostic}")
        
        # Create the main window
        main_window = ui.MainWindow()
        
        # Run the application
        main_window.run()
        
    except Exception as e:
        print(f"Error loading or running the application: {e}")
        
        # If there are any Slint-specific diagnostics, print them
        try:
            if hasattr(slint, 'diagnostics'):
                print("Slint diagnostics:")
                for diagnostic in slint.diagnostics:
                    print(f"  {diagnostic}")
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()