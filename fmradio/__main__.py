#!/usr/bin/env python3
"""Main entry point for the FM Radio application."""

import os
from pathlib import Path

from .ui.app import RadioApp
from .ui.styles import RADIO_APP_CSS

def main():
    """Run the FM Radio application."""
    # Write CSS file
    css_path = Path("radio_app_styles.css")
    try:
        with open(css_path, "w") as css_file:
            css_file.write(RADIO_APP_CSS)
    except IOError as e:
        print(f"Warning: Could not write CSS file '{css_path}': {e}")
        print("The application will run without custom styles if the file is missing or unwritable.")

    # Run the application
    app = RadioApp()
    app.run()

if __name__ == "__main__":
    main() 