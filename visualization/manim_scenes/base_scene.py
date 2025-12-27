"""
Base Manim scene with branding and common utilities.
"""

from manim import *
import sys
from pathlib import Path

# Add parent directory to path to import palette
sys.path.append(str(Path(__file__).parent.parent.parent))
from visualization.palette import *


class BaseVisualizationScene(Scene):
    """
    Base scene with green branding and common utilities.
    All visualization scenes should inherit from this.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_branding()
    
    def setup_branding(self):
        """Setup branding colors and styles."""
        self.background_color = BACKGROUND_DARK
        self.primary_color = PRIMARY_GREEN
        self.secondary_color = SECONDARY_GREEN
        self.text_color = TEXT_WHITE
    
    def create_title(self, text: str, font_size: int = FONT_SIZE_TITLE) -> Text:
        """
        Create a title text with branding.
        
        Args:
            text: Title text
            font_size: Font size
            
        Returns:
            Text object
        """
        title = Text(
            text,
            font_size=font_size,
            color=self.primary_color,
            weight=BOLD,
        )
        return title
    
    def create_label(self, text: str, font_size: int = FONT_SIZE_LABEL) -> Text:
        """
        Create a label text.
        
        Args:
            text: Label text
            font_size: Font size
            
        Returns:
            Text object
        """
        label = Text(
            text,
            font_size=font_size,
            color=self.text_color,
        )
        return label
    
    def create_axes_2d(
        self,
        x_range: tuple = (-3, 3),
        y_range: tuple = (-3, 3),
        axis_config: dict = None,
    ) -> Axes:
        """
        Create 2D axes with green branding.
        
        Args:
            x_range: X-axis range
            y_range: Y-axis range
            axis_config: Additional axis configuration
            
        Returns:
            Axes object
        """
        if axis_config is None:
            axis_config = {}
        
        default_config = {
            'axis_config': {
                'color': PRIMARY_GREEN,
                'stroke_width': AXIS_WIDTH,
            },
            'x_axis_config': {
                'color': PRIMARY_GREEN,
            },
            'y_axis_config': {
                'color': PRIMARY_GREEN,
            },
            'background_line_style': {
                'stroke_color': GRID_COLOR,
                'stroke_width': 1,
                'stroke_opacity': GRID_OPACITY,
            },
        }
        default_config.update(axis_config)
        
        axes = Axes(
            x_range=x_range,
            y_range=y_range,
            **default_config,
        )
        return axes

