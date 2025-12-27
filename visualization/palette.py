"""
Color palette and styling constants for visualizations.
Green-based branding palette for TikTok/YouTube content.
"""

# Primary Green Branding Colors
PRIMARY_GREEN = "#00FF88"
SECONDARY_GREEN = "#00CC6A"
ACCENT_GREEN = "#33FFAA"
DARK_GREEN = "#009955"

# Background Colors
BACKGROUND_DARK = "#0A0A0A"
BACKGROUND_DARKER = "#050505"
BACKGROUND_LIGHT = "#1A1A1A"

# Text Colors
TEXT_WHITE = "#FFFFFF"
TEXT_LIGHT_GRAY = "#E0E0E0"
TEXT_GRAY = "#A0A0A0"

# Complementary Colors (for class differentiation)
COMPLEMENTARY_COLORS = [
    "#FF6B9D",  # Pink
    "#4ECDC4",  # Teal
    "#FFE66D",  # Yellow
    "#A8E6CF",  # Light green
    "#FF8B94",  # Coral
    "#95E1D3",  # Mint
    "#F38181",  # Light red
    "#AA96DA",  # Lavender
    "#FCBAD3",  # Light pink
    "#FFFFD2",  # Cream
]

# Green Gradient for Single-Class Visualizations
GREEN_GRADIENT = [
    "#003322",  # Darkest
    "#006644",
    "#009955",
    "#00CC6A",
    "#00FF88",  # Primary
    "#33FFAA",
    "#66FFBB",
    "#99FFCC",  # Lightest
]

# Typography
FONT_FAMILY = "Inter"  # Fallback to system sans-serif if not available
FONT_SIZE_TITLE = 48
FONT_SIZE_LABEL = 32
FONT_SIZE_SMALL = 24

# Animation Timing (in seconds)
TRANSITION_FAST = 0.5
TRANSITION_MEDIUM = 1.0
TRANSITION_SLOW = 2.0

# Point Sizes for Scatter Plots
POINT_SIZE_SMALL = 0.03
POINT_SIZE_MEDIUM = 0.05
POINT_SIZE_LARGE = 0.08

# Axes Styling
AXIS_COLOR = PRIMARY_GREEN
AXIS_WIDTH = 2
GRID_COLOR = "#333333"
GRID_OPACITY = 0.3

def get_class_color(class_idx: int, num_classes: int) -> str:
    """
    Get color for a class index.
    Uses a different green shade for class 0 (to distinguish from axes),
    complementary colors for others.
    """
    if class_idx == 0:
        # Use ACCENT_GREEN instead of PRIMARY_GREEN to distinguish from axes
        return ACCENT_GREEN  # "#33FFAA" - lighter green, different from axes
    elif class_idx < len(COMPLEMENTARY_COLORS):
        return COMPLEMENTARY_COLORS[class_idx - 1]
    else:
        # Cycle through colors if more classes than colors
        return COMPLEMENTARY_COLORS[class_idx % len(COMPLEMENTARY_COLORS)]

def get_gradient_color(value: float, min_val: float = 0.0, max_val: float = 1.0) -> str:
    """
    Get color from green gradient based on normalized value.
    value should be between min_val and max_val.
    """
    # Normalize to [0, 1]
    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    normalized = max(0.0, min(1.0, normalized))
    
    # Map to gradient index
    idx = int(normalized * (len(GREEN_GRADIENT) - 1))
    return GREEN_GRADIENT[idx]

