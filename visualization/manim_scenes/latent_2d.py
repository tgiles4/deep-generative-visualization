"""
2D latent space visualization scene with training progression.
"""

from manim import *
from manim import config
import numpy as np
import sys
import os
import multiprocessing
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from visualization.manim_scenes.base_scene import BaseVisualizationScene
from visualization.palette import *

# Configure Manim for parallel rendering (renders partial movie files in parallel)
# Default is sequential (jobs=1), set to number of CPU cores for parallel rendering
if 'MANIM_JOBS' in os.environ:
    config.jobs = int(os.environ['MANIM_JOBS'])
else:
    # Default to number of CPU cores, but cap at 4 to avoid overwhelming the system
    config.jobs = min(multiprocessing.cpu_count(), 8)


class Latent2DScene(BaseVisualizationScene):
    """
    Scene for visualizing 2D latent space evolution during training.
    Shows scatter plot of latent vectors colored by class labels.
    """
    
    def __init__(
        self,
        latent_data: dict,  # {epoch: (latent_vectors, labels)}
        num_classes: int = 10,
        point_size: float = POINT_SIZE_MEDIUM,
        seconds_per_epoch: float = 1.0,
        show_labels: bool = True,
        **kwargs,
    ):
        """
        Initialize 2D latent space scene.
        
        Args:
            latent_data: Dictionary mapping epoch to (latent_vectors, labels) tuples
            num_classes: Number of classes
            point_size: Size of scatter plot points
            seconds_per_epoch: Animation duration per epoch
            show_labels: Whether to show axis labels and epoch number
        """
        super().__init__(**kwargs)
        self.latent_data = latent_data
        self.num_classes = num_classes
        self.point_size = point_size
        self.seconds_per_epoch = seconds_per_epoch
        self.show_labels = show_labels
        
        # Get all epochs sorted
        self.epochs = sorted(latent_data.keys())
    
    def construct(self):
        """Construct the animation."""
        # Determine axis ranges from all data
        all_latents = np.concatenate([data[0] for data in self.latent_data.values()])
        x_min, x_max = all_latents[:, 0].min(), all_latents[:, 0].max()
        y_min, y_max = all_latents[:, 1].min(), all_latents[:, 1].max()
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_range = (x_min - x_padding, x_max + x_padding)
        y_range = (y_min - y_padding, y_max + y_padding)
        
        # Create axes - scale smaller to fit on screen
        axes = self.create_axes_2d(x_range=x_range, y_range=y_range)
        axes.scale(0.6).to_edge(DOWN, buff=0.3)
        
        # Create title
        title = self.create_title("2D Latent Space Evolution")
        title.to_edge(UP, buff=0.3)
        
        # Create epoch label
        epoch_label = self.create_label("Epoch: 0", font_size=FONT_SIZE_SMALL)
        epoch_label.to_edge(RIGHT, buff=0.5)
        epoch_label.to_edge(UP, buff=0.5)
        
        # Add static elements
        self.add(axes, title)
        if self.show_labels:
            self.add(epoch_label)
        
        # Animate through epochs
        dots_group = None
        
        for i, epoch in enumerate(self.epochs):
            latent_vectors, labels = self.latent_data[epoch]
            
            # Create dots for this epoch
            dots = VGroup()
            for latent, label in zip(latent_vectors, labels):
                color = get_class_color(int(label), self.num_classes)
                dot = Dot(
                    axes.coords_to_point(latent[0], latent[1]),
                    color=color,
                    radius=self.point_size,
                )
                dots.add(dot)
            
            # Animate dots
            if i == 0:
                # First epoch: fade in
                self.play(FadeIn(dots), run_time=0.5)
                dots_group = dots
            else:
                # Subsequent epochs: transform positions
                if dots_group is not None and len(dots) == len(dots_group):
                    # Same number of points: animate movement
                    animations = []
                    for k in range(len(dots)):
                        # Create new dot at target position
                        target_dot = dots[k]
                        # Transform existing dot to new position
                        animations.append(
                            dots_group[k].animate.move_to(target_dot.get_center())
                        )
                        # Update color if needed
                        if dots_group[k].color != target_dot.color:
                            animations.append(
                                dots_group[k].animate.set_color(target_dot.color)
                            )
                    
                    if animations:
                        self.play(*animations, run_time=self.seconds_per_epoch * 0.8)
                    
                    # Replace dots_group with new dots (for next iteration)
                    self.remove(dots_group)
                    dots_group = dots
                    self.add(dots_group)
                else:
                    # Different number of points: fade out and in
                    if dots_group is not None:
                        self.play(FadeOut(dots_group), run_time=0.2)
                    self.play(FadeIn(dots), run_time=0.3)
                    dots_group = dots
            
            # Update epoch label
            if self.show_labels:
                new_label = self.create_label(f"Epoch: {epoch}", font_size=FONT_SIZE_SMALL)
                new_label.to_edge(RIGHT, buff=0.5)
                new_label.to_edge(UP, buff=0.5)
                if i == 0:
                    # First epoch: set initial label
                    epoch_label.become(new_label)
                else:
                    self.play(Transform(epoch_label, new_label), run_time=0.2)
            
            # Pause between epochs (except last)
            if i < len(self.epochs) - 1:
                self.wait(0.1)
        
        # Final pause
        self.wait(1.0)


def create_latent_2d_scene_from_checkpoints(
    checkpoint_dir: str,
    epochs: list = None,
    **scene_kwargs,
) -> Latent2DScene:
    """
    Load checkpoint data and render Manim animation.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        epochs: List of epochs to visualize (None for all)
        **scene_kwargs: Additional arguments for Latent2DScene
        
    Returns:
        Latent2DScene instance
    """
    from training.checkpoint_manager import CheckpointManager
    
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Get available epochs
    if epochs is None:
        epochs = checkpoint_manager.get_available_epochs()
    
    # Load latent data
    latent_data = {}
    for epoch in epochs:
        try:
            snapshot = checkpoint_manager.load_latent_snapshot(epoch, include_images=False)
            latent_data[epoch] = (snapshot['latent_vectors'], snapshot['labels'])
        except FileNotFoundError:
            print(f"Warning: Checkpoint for epoch {epoch} not found, skipping")
            continue
    
    if not latent_data:
        raise ValueError(f"No latent data found in checkpoint directory: {checkpoint_dir}")
    
    # Create and render scene
    print(f"Rendering visualization for {len(latent_data)} epochs...")
    print(f"Using {config.jobs} parallel jobs for rendering")
    scene = Latent2DScene(latent_data, **scene_kwargs)
    scene.render()
    print(f"Video saved to: {scene.renderer.file_writer.movie_file_path}")
    
    return scene

