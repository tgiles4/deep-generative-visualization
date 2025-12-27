"""
Script to render Manim visualization from checkpoints.
Run this after training to generate the video.
"""

import sys
from pathlib import Path
from visualization.manim_scenes.latent_2d import create_latent_2d_scene_from_checkpoints


def main():
    """Render visualization from checkpoints."""
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    else:
        checkpoint_dir = "./checkpoints/mlp_mnist"
    
    print("=" * 60)
    print("Rendering 2D Latent Space Visualization")
    print("=" * 60)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print()
    
    # Load checkpoints and render visualization
    print("Loading checkpoint data and rendering...")
    print("(This may take a while - Manim rendering can be slow)")
    print()
    
    scene = create_latent_2d_scene_from_checkpoints(
        checkpoint_dir=checkpoint_dir,
        epochs=None,  # Use all available epochs
        num_classes=10,
        point_size=0.04,
        seconds_per_epoch=1.0,
        show_labels=True,
    )
    
    print()
    print("=" * 60)
    print("Rendering complete! Check media/videos/ for the output.")
    print("=" * 60)


if __name__ == "__main__":
    main()

