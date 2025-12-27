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
    
    # Create scene from checkpoints
    print("Loading checkpoint data...")
    scene = create_latent_2d_scene_from_checkpoints(
        checkpoint_dir=checkpoint_dir,
        epochs=None,  # Use all available epochs
        num_classes=10,
        point_size=0.04,
        seconds_per_epoch=1.0,
        show_labels=True,
    )
    
    print(f"Found {len(scene.epochs)} epochs to visualize")
    print()
    print("Rendering scene...")
    print("(This may take a while - Manim rendering can be slow)")
    print()
    
    # Render the scene
    # Note: This will create the video in media/videos/
    scene.render()
    
    print()
    print("=" * 60)
    print("Rendering complete! Check media/videos/ for the output.")
    print("=" * 60)


if __name__ == "__main__":
    main()

