"""
Script to render Manim visualization from checkpoints.
Run this after training to generate the video.

Usage: python render_visualization.py <checkpoint_dir>
"""

import sys
from visualization.manim_scenes.latent_2d import create_latent_2d_scene_from_checkpoints


def main():
    """Render visualization from checkpoints."""
    if len(sys.argv) < 2:
        print("Usage: python render_visualization.py <checkpoint_dir>")
        print("Example: python render_visualization.py ./checkpoints/mlp_mnist")
        sys.exit(1)
    
    checkpoint_dir = sys.argv[1]
    
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
    )
    
    print()
    print("=" * 60)
    print("Rendering complete! Check media/videos/ for the output.")
    print("=" * 60)


if __name__ == "__main__":
    main()

