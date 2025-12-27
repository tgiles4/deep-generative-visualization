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
    # Default to number of CPU cores, but cap at 8 to avoid overwhelming the system
    config.jobs = min(multiprocessing.cpu_count(), 8)


class Latent2DScenePolished(BaseVisualizationScene):
    def __init__(
        self,
        latent_data: dict,                 # {epoch: (latent_vectors, labels)}
        num_classes: int = 10,
        point_radius: float = POINT_SIZE_MEDIUM,
        seconds_per_epoch: float = 1.0,
        show_labels: bool = True,
        show_centroids: bool = True,
        grid_opacity: float = 0.25,
        **kwargs,
    ):
        # Filter out scene-specific kwargs that shouldn't go to parent Scene
        # Only pass through valid Manim Scene kwargs
        scene_kwargs = {}
        valid_scene_kwargs = ['camera_class', 'background_color', 'frame_rate']
        for key, value in kwargs.items():
            if key in valid_scene_kwargs:
                scene_kwargs[key] = value
        
        super().__init__(**scene_kwargs)
        self.latent_data = latent_data
        self.num_classes = num_classes
        self.point_radius = point_radius
        self.seconds_per_epoch = seconds_per_epoch
        self.show_labels = show_labels
        self.show_centroids = show_centroids
        self.grid_opacity = grid_opacity
        self.epochs = sorted(latent_data.keys())

    def construct(self):
        # Test camera and rendering - ensure objects are visible
        self.camera.background_color = BACKGROUND_DARK
        
        # # Create test objects - use explicit positioning
        # test_text = Text("CAMERA OK", font_size=64, color="#FFFFFF")
        # test_dot = Dot(radius=0.2, color="#FF0000")
        
        # # Position at origin explicitly
        # test_text.move_to([0, 0, 0])
        # test_dot.move_to([0, 0, 0])
        
        # print(f"DEBUG: Adding test objects")
        # print(f"  Text center: {test_text.get_center()}")
        # print(f"  Dot center: {test_dot.get_center()}")
        # print(f"  Camera background: {self.camera.background_color}")
        
        # self.add(test_text, test_dot)
        # self.wait(2.0)  # Wait longer to see it
        # self.remove(test_text, test_dot)

        # --- ranges ---
        all_latents = np.concatenate([data[0] for data in self.latent_data.values()])
        
        if len(all_latents) == 0:
            print("ERROR: No latent vectors to visualize!")
            return
        
        x_min, x_max = float(all_latents[:, 0].min()), float(all_latents[:, 0].max())
        y_min, y_max = float(all_latents[:, 1].min()), float(all_latents[:, 1].max())
        
        print(f"Scene: Data range x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")

        x_pad = (x_max - x_min) * 0.12 if x_max > x_min else 1.0
        y_pad = (y_max - y_min) * 0.12 if y_max > y_min else 1.0
        x_range = (x_min - x_pad, x_max + x_pad)
        y_range = (y_min - y_pad, y_max + y_pad)

        # --- plane + axes ---
        # choose a fixed on-screen footprint
        X_LEN = 10
        Y_LEN = 5.6

        # Make grid fainter (0.12-0.18 opacity)
        grid_opacity = 0.15
        
        plane = NumberPlane(
            x_range=[x_range[0], x_range[1], (x_range[1] - x_range[0]) / 6],
            y_range=[y_range[0], y_range[1], (y_range[1] - y_range[0]) / 6],
            x_length=X_LEN,
            y_length=Y_LEN,
            background_line_style={"stroke_opacity": grid_opacity, "stroke_width": 1},
        ).set_color(PRIMARY_GREEN)

        x_pad = (x_max - x_min) * 0.12 if x_max > x_min else 1.0
        y_pad = (y_max - y_min) * 0.12 if y_max > y_min else 1.0

        x_min_p, x_max_p = x_min - x_pad, x_max + x_pad
        y_min_p, y_max_p = y_min - y_pad, y_max + y_pad

        # Choose a reasonable tick/grid step
        x_step = (x_max_p - x_min_p) / 6 if (x_max_p > x_min_p) else 1.0
        y_step = (y_max_p - y_min_p) / 6 if (y_max_p > y_min_p) else 1.0

        axes_x_range = [x_min_p, x_max_p, x_step]
        axes_y_range = [y_min_p, y_max_p, y_step]

        axes = Axes(
            x_range=axes_x_range,
            y_range=axes_y_range,
            x_length=X_LEN,
            y_length=Y_LEN,
            axis_config={"color": PRIMARY_GREEN, "stroke_width": AXIS_WIDTH, "include_tip": True},
        )

        plot_group = VGroup(plane, axes).to_edge(DOWN, buff=0.35)  # <-- no .scale()

        self.axes = axes
        # --- labels ---
        # Epoch label with pill background - positioned inside plot area
        epoch_text = Text(f"Epoch {self.epochs[0]}", font_size=28).set_color("#FFFFFF")  # WHITE
        pill = RoundedRectangle(corner_radius=0.15, height=0.45, width=epoch_text.width + 0.5)
        pill.set_fill("#000000", opacity=0.55).set_stroke(width=0)  # BLACK with opacity
        
        epoch_label = VGroup(pill, epoch_text)
        epoch_text.move_to(pill.get_center())
        epoch_label.set_z_index(50)
        
        # Position epoch label inside plot area (slightly above x-axis, left of origin)
        epoch_label.move_to(axes.coords_to_point(
            x_min_p + 0.15 * (x_max_p - x_min_p),
            y_max_p - 0.10 * (y_max_p - y_min_p)
        ))

        x_lab = self.create_label("z₁", font_size=FONT_SIZE_SMALL).next_to(axes.x_axis, RIGHT, buff=0.2)
        y_lab = self.create_label("z₂", font_size=FONT_SIZE_SMALL).next_to(axes.y_axis, UP, buff=0.2)

        # Create legend with colored dots for classes 0-9
        legend_dots = VGroup()
        legend_spacing = 0.25
        legend_start_x = -config.frame_width / 2 + 0.5
        legend_y = -config.frame_height / 2 + 0.3
        
        for c in range(min(self.num_classes, 10)):
            dot = Dot(radius=0.08, color=get_class_color(c, self.num_classes))
            dot.move_to([legend_start_x + c * legend_spacing, legend_y, 0])
            label = Text(str(c), font_size=16, color=TEXT_WHITE)
            label.move_to(dot.get_center())
            legend_item = VGroup(dot, label)
            legend_dots.add(legend_item)

        # Add plot_group first so axes are in the scene and transformed
        self.add(plot_group)
        if self.show_labels:
            self.add(epoch_label, x_lab, y_lab, legend_dots)
        
        # Wait a frame to ensure transformations are applied
        self.wait(0.01)

        # --- initialize dots once ---
        e0 = self.epochs[0]
        lat0, lab0 = self.latent_data[e0]
        
        print(f"Scene: Creating {len(lat0)} dots for epoch {e0}")
        if len(lat0) == 0:
            print("ERROR: No dots to create!")
            return
        
        dots = VGroup()
        for i, (latent, label) in enumerate(zip(lat0, lab0)):
            color = get_class_color(int(label), self.num_classes)
            # Use the transformed axes for coordinate conversion
            # The axes are part of plot_group which has been scaled and moved
            point = self.axes.coords_to_point(float(latent[0]), float(latent[1]))
            d = Dot(
                point=point,
                radius=self.point_radius,
            )
            d.set_color(color)
            d.set_opacity(0.70)  # Reduced opacity for less clutter
            dots.add(d)
            
            # Debug first few dots
            if i < 3:
                print(f"  Dot {i}: latent=({latent[0]:.3f}, {latent[1]:.3f}), "
                      f"point=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}), "
                      f"label={int(label)}, color={color}")
        
        print(f"Scene: Created {len(dots)} dots, adding to scene...")

        self.play(FadeIn(dots, shift=0.1 * UP), run_time=0.6, rate_func=smooth)

        # --- optional: centroids (anchors make "learning" read better) ---
        def compute_centroids(latents, labels):
            centroids = []
            for c in range(self.num_classes):
                idx = np.where(labels.astype(int) == c)[0]
                if len(idx) == 0:
                    centroids.append(None)
                else:
                    centroids.append(latents[idx].mean(axis=0))
            return centroids

        # Initialize centroid containers outside the if block so they're accessible later
        centroid_dots = VGroup()
        centroid_labels = VGroup()
        
        if self.show_centroids:
            cents0 = compute_centroids(lat0, lab0)
            for c, mu in enumerate(cents0):
                if mu is None:
                    continue
                class_color = get_class_color(c, self.num_classes)
                
                # Centroid dot - slightly bigger with white stroke
                cd = Dot(
                    point=self.axes.coords_to_point(float(mu[0]), float(mu[1])),
                    radius=self.point_radius * 2.6,
                ).set_color(class_color)
                cd.set_opacity(0.95)
                cd.set_stroke("#FFFFFF", width=2.5, opacity=0.95)  # WHITE stroke
                cd.set_z_index(20)
                
                # Label inside centroid - dead center with black stroke for readability
                cl = Text(str(c), font_size=22, weight=BOLD).set_color("#FFFFFF")  # WHITE
                cl.set_stroke("#000000", width=6, opacity=1.0)  # BLACK stroke - key for readability
                cl.move_to(cd.get_center())
                cl.set_z_index(30)
                
                centroid_dots.add(cd)
                centroid_labels.add(cl)

            self.play(FadeIn(centroid_dots), FadeIn(centroid_labels), run_time=0.4)

        # --- animate epochs (in-place updates; no rebuilding) ---
        for i, epoch in enumerate(self.epochs[1:], start=1):
            lat, lab = self.latent_data[epoch]

            # If ordering isn't stable across epochs, movement will look wrong.
            # (Fix = store a stable sample_id ordering and align by id.)
            if len(lat) != len(dots):
                self.play(FadeOut(dots), run_time=0.25)
                dots = VGroup()
                for latent, label in zip(lat, lab):
                    d = Dot(
                        point=self.axes.coords_to_point(float(latent[0]), float(latent[1])),
                        radius=self.point_radius,
                    ).set_color(get_class_color(int(label), self.num_classes))
                    d.set_opacity(0.70)  # Reduced opacity for less clutter
                    dots.add(d)
                self.play(FadeIn(dots), run_time=0.35)
            else:
                anims = []
                for d, latent, label in zip(dots, lat, lab):
                    target = self.axes.coords_to_point(float(latent[0]), float(latent[1]))
                    color = get_class_color(int(label), self.num_classes)
                    anims.append(d.animate.move_to(target).set_color(color))

                self.play(
                    AnimationGroup(*anims, lag_ratio=0.0),
                    run_time=self.seconds_per_epoch,
                    rate_func=rate_functions.ease_in_out_cubic,
                )

            if self.show_centroids:
                cents = compute_centroids(lat, lab)
                cent_anims = []
                lab_anims = []
                # update only the ones that exist
                k = 0
                for c, mu in enumerate(cents):
                    if mu is None:
                        continue
                    target = self.axes.coords_to_point(float(mu[0]), float(mu[1]))
                    # Keep labels glued to centroid center during motion
                    cent_anims.append(centroid_dots[k].animate.move_to(target))
                    lab_anims.append(centroid_labels[k].animate.move_to(target))
                    k += 1
                if cent_anims:
                    self.play(AnimationGroup(*cent_anims, *lab_anims, lag_ratio=0.0),
                              run_time=0.35,
                              rate_func=smooth)

            if self.show_labels:
                # Update epoch label with pill background - keep it inside plot area
                new_epoch_text = Text(f"Epoch {epoch}", font_size=28).set_color("#FFFFFF")
                new_pill = RoundedRectangle(corner_radius=0.15, height=0.45, width=new_epoch_text.width + 0.5)
                new_pill.set_fill("#000000", opacity=0.55).set_stroke(width=0)
                new_epoch_label_group = VGroup(new_pill, new_epoch_text)
                new_epoch_text.move_to(new_pill.get_center())
                new_epoch_label_group.set_z_index(50)
                
                # Position inside plot area (same location as initial)
                new_epoch_label_group.move_to(axes.coords_to_point(
                    x_min_p + 0.15 * (x_max_p - x_min_p),
                    y_max_p - 0.10 * (y_max_p - y_min_p)
                ))
                
                # Transform the old epoch label to the new one
                self.play(Transform(epoch_label, new_epoch_label_group), run_time=0.2)
                epoch_label = new_epoch_label_group  # Update reference

            # small settle helps perception
            self.wait(0.05)

        self.wait(1.0)


def create_latent_2d_scene_from_checkpoints(
    checkpoint_dir: str,
    epochs: list = None,
    **scene_kwargs,
) -> Latent2DScenePolished:
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
    
    # Load latent data and align by sample_nums
    latent_data = {}
    all_sample_nums = None
    
    for epoch in epochs:
        try:
            snapshot = checkpoint_manager.load_latent_snapshot(epoch, include_images=False)
            sample_nums = snapshot['sample_nums']
            latent_vectors = snapshot['latent_vectors']
            labels = snapshot['labels']
            
            print(f"Epoch {epoch}: Loaded {len(latent_vectors)} samples, {len(sample_nums)} sample_nums")
            if len(latent_vectors) > 0:
                print(f"  Latent range: [{latent_vectors[:, 0].min():.3f}, {latent_vectors[:, 0].max():.3f}], "
                      f"[{latent_vectors[:, 1].min():.3f}, {latent_vectors[:, 1].max():.3f}]")
            
            # Find intersection of sample_nums across all epochs
            if all_sample_nums is None:
                all_sample_nums = set(sample_nums)
                print(f"  Initial sample_nums set: {len(all_sample_nums)} unique IDs")
            else:
                before = len(all_sample_nums)
                all_sample_nums = all_sample_nums.intersection(set(sample_nums))
                after = len(all_sample_nums)
                print(f"  After intersection: {before} -> {after} common IDs")
            
            latent_data[epoch] = (
                latent_vectors,
                labels,
                sample_nums
            )
        except FileNotFoundError:
            print(f"Warning: Checkpoint for epoch {epoch} not found, skipping")
            continue
    
    if not latent_data:
        raise ValueError(f"No latent data found in checkpoint directory: {checkpoint_dir}")
    
    # Convert to sorted array for consistent ordering
    if all_sample_nums:
        all_sample_nums = np.array(sorted(all_sample_nums))
        print(f"\nFinal common sample_nums: {len(all_sample_nums)} IDs")
    else:
        raise ValueError("No common sample_nums found across epochs")
    
    # Align all epochs by sample_nums (filter and sort)
    aligned_latent_data = {}
    for epoch, (latent_vectors, labels, sample_nums) in latent_data.items():
        # Filter to common sample_nums
        mask = np.isin(sample_nums, all_sample_nums)
        filtered_latents = latent_vectors[mask]
        filtered_labels = labels[mask] if labels is not None else None
        filtered_sample_nums = sample_nums[mask]
        
        print(f"Epoch {epoch}: After filtering to common IDs: {len(filtered_latents)} samples")
        
        # Sort by sample_nums for consistent ordering
        sort_idx = np.argsort(filtered_sample_nums)
        aligned_latents = filtered_latents[sort_idx]
        aligned_labels = filtered_labels[sort_idx] if filtered_labels is not None else None
        
        aligned_latent_data[epoch] = (aligned_latents, aligned_labels)
    
    latent_data = aligned_latent_data
    
    # Create and render scene
    print(f"Rendering visualization for {len(latent_data)} epochs...")
    print(f"Using {config.jobs} parallel jobs for rendering")
    print(f"Aligned {len(all_sample_nums)} samples across all epochs")
    
    # Filter scene_kwargs to only include valid parameters for Latent2DScenePolished
    # Map point_size to point_radius if provided (for backward compatibility)
    filtered_kwargs = {}
    valid_params = ['num_classes', 'point_radius', 'seconds_per_epoch', 'show_labels', 
                   'show_centroids', 'grid_opacity']
    
    for key, value in scene_kwargs.items():
        if key == 'point_size':
            # Map point_size to point_radius for backward compatibility
            filtered_kwargs['point_radius'] = value
        elif key in valid_params:
            filtered_kwargs[key] = value
    
    # Create scene instance
    scene = Latent2DScenePolished(latent_data, **filtered_kwargs)
    
    # Render the scene
    # Note: When calling render() directly, ensure Manim config is set up properly
    try:
        scene.render()
        if hasattr(scene, 'renderer') and hasattr(scene.renderer, 'file_writer'):
            print(f"Video saved to: {scene.renderer.file_writer.movie_file_path}")
        else:
            print("Warning: Could not determine output file path")
    except Exception as e:
        print(f"Error during rendering: {e}")
        print("Try rendering via Manim CLI instead:")
        print(f"  manim -pql visualization/manim_scenes/latent_2d.py Latent2DScenePolished")
        raise
    
    return scene
