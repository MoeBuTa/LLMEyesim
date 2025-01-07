import base64
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image
from eye import QVGA_X, QVGA_Y
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from LLMEyesim.eyesim.utils.models import PlotConfig


class ImageProcess:
    """Enhanced image processing class with optimizations and caching"""

    def __init__(self, plot_config: Optional[PlotConfig] = None):
        """Initialize with optional custom plot configuration"""
        self.config = plot_config or PlotConfig()
        # Set both style and context
        sns.set_style(self.config.style)
        sns.set_context(self.config.context)
        self._setup_plot_defaults()

    def _setup_plot_defaults(self) -> None:
        """Set up default plotting parameters"""
        plt.rcParams['figure.dpi'] = self.config.dpi
        plt.rcParams['savefig.dpi'] = self.config.dpi
        plt.rcParams['figure.autolayout'] = True

    @staticmethod
    @lru_cache(maxsize=32)
    def _generate_degree_arrays(num_points: int = 360) -> Tuple[np.ndarray, np.ndarray]:
        """Generate cached degree arrays for plotting"""
        degrees = np.linspace(-180, 179, num=num_points)
        radians = np.deg2rad(np.arange(0, num_points))
        return degrees, radians

    def lidar2image_lineplot(self,
                             scan: List[int],
                             experiment_time: str,
                             save_path: str,
                             figure_size: Tuple[int, int] = (2, 2)) -> None:
        """
        Create and save a line plot of LiDAR data with optimized rendering
        """
        try:
            degrees, _ = self._generate_degree_arrays()

            # Use plt.style.context instead of with statement for style
            fig, ax = plt.subplots(figsize=figure_size)

            # Use numpy operations for better performance
            scan_array = np.array(scan)
            sns.lineplot(x=degrees, y=scan_array, ax=ax)

            ax.set_title(f"LiDAR Data Plot (t={experiment_time})")
            ax.set_xlabel("Degree")
            ax.set_ylabel("Distance")
            ax.set_xlim(-180, 180)
            ax.set_xticks(np.arange(-180, 181, 30))

            # Optimize file saving
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        except Exception as e:
            plt.close('all')  # Cleanup on error
            raise RuntimeError(f"Failed to create line plot: {str(e)}")

    def lidar2image(self, scan: List[int], save_path: str) -> None:
        """
        Create and save a polar plot of LiDAR data with optimized rendering
        """
        try:
            # Optimize array operations using numpy
            scan_array = np.array(scan)
            shift_index = 179
            shifted_scan = np.roll(scan_array, -shift_index)

            _, radians = self._generate_degree_arrays()
            normalized_scan = shifted_scan / np.max(shifted_scan)

            fig, ax = plt.subplots(
                subplot_kw={"projection": "polar"},
                figsize=self.config.figsize
            )

            scatter = ax.scatter(
                radians,
                shifted_scan,
                s=self.config.marker_size,
                c=normalized_scan,
                cmap=self.config.cmap,
                alpha=self.config.alpha
            )

            self._configure_polar_plot(ax, shifted_scan)
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

        except Exception as e:
            plt.close('all')  # Cleanup on error
            raise RuntimeError(f"Failed to create polar plot: {str(e)}")

    def _configure_polar_plot(self, ax: plt.Axes, scan_data: np.ndarray) -> None:
        """Configure polar plot appearance and settings"""
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, np.max(scan_data))
        ax.set_facecolor("white")

        ax.grid(
            color=self.config.grid_color,
            linestyle="-",
            linewidth=0.5,
            alpha=self.config.grid_alpha
        )

        # Optimize tick generation
        degree_ticks = np.arange(0, 360, 30)
        ax.set_xticks(np.deg2rad(degree_ticks))
        ax.set_xticklabels([f"{int(d)}°" for d in degree_ticks])

    @staticmethod
    def cam2image(image_data: bytes) -> Image.Image:
        """
        Convert camera image bytes to PIL Image with error handling

        Args:
            image_data: Raw image bytes in RGB format

        Returns:
            PIL Image object
        """
        try:
            return Image.frombytes("RGB", (QVGA_X, QVGA_Y), image_data)
        except Exception as e:
            raise RuntimeError(f"Failed to convert camera image: {str(e)}")

    @staticmethod
    def encode_image(image_path: Union[str, Path]) -> str:
        """
        Encode image file to base64 string with optimized reading

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string of the image
        """
        try:
            image_path = Path(image_path)
            with image_path.open('rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to encode image {image_path}: {str(e)}")

    @staticmethod
    def save_image_grid(images: List[Image.Image],
                        save_path: str,
                        cols: int = 3,
                        padding: int = 10) -> None:
        """
        Create and save a grid of images (new functionality)

        Args:
            images: List of PIL Image objects
            save_path: Path to save the output image
            cols: Number of columns in the grid
            padding: Padding between images in pixels

        Raises:
            ValueError: If no images are provided
            RuntimeError: If image grid creation or saving fails
        """
        if not images:
            raise ValueError("No images provided")

        grid_img = None
        try:
            # Calculate grid dimensions
            n_images = len(images)
            rows = (n_images + cols - 1) // cols

            # Get max dimensions
            max_w = max(img.width for img in images)
            max_h = max(img.height for img in images)

            # Create grid image
            grid_w = cols * max_w + (cols + 1) * padding
            grid_h = rows * max_h + (rows + 1) * padding
            grid_img = Image.new('RGB', (grid_w, grid_h), 'white')

            # Place images in grid
            for idx, img in enumerate(images):
                row = idx // cols
                col = idx % cols
                x = col * (max_w + padding) + padding
                y = row * (max_h + padding) + padding
                grid_img.paste(img, (x, y))

            grid_img.save(save_path, quality=95, optimize=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create or save image grid: {str(e)}")
        finally:
            if grid_img:
                grid_img.close()