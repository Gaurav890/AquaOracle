"""Figure and image extraction from PDFs."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from dataclasses import dataclass
from loguru import logger


@dataclass
class Figure:
    """Represents an extracted figure/image from PDF."""

    page_number: int
    bbox: tuple  # (x0, y0, x1, y1)
    image_index: int
    width: float
    height: float
    caption: Optional[str] = None
    image_data: Optional[bytes] = None
    image_ext: Optional[str] = None


class FigureExtractor:
    """Extract figures and images from PDFs."""

    def __init__(self, extract_images: bool = False):
        """
        Initialize figure extractor.

        Args:
            extract_images: Whether to extract actual image data
        """
        self.logger = logger.bind(name="FigureExtractor")
        self.extract_images = extract_images

    def extract_figures(self, pdf_path: Path) -> List[Figure]:
        """
        Extract all figures from a PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of extracted figures
        """
        self.logger.info(f"Extracting figures from: {pdf_path}")

        figures = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page_figures = self._extract_page_figures(doc, page_num)
                figures.extend(page_figures)

            doc.close()

            self.logger.info(f"Extracted {len(figures)} figures from {pdf_path.name}")
            return figures

        except Exception as e:
            self.logger.error(f"Failed to extract figures from {pdf_path}: {e}")
            return []

    def _extract_page_figures(self, doc: fitz.Document, page_num: int) -> List[Figure]:
        """Extract figures from a single page."""
        page = doc[page_num]
        figures = []

        try:
            # Get list of images on the page
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]  # Image xref number

                # Get image bbox
                try:
                    img_rect = page.get_image_bbox(img)
                    bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)
                    width = img_rect.width
                    height = img_rect.height
                except:
                    # Fallback if bbox extraction fails
                    bbox = (0, 0, 0, 0)
                    width = 0
                    height = 0

                # Extract image data if requested
                image_data = None
                image_ext = None
                if self.extract_images:
                    try:
                        base_image = doc.extract_image(xref)
                        image_data = base_image["image"]
                        image_ext = base_image["ext"]
                    except:
                        pass

                # Try to find caption (text near the figure)
                caption = self._find_caption(page, bbox)

                figures.append(Figure(
                    page_number=page_num + 1,
                    bbox=bbox,
                    image_index=img_index,
                    width=width,
                    height=height,
                    caption=caption,
                    image_data=image_data,
                    image_ext=image_ext,
                ))

        except Exception as e:
            self.logger.warning(f"Error extracting figures from page {page_num + 1}: {e}")

        return figures

    def _find_caption(self, page, image_bbox: tuple, search_distance: float = 50) -> Optional[str]:
        """
        Find caption text near an image.

        Args:
            page: PyMuPDF page object
            image_bbox: Image bounding box
            search_distance: How far below image to search for caption

        Returns:
            Caption text if found
        """
        try:
            x0, y0, x1, y1 = image_bbox

            # Search for text below the image
            caption_rect = fitz.Rect(x0, y1, x1, y1 + search_distance)
            caption_text = page.get_text("text", clip=caption_rect).strip()

            # Simple heuristic: captions usually start with "Figure", "Fig.", "Image", etc.
            if caption_text and any(caption_text.lower().startswith(prefix) for prefix in [
                "figure", "fig.", "fig:", "image", "img.", "diagram", "chart", "graph"
            ]):
                # Return first line only
                return caption_text.split('\n')[0]

        except Exception:
            pass

        return None

    def figure_to_text(self, figure: Figure) -> str:
        """
        Convert figure to text representation.

        Args:
            figure: Figure object

        Returns:
            Text description of figure
        """
        lines = []
        lines.append(f"[Figure {figure.image_index + 1} on page {figure.page_number}]")

        if figure.caption:
            lines.append(f"Caption: {figure.caption}")

        lines.append(f"Size: {figure.width:.1f} x {figure.height:.1f}")

        return "\n".join(lines)

    def save_figure(self, figure: Figure, output_path: Path) -> bool:
        """
        Save figure image to file.

        Args:
            figure: Figure object with image data
            output_path: Path to save image

        Returns:
            True if successful
        """
        if not figure.image_data or not figure.image_ext:
            self.logger.warning("Figure has no image data to save")
            return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = output_path / f"fig_p{figure.page_number}_i{figure.image_index}.{figure.image_ext}"

            with open(output_file, "wb") as f:
                f.write(figure.image_data)

            self.logger.info(f"Saved figure to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save figure: {e}")
            return False
