import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def draw_tumor_overlay(
        image_rgb: np.ndarray,
        mask: torch.Tensor | np.ndarray,
        color: tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw a semi-transparent color overlay on top of the tumor region,
    for a heatmap-style visualization alongside (or instead of) a bbox.

    Args:
        image_rgb: RGB numpy image (H, W, 3), uint8 — e.g. draw_tumor_bbox's
            "image_rgb" output, to guarantee the exact same base image.
        mask: Binary mask (H, W) with 0=background, 1=tumor.
        color: RGB color of the overlay.
        alpha: Opacity of the overlay in the tumor region (0=invisible,
            1=fully opaque color, no underlying image visible).

    Returns:
        numpy array (H, W, 3), RGB, uint8 — image with the overlay blended
        only inside the tumor region (background pixels untouched).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask_bool = mask.astype(bool)

    overlay = image_rgb.copy()
    color_layer = np.full_like(image_rgb, color, dtype=np.uint8)

    blended = cv2.addWeighted(image_rgb, 1 - alpha, color_layer, alpha, 0)
    overlay[mask_bool] = blended[mask_bool]

    return overlay


def draw_tumor_bbox(
        image: Image.Image | np.ndarray,
        mask: torch.Tensor | np.ndarray,
        min_area: int = 20,
        box_color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
) -> dict:
    """
    Find tumor regions in a binary segmentation mask and draw bounding
    boxes around them on the original image using OpenCV.

    Args:
        image: The original image (PIL.Image or numpy array, RGB or grayscale).
            Should already be resized/aligned to the mask's coordinate space.
        mask: Binary mask (H, W) with 0=background, 1=tumor. Accepts a
            torch.Tensor (e.g. from predict_mask's "predicted_mask") or
            a numpy array.
        min_area: Minimum contour area (in pixels) to keep.
        box_color: RGB color for the bounding box.
        thickness: Line thickness of the bounding box.

    Returns:
        A dict with:
            - "image_with_boxes": numpy array (H, W, 3), RGB, uint8
            - "boxes": list of (x, y, w, h), sorted by area descending
            - "contours": list of filtered cv2 contours (same order as boxes),
                kept so downstream functions (overlay, hull, centroid) don't
                need to re-run findContours
            - "num_regions": number of tumor regions found
            - "largest_box": (x, y, w, h) of the largest region, or None
            - "image_rgb": the normalized RGB numpy image (H, W, 3), uint8 —
                kept so downstream functions share the exact same base image
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask_uint8 = (mask.astype(np.uint8)) * 255

    if isinstance(image, Image.Image):
        image_rgb = np.array(image.convert("RGB"))
    else:
        image_rgb = image.copy()
        if image_rgb.ndim == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

    if image_rgb.shape[:2] != mask_uint8.shape[:2]:
        raise ValueError(
            f"image size {image_rgb.shape[:2]} and mask size "
            f"{mask_uint8.shape[:2]} don't match. Resize the mask to the "
            f"image's size (or vice versa) before calling this function."
        )

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    kept = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        kept.append((x, y, w, h, area, contour))

    kept.sort(key=lambda b: b[4], reverse=True)
    boxes = [(x, y, w, h) for x, y, w, h, _, _ in kept]
    filtered_contours = [c for _, _, _, _, _, c in kept]

    image_with_boxes = image_rgb.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), box_color, thickness)

    return {
        "image_with_boxes": image_with_boxes,
        "boxes": boxes,
        "contours": filtered_contours,
        "num_regions": len(boxes),
        "largest_box": boxes[0] if boxes else None,
        "image_rgb": image_rgb,
    }


def draw_tumor_centroids(
        image_rgb: np.ndarray,
        contours: list,
        total_pixels: int | None = None,
        text_color: tuple[int, int, int] = (255, 255, 0),
        dot_color: tuple[int, int, int] = (255, 255, 0),
        font_scale: float = 0.5,
) -> np.ndarray:
    """
    Mark the centroid of each tumor region and label it with its index
    and (optionally) its share of the image area.

    Args:
        image_rgb: RGB numpy image (H, W, 3), uint8.
        contours: list of cv2 contours — e.g. draw_tumor_bbox's "contours"
            output (already filtered and sorted by area descending).
        total_pixels: If given, each label includes the region's percentage
            of the total image (H*W). If None, only the region index is shown.
        text_color: RGB color for the label text.
        dot_color: RGB color for the centroid dot.
        font_scale: cv2.putText font scale.

    Returns:
        numpy array (H, W, 3), RGB, uint8 — image with centroid dots and
        labels drawn, e.g. "#1 (3.2%)".
    """
    image_labeled = image_rgb.copy()

    for i, contour in enumerate(contours, start=1):
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue  # degenerate contour, skip
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        cv2.circle(image_labeled, (cx, cy), 4, dot_color, -1)

        if total_pixels is not None:
            area = cv2.contourArea(contour)
            pct = (area / total_pixels) * 100
            label = f"#{i} ({pct:.1f}%)"
        else:
            label = f"#{i}"

        cv2.putText(
            image_labeled, label, (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA
        )

    return image_labeled


def draw_tumor_hull(
        image_rgb: np.ndarray,
        contours: list,
        hull_color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
) -> np.ndarray:
    """
    Draw the convex hull of each tumor contour, showing the actual rough
    shape of the tumor instead of a rectangular bounding box.

    Args:
        image_rgb: RGB numpy image (H, W, 3), uint8 — e.g. draw_tumor_bbox's
            "image_rgb" output.
        contours: list of cv2 contours — e.g. draw_tumor_bbox's "contours"
            output (already filtered by min_area).
        hull_color: RGB color for the hull outline.
        thickness: Line thickness of the hull outline.

    Returns:
        numpy array (H, W, 3), RGB, uint8 — image with convex hull(s) drawn.
    """
    image_with_hull = image_rgb.copy()
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.drawContours(image_with_hull, [hull], -1, hull_color, thickness)

    return image_with_hull


def plot_tumor_report(
        image_rgb: np.ndarray,
        mask: torch.Tensor | np.ndarray,
        bbox_result: dict,
        figsize: tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Build a side-by-side comparison figure: original image | raw mask |
    image with bbox + overlay + hull + centroid labels combined.

    Args:
        image_rgb: RGB numpy image (H, W, 3), uint8 — the original image.
        mask: Binary mask (H, W) with 0=background, 1=tumor.
        bbox_result: The dict returned by draw_tumor_bbox for this same
            image/mask pair (reused so contours/boxes aren't recomputed).
        figsize: Matplotlib figure size.

    Returns:
        A matplotlib.figure.Figure with 3 panels, ready to st.pyplot() in
        Streamlit or fig.savefig() for a report.
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask

    total_pixels = mask_np.size

    combined = draw_tumor_overlay(image_rgb, mask_np, alpha=0.35)
    combined = draw_tumor_hull(combined, bbox_result["contours"])
    for (x, y, w, h) in bbox_result["boxes"]:
        cv2.rectangle(combined, (x, y), (x + w, y + h), (255, 0, 0), 2)
    combined = draw_tumor_centroids(combined, bbox_result["contours"], total_pixels=total_pixels)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    axes[2].imshow(combined)
    axes[2].set_title(f"Analysis ({bbox_result['num_regions']} region(s))")
    axes[2].axis("off")

    fig.tight_layout()
    return fig
