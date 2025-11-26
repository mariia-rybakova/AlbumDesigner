from pathlib import Path
from typing import Dict, List, Iterable
import os
import math
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _candidate_paths(images_dir: Path, image_id: str) -> List[Path]:
    """Return candidate file paths for an image_id without extension."""
    exts = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"]
    return [images_dir / f"{image_id}{ext}" for ext in exts]


def _resolve_image_path(images_dir: Path, image_id: str) -> Path:
    """Try common extensions; return first existing path or None."""
    for p in _candidate_paths(images_dir, str(image_id)):
        if p.exists():
            return p
    return None


def _chunk_iterable(seq: List, size: int) -> Iterable[List]:
    """Yield successive chunks of length size from seq."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def plot_groups_to_pdf(
    groups: Dict[str, List[int]],
    alloc: Dict[str, int],
    df: pd.DataFrame,
    images_dir: os.PathLike,
    cluster_name: str,
    cluster_label: str,
    output_dir: os.PathLike = None,
    selected_images:list=[],
    *,
    cols: int = 5,
    rows: int = 6,
    facecolor: str = "white",
    tight_layout_pad: float = 0.2,
) -> str:
    """
    Plot grouped images into a multi-page PDF.

    Each image shows:
      • image_id (top)
      • thumbnail (middle)
      • image_time_date (bottom)
      • cluster_label (under time)
    Red border marks selected images according to alloc[group_key].
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir) if output_dir is not None else images_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    thumbs_per_page = cols * rows
    pdf_path = str((output_dir / f"groups_preview_{cluster_name}_{cluster_label}.pdf").resolve())

    # Normalise "_SINGLES_" into standard keys
    groups_norm: Dict[str, List[int]] = {}
    for gk, idxs in groups.items():
        if gk == "_SINGLES_":
            for i, sub in enumerate(idxs):
                groups_norm[f"S{i}"] = list(sub)
        else:
            groups_norm[gk] = list(idxs)

    # Convert selected_images to a set of ints for O(1) lookup
    selected_set = set(map(int, selected_images)) if selected_images else set()

    with PdfPages(pdf_path) as pdf:
        for gk, idxs in groups_norm.items():
            if not idxs:
                continue

            group_df = df.loc[idxs]
            k_select = min(int(alloc.get(gk, 0)), len(group_df))

            # Build (image_id, time, path, label)
            image_info = []
            for _, row in group_df.iterrows():
                image_id = int(row["image_id"])  # keep as int for comparison
                image_time = str(row.get("image_time_date", ""))
                path = _resolve_image_path(images_dir, image_id)
                label = row["cluster_label"]
                image_info.append((image_id, image_time, path, label))

            # Paginate through group
            for page_i, chunk in enumerate(_chunk_iterable(image_info, thumbs_per_page), 1):
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2), facecolor=facecolor)
                axes = np.array(axes).reshape(rows, cols)

                title = (
                    f"Cluster: {cluster_label} | Group: {gk} | Size: {len(image_info)} | "
                    f"Select: {k_select}"
                )
                if len(image_info) > thumbs_per_page:
                    total_pages = math.ceil(len(image_info) / thumbs_per_page)
                    title += f" | Page {page_i}/{total_pages}"
                fig.suptitle(title, fontsize=13, fontweight="bold")

                for ax_idx, ax in enumerate(axes.ravel()):
                    ax.axis("off")
                    if ax_idx >= len(chunk):
                        continue

                    global_idx = (page_i - 1) * thumbs_per_page + ax_idx
                    image_id, image_time, path, label = chunk[ax_idx]

                    # ----- selection flags -----
                    is_selected_by_alloc = global_idx < k_select
                    is_user_selected = image_id in selected_set

                    # === Show image ===
                    if path and path.exists():
                        try:
                            with Image.open(path) as im:
                                ax.imshow(im)
                        except (UnidentifiedImageError, OSError):
                            ax.text(0.5, 0.5, "Unreadable", ha="center", va="center", fontsize=8)
                    else:
                        ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=8)

                    # === Titles & labels ===
                    ax.set_title(f"{global_idx + 1}. {image_id}", fontsize=8, pad=1)
                    ax.text(
                        0.5, -0.08, str(image_time),
                        fontsize=7, ha="center", va="top", transform=ax.transAxes
                    )
                    ax.text(
                        0.5, -0.15, f"Cluster: {label}",
                        fontsize=6, ha="center", va="top", transform=ax.transAxes, color="dimgray"
                    )

                    # === Borders (red for alloc‑selected) ===
                    ax.set_frame_on(True)
                    for spine in ax.spines.values():
                        spine.set_linewidth(3 if is_selected_by_alloc else 0.8)
                        spine.set_color("tab:red" if is_selected_by_alloc else "gray")

                    # === Green rectangle overlay for user‑selected images ===
                    if is_user_selected:
                        # Create a rectangle that exactly covers the image area
                        rect = plt.Rectangle(
                            (0, 0), 1, 1, transform=ax.transAxes,
                            facecolor="green", edgecolor="green",
                            linewidth=4, alpha=0.3, zorder=10
                        )
                        ax.add_patch(rect)

                plt.tight_layout(pad=tight_layout_pad, rect=[0, 0, 1, 0.95])
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

    return pdf_path


import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from PIL import Image, UnidentifiedImageError


def plot_selected_rows_to_pdf(selected_rows: pd.DataFrame,
                              images_dir=r'C:\Users\user\Desktop\PicTime\AlbumDesigner\dataset\47981912',
                              output_dir=r'C:\Users\user\Desktop\PicTime\AlbumDesigner\output\47981912',
                              cols=5,
                              rows=5,  # Reduced slightly to give more breathing room per page
                              facecolor="white",
                              dpi=150,
                              pdf_name="parents.pdf"):
    # --- Setup Paths ---
    images_dir = Path(images_dir)
    output_dir = Path(output_dir) if output_dir is not None else images_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = str((output_dir / pdf_name).resolve())

    if selected_rows is None or selected_rows.empty:
        print("No data to plot.")
        return pdf_path

    # --- Helper: Text Wrapper ---
    # Wraps text to a specific character width so it doesn't bleed into neighbors
    def _wrap_text(text, width=25):
        if pd.isna(text) or text == "nan":
            return "None"
        return textwrap.fill(str(text), width=width)

    # --- Helper: Meta Plotter ---
    def _plot_meta(ax, row_data):
        # Extract and clean data
        img_id = row_data.get("image_id", "")
        time_val = row_data.get("image_time_date", "")
        query = _wrap_text(row_data.get("image_query_content", ""), width=30)
        sub = _wrap_text(row_data.get("image_subquery_content", ""), width=30)
        ctx = _wrap_text(row_data.get("cluster_context", ""), width=30)
        parents = _wrap_text(row_data.get("parent_category", ""), width=30)

        meta_str = (
            f"ID: {img_id}\n"
            f"Time: {time_val}\n"
            f"Query: {query}\n"
            f"Sub: {sub}\n"
            f"Ctx: {ctx}\n"
            f"Parents: {parents}"
        )

        # Position text:
        # x=0.0, y=-0.05 puts it just below the left corner of the image.
        # va='top' ensures it grows downwards.
        ax.text(
            0.0, -0.08, meta_str,
            fontsize=6,
            ha="left",
            va="top",
            transform=ax.transAxes,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="none", alpha=0.5)  # Light formatting box
        )

    # --- Main Plotting Logic ---
    thumbs_per_page = cols * rows
    total = len(selected_rows)

    # Mock function for _resolve_image_path if not provided in snippet
    # Assuming it exists in your scope, otherwise use simplified logic:
    def _resolve_image_path_safe(base, img_id):
        # Try exact match or jpg/png extensions
        p = base / str(img_id)
        if p.exists(): return p
        if (base / f"{img_id}.jpg").exists(): return base / f"{img_id}.jpg"
        if (base / f"{img_id}.png").exists(): return base / f"{img_id}.png"
        return None

    with PdfPages(pdf_path) as pdf:
        fig = None
        axes = None

        for i, (_, row) in enumerate(selected_rows.iterrows(), start=1):
            # Check if we need a new page
            if (i - 1) % thumbs_per_page == 0:
                # Save previous page if it exists
                if fig is not None:
                    pdf.savefig(fig, dpi=dpi)
                    plt.close(fig)

                # Create new Figure
                # Increased height multiplier (4.0) to accommodate metadata block
                fig, axes = plt.subplots(rows, cols,
                                         figsize=(cols * 3.0, rows * 4.5),
                                         facecolor=facecolor)

                # Adjust layout: hspace=0.7 gives 70% of subplot height as gap between rows
                fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.3, hspace=0.8)

                axes = np.array(axes).reshape(rows, cols)

                # Page Header
                page_idx = (i - 1) // thumbs_per_page + 1
                total_pages = math.ceil(total / thumbs_per_page)
                fig.suptitle(f"Selected Parents Images — Page {page_idx}/{total_pages}",
                             fontsize=14, fontweight="bold", y=0.98)

                # Clean up axes initially
                for ax in axes.ravel():
                    ax.axis("off")

            # Determine current cell
            cell_idx = (i - 1) % thumbs_per_page
            ax = axes.ravel()[cell_idx]
            ax.axis("on")  # Turn on for border calculation

            # --- Load Image ---
            image_id = row["image_id"]
            path = _resolve_image_path_safe(images_dir, image_id)  # Using safe wrapper

            if path and path.exists():
                try:
                    with Image.open(path) as im:
                        # Aspect='equal' prevents distortion, extent maintains coord system
                        ax.imshow(im, aspect='equal')
                except (UnidentifiedImageError, OSError):
                    ax.text(0.5, 0.5, "Corrupt File", ha="center", va="center", fontsize=8, color="red")
            else:
                ax.text(0.5, 0.5, "Image Missing", ha="center", va="center", fontsize=8, color="gray")

            # --- Styling ---
            ax.set_xticks([])
            ax.set_yticks([])

            # Title (Image ID + Label)
            label = row.get("cluster_label", "Unknown")
            ax.set_title(f"{i}. {image_id} | {label}", fontsize=8, pad=4, fontweight='bold')

            # Add Metadata Block
            _plot_meta(ax, row)

            # Clean Borders
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")

            # Save the FINAL page
            if i == total:
                pdf.savefig(fig, dpi=dpi)
                plt.close(fig)

    print(f"PDF generated successfully at: {pdf_path}")
    return pdf_path