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


def plot_selected_rows_to_pdf(selected_rows: pd.DataFrame,
                                  images_dir=r'C:\Users\user\Desktop\PicTime\AlbumDesigner\dataset\46227780',output_dir=r'C:\Users\user\Desktop\PicTime\AlbumDesigner\output\46227780',
                                  cols=5,
                                  rows=6,
                                  facecolor="white",
                                  dpi=150,
                                  pdf_name="parents.pdf"):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir) if output_dir is not None else images_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    thumbs_per_page = cols * rows
    pdf_path = str((output_dir / pdf_name).resolve())
    if selected_rows is None or selected_rows.empty:
        return pdf_path

    def _plot_meta(ax, image_id, image_time, img_query, img_subquery, img_context,parents):
        meta_lines = [
            f"ID: {image_id}",
            f"Time: {image_time}",
            f"Query: {img_query}",
            f"Sub: {img_subquery}",
            f"Context: {'None' if pd.isna(img_context) else img_context}",
            f"parents: {parents}",
        ]
        # Put the block a bit lower and reserve bottom margin at the figure level
        ax.text(
            0.5, -0.12, "\n".join(meta_lines),
            fontsize=7, ha="center", va="top",
            transform=ax.transAxes, family="monospace", linespacing=1.15, wrap=True
        )

    with PdfPages(pdf_path) as pdf:
        fig = None
        axes = None
        total = len(selected_rows)

        for i, (_, row) in enumerate(selected_rows.iterrows(), start=1):
            # New page
            if (i - 1) % thumbs_per_page == 0:
                if fig is not None:
                    plt.tight_layout(pad=1.2, rect=[0, 0.20, 1, 0.94])  # extra bottom for text
                    pdf.savefig(fig, dpi=dpi)
                    plt.close(fig)

                # Slightly larger thumbs than before (clearer, but not huge)
                fig, axes = plt.subplots(rows, cols,
                                         figsize=(cols * 2.4, rows * 3.0),
                                         facecolor=facecolor)
                axes = np.array(axes).reshape(rows, cols)
                page_idx = (i - 1) // thumbs_per_page + 1
                total_pages = math.ceil(total / thumbs_per_page)
                fig.suptitle(f"Selected Parents Images — Page {page_idx}/{total_pages}",
                             fontsize=14, fontweight="bold")

                for ax in axes.ravel():
                    ax.axis("on")  # keep spines for borders
                    ax.set_xticks([]);
                    ax.set_yticks([])  # hide tick marks
                    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            # Cell
            cell = (i - 1) % thumbs_per_page
            ax = axes.ravel()[cell]

            # Data
            image_id = int(row["image_id"])
            image_time = str(row.get("image_time_date", ""))
            path = _resolve_image_path(images_dir, image_id)
            label = row.get("cluster_label", "")
            img_query = str(row.get("image_query_content", ""))
            img_subquery = str(row.get("image_subquery_content", ""))
            img_context = row.get("cluster_context", None)
            parents = row.get("parent_category", "")

            # Image
            if path and path.exists():
                try:
                    with Image.open(path) as im:
                        ax.imshow(im)
                except (UnidentifiedImageError, OSError):
                    ax.text(0.5, 0.5, "Unreadable", ha="center", va="center", fontsize=9)
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=9)

            # Title above image
            ax.set_title(f"{i}. {image_id} | {label}", fontsize=9, pad=2)

            # Metadata block (5 separate lines) under image
            _plot_meta(ax, image_id, image_time, img_query, img_subquery, img_context,parents)

            # Clean border (thin gray)
            for spine in ax.spines.values():
                spine.set_linewidth(0.9)
                spine.set_color("gray")

            # Save page if needed
            if (i % thumbs_per_page == 0) or (i == total):
                plt.tight_layout(pad=1.2, rect=[0, 0.20, 1, 0.94])
                pdf.savefig(fig, dpi=dpi)
                plt.close(fig)
                fig = None

    return pdf_path