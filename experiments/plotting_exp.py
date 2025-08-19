import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

import textwrap




def plot_images_with_first_last(df, image_dir, output_pdf,
                                first_page_images=None,
                                last_page_images=None,
                                images_per_row=3,
                                max_images_per_page=9):
    """
    Plot images in a PDF. First page and last page can be forced from given lists of image IDs.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['image_id', 'content'].
    image_dir : str
        Base directory where images are stored.
    output_pdf : str
        Path to save the PDF.
    first_page_images : list[str]
        List of image_ids to be placed on the first page (in order).
    last_page_images : list[str]
        List of image_ids to be placed on the last page (in order).
    images_per_row : int
        Number of images per row.
    max_images_per_page : int
        Number of images per page.
    """

    def render_page(image_ids, pdf, df, image_dir, images_per_row, max_images_per_page):
        """Helper to render one page of images given a list of image IDs."""
        if not image_ids:
            return

        fig, axes = plt.subplots(
            nrows=max_images_per_page // images_per_row,
            ncols=images_per_row,
            figsize=(12, 12)
        )
        axes = axes.flatten()

        for idx, image_id in enumerate(image_ids):
            ax = axes[idx]
            row = df[df["image_id"] == image_id]
            if row.empty:
                ax.axis("off")
                continue

            img_path = os.path.join(image_dir, f"{image_id}.jpg")
            if not os.path.exists(img_path):
                ax.axis("off")
                continue

            # Load image
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis("off")

            # Get metadata
            content = str(row.iloc[0]["cluster_context"])
            wrapped = textwrap.wrap(content, width=40)[:2]
            wrapped_text = "\n".join(wrapped)

            ax.set_title(f"ID: {image_id}\n{wrapped_text}",
                         fontsize=8, wrap=True, pad=4)

        # Turn off any unused subplots
        for j in range(len(image_ids), len(axes)):
            axes[j].axis("off")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(output_pdf) as pdf:
        # First page
        if first_page_images:
            render_page(first_page_images, pdf, df, image_dir, images_per_row, max_images_per_page)

        # Last page
        if last_page_images:
            render_page(last_page_images, pdf, df, image_dir, images_per_row, max_images_per_page)

    print(f"✅ PDF saved to {output_pdf}")


def plot_images_to_pdf(df, image_dir, output_pdf, keyword="bride and groom",
                       images_per_row=3, max_images_per_page=9):
    """
    Plot images with titles and content in a PDF, filtering rows by keyword in 'content'.

    Parameters:
    -----------
    df : pandas.DataFrame
        Must contain columns ['image_id', 'content'].
    image_dir : str
        Base directory where images are stored.
    output_pdf : str
        Path to save the PDF.
    keyword : str
        Keyword to filter 'content' column. Only matching rows will be included.
    images_per_row : int
        Number of images per row in the grid.
    max_images_per_page : int
        Max number of images per page in the PDF.
    """
    # Filter DataFrame
    filtered_df = df[df["cluster_context"].str.contains(keyword, case=False, na=False)]

    if filtered_df.empty:
        print(f"No images found with keyword '{keyword}'. PDF will not be created.")
        return

    with PdfPages(output_pdf) as pdf:
        fig = None
        for i, row in enumerate(filtered_df.itertuples(index=False), start=1):
            image_id, content = row.image_id, row.image_subquery_content
            img_path = os.path.join(image_dir, f"{image_id}.jpg")

            if not os.path.exists(img_path):
                print(f"⚠️ Image not found: {img_path}, skipping...")
                continue

            # Start a new figure for every page
            if (i - 1) % max_images_per_page == 0:
                if fig:
                    pdf.savefig(fig)
                    plt.close(fig)
                fig, axes = plt.subplots(
                    nrows=max_images_per_page // images_per_row,
                    ncols=images_per_row,
                    figsize=(12, 12)
                )
                axes = axes.flatten()

            ax = axes[(i - 1) % max_images_per_page]
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis("off")

            # Wrap text into max 2 lines
            wrapped = textwrap.wrap(content, width=40)[:2]
            wrapped_text = "\n".join(wrapped)

            ax.set_title(f"ID: {image_id}\n{wrapped_text}",
                         fontsize=8, wrap=True, pad=4)

        # Save last page
        if fig:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ PDF saved to {output_pdf}")


# output_pdf = r'C:\Users\karmel\Desktop\AlbumDesigner\output\brindeandgroom_2.pdf'
# plot_images_to_pdf(gallery_info_df, r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/46670335/',
#                    output_pdf, keyword="bride and groom",
#                    images_per_row=3, max_images_per_page=9)

# plot_images_with_first_last( df=data_df,
# image_dir=r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/46670335/',
# output_pdf="output/covers_images.pdf",
# first_page_images=first_page_ids,
# last_page_images=last_page_ids)

# plot_images_with_first_last( df=data_df,
    # image_dir=r'C:\Users\karmel\Desktop\AlbumDesigner\dataset\newest_wedding_galleries/36048323/',
    # output_pdf="output/covers_images_36048323_afterchanging.pdf",
    # first_page_images=first_page_ids,
    # last_page_images=last_page_ids)

