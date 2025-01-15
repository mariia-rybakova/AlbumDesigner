import io
import numpy as np

from ptinfra.azure.pt_file import PTFile


def get_image_embeddings(file, df, logger=None):
    embed = {}
    try:
        # Open the file and read content
        fb = PTFile(file)
        fileBytes = fb.read_blob()
        fileBytes = io.BytesIO(fileBytes)

        # Read header and number of images
        header_b = fileBytes.read1(4)
        header = header_b.decode('utf-8')
        num_images_b = fileBytes.read1(4)
        num_images = int.from_bytes(num_images_b, 'little')

        for i in range(num_images):
            # Read photo_id
            photo_id_b = fileBytes.read1(8)
            photo_id = int.from_bytes(photo_id_b, 'little')

            # Read embedding size and embedding data
            emb_size_b = fileBytes.read1(4)
            emb_size = int.from_bytes(emb_size_b, 'little')
            embedding_b = fileBytes.read1(4 * emb_size)
            embedding = np.frombuffer(embedding_b, dtype='float32').reshape((emb_size,))

            # Store embedding in dictionary
            embed[photo_id] = {'embedding': embedding}

    except Exception as e:
        if logger:
            logger.error(f"Error reading embeddings from file: {e}")
        return None

    # Add embedding column to DataFrame
    def get_embedding(image_id):
        return embed.get(image_id, {}).get('embedding', np.nan)

    df['embedding'] = df['image_id'].apply(get_embedding)

    return df
