import io
import pandas as pd
import numpy as np

from ptinfra.azure.pt_file import PTFile


def get_image_embeddings(file, logger=None):
    embed = {}

    try:
        fb = PTFile(file)
        fileBytes = fb.read_blob()
        fileBytes = io.BytesIO(fileBytes)

        header_b = fileBytes.read1(4)
        header = header_b.decode('utf-8')
        num_images_b = fileBytes.read1(4)
        num_images = int.from_bytes(num_images_b, 'little')

        for _ in range(num_images):
            photo_id_b = fileBytes.read1(8)
            photo_id = int.from_bytes(photo_id_b, 'little')

            emb_size_b = fileBytes.read1(4)
            emb_size = int.from_bytes(emb_size_b, 'little')
            embedding_b = fileBytes.read1(4 * emb_size)
            embedding = np.frombuffer(embedding_b, dtype='float32').reshape((emb_size,))

            embed[photo_id] = {'embedding': embedding}

    except Exception as e:
        if logger:
            logger.error(f"Error reading embeddings from file: {e}")
        return None

    df = pd.DataFrame([
        {"image_id": photo_id, "embedding": data["embedding"]}
        for photo_id, data in embed.items()
    ])

    return df

