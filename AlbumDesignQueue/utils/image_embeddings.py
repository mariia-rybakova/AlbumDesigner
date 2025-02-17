import io
import numpy as np

from ptinfra.azure.pt_file import PTFile


def get_image_embeddings(file, df, logger=None):
    embed = {}
    required_ids = set(df['image_id'].tolist())

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

            if photo_id in required_ids:
                embed[photo_id] = {'embedding': embedding}

    except Exception as e:
        if logger:
            logger.error(f"Error reading embeddings from file: {e}")
        return None

    df['embedding'] = df['image_id'].map(lambda x: embed.get(x, {}).get('embedding', np.nan))
    return df

