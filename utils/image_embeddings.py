



def get_image_embeddings(file, logger=None):
    embed = {}
    try:
        # get matrix_url content
        fb = PTFile(file)
        fileBytes = fb.read_blob()
        fileBytes = io.BytesIO(fileBytes)
        # header
        header_b = fileBytes.read1(4)
        header = header_b.decode('utf-8')
        num_images_b = fileBytes.read1(4)
        num_images = int.from_bytes(num_images_b, 'little')
        for i in range(num_images):
            photo_id_b = fileBytes.read1(8)
            photo_id = int.from_bytes(photo_id_b, 'little')
            emb_size_b = fileBytes.read1(4)
            emb_size = int.from_bytes(emb_size_b, 'little')
            embedding_b = fileBytes.read1(4 * emb_size)
            embedding = np.frombuffer(embedding_b, dtype='float32').reshape((emb_size,))
            embed[photo_id] = {'embedding': embedding}
    except Exception as e:
        if logger is not None:
            logger.error('Embeddings reading error',
                         extra={'error': e,
                                'matrix_url': file,
                                'info': sys.exc_info()[0]})
        else:
            print('Embeddings reading error', 'error', e)

    return embed
