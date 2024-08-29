def get_persons_ids(persons_file, images_dict={},logger=None):
    try:
        person_info_bytes = PTFile(persons_file)  # load file
        if not person_info_bytes.exists():
            return None
        person_info_bytes = person_info_bytes.read_blob()
        person_descriptor = person_vector.PersonInfoMessageWrapper()
        person_descriptor.ParseFromString(person_info_bytes)

    except Exception as e:
        # logger.warning('Cant load cluster data from server: {}. Loading from local directory.'.format(e))
        print('Cant load cluster data from server: {}. will Load it from local directory.'.format(e))

        # load data locally
        try:
            person_descriptor = person_vector.PersonInfoMessageWrapper()
            with open(persons_file, 'rb') as f:
                person_descriptor.ParseFromString(f.read())
        except Exception as e:
            # logger.warning('Faces data could not be loaded local: {}'.format(e))
            print('Cant load cluster data from local: {}.'.format(e))
            return None

    if person_descriptor.WhichOneof("versions") == 'v1':
        message_data = person_descriptor.v1
    else:
        raise ValueError('There is no appropriate version of Person vector message.')

    identity_info = message_data.identities

    for iden in identity_info:
        id = iden.identityNumeralId
        infos = iden.personInfo
        for im in infos.imagesInfo:
            if im.photoId not in images_dict:
                images_dict[im.photoId] = {"embedding": [], "persons_ids": []}

            if "persons_ids" not in images_dict[im.photoId]:
                images_dict[im.photoId].update({"persons_ids": []})

            images_dict[im.photoId]['persons_ids'].append(id)

    return images_dict
