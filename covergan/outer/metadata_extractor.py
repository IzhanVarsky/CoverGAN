import logging
from io import BytesIO

import eyed3
import eyed3.utils.art
# Because one tag parser is not enough
import mutagen
from PIL import Image
from eyed3 import id3

logger = logging.getLogger("metadata_extractor")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def get_tag_map(filename: str):
    logger.info(f'Extracting metadata for {filename}')

    eyed3_tag = id3.Tag()
    if not eyed3_tag.parse(filename):
        return None

    try:
        mutagen_tag = mutagen.File(filename)
    except mutagen.MutagenError:
        return None

    result = {
        'filename': eyed3_tag.file_info.name,
        'artist': eyed3_tag.artist,
        'title': eyed3_tag.title,
        'release_year': eyed3_tag.getBestDate().year,
        'genre': eyed3_tag.genre
    }

    # eyeD3
    covers = eyed3.utils.art.getArtFromTag(eyed3_tag, id3.frames.ImageFrame.FRONT_COVER)
    if covers:
        result['cover'] = Image.open(BytesIO(covers[0].image_data))

    # mutagen
    result['country'] = mutagen_tag.get('releasecountry')
    result['country'] = None if result['country'] is None else result['country'][0]
    result['release_date'] = mutagen_tag.get('date')  # less reliable than eyeD3's year?f

    # mutagen + MusicBrainz
    result['artist_id'] = mutagen_tag.get('musicbrainz_artistid')
    result['album_id'] = mutagen_tag.get('musicbrainz_albumid')
    result['track_id'] = mutagen_tag.get('musicbrainz_trackid')
    result['release_track_id'] = mutagen_tag.get('musicbrainz_releasetrackid')
    result['release_group_id'] = mutagen_tag.get('musicbrainz_releasegroupid')

    return result
