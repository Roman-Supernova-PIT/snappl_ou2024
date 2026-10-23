# this is a module level file
import pytest # noqa: F401

import tox # noqa: F401
from tox.pytest import init_fixture # noqa: F401

from snappl.imagecollection import ImageCollection


@pytest.fixture( scope="session" )
def ou2024collection():
    return ImageCollection.get_collection( 'ou2024' )


@pytest.fixture( scope='session' )
def ou2024imagerelpath():
    return 'Y106/13205/Roman_TDS_simple_model_Y106_13205_1.fits.gz'


@pytest.fixture( scope='module' )
def ou2024imagepath( ou2024imagerelpath, ou2024collection ):
    return str( ou2024collection.base_path / ou2024imagerelpath )


@pytest.fixture
def ou2024image( ou2024collection, ou2024imagerelpath ):
    return ou2024collection.get_image( path=ou2024imagerelpath )


# If you use this next fixture, you aren't supposed
#   to modify the image!  Make sure any modifications
#   you make are undone at the end of your test.
@pytest.fixture( scope='module' )
def ou2024image_module(  ou2024collection, ou2024imagerelpath ):
    image = ou2024collection.get_image( path=ou2024imagerelpath )
    image.get_data( which='all', cache=True )
    image.get_wcs()
    return image
