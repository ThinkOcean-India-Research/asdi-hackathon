import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt
import geopandas as gpd
import utils.solaris.solaris as sol
from shapely import speedups


speedups.disable()


SATELLITE_IMAGERY_PATH = 'data/AOI_1_rio/3band'
GEOJSON_PATH = 'data/AOI_1_rio/geojson'

filename = 'AOI_1_RIO_img1200'
m = sol.vector.mask.df_to_px_mask(
    df = gpd.read_file(f'{GEOJSON_PATH}/Geo_{filename}.geojson'),
    reference_im = f'{SATELLITE_IMAGERY_PATH}/3band_{filename}.tif',
    channels=['footprint', 'boundary', 'contact'],
    boundary_width=5, contact_spacing=10, meters=True
)
plt.imshow(m)