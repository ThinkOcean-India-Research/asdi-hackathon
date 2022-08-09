import os
import shapely
from affine import Affine
import rasterio
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from shapely.geometry import box, Polygon
import pandas as pd
import geopandas as gpd
from rtree.core import RTreeError
import shutil
import geopandas as gpd
import pyproj
from distutils.version import LooseVersion
import numpy as np
from shapely.geometry import shape, mapping
from shapely.wkt import loads
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from osgeo import gdal
from rasterio import features


## solaris is quite messed up, so this is a mix of functions i needed adapted from their source code

def footprint_mask(df, out_file=None, reference_im=None, geom_col='geometry',
                   do_transform=None, affine_obj=None, shape=(900, 900),
                   out_type='int', burn_value=255, burn_field=None):
    # start with required checks and pre-population of values
    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')
    df = gpd.read_file(df) 

    if len(df) == 0 and not out_file:
        return np.zeros(shape=shape, dtype='uint8')

    if do_transform is None:
        # determine whether or not transform should be done
        do_transform = _check_do_transform(df, reference_im, affine_obj)

    df[geom_col] = df[geom_col].apply(_check_geom)  # load in geoms if wkt
    if not do_transform:
        affine_obj = Affine(1, 0, 0, 0, 1, 0)  # identity transform

    if reference_im:
        reference_im = rasterio.open(reference_im)
        shape = reference_im.shape
        if do_transform:
            affine_obj = reference_im.transform

    # extract geometries and pair them with burn values
    if burn_field:
        if out_type == 'int':
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('uint8')))
        else:
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('float32')))
    else:
        feature_list = list(zip(df[geom_col], [burn_value]*len(df)))

    if len(df) > 0:
        output_arr = features.rasterize(shapes=feature_list, out_shape=shape,
                                        transform=affine_obj)
    else:
        output_arr = np.zeros(shape=shape, dtype='uint8')
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        if out_type == 'int':
            meta.update(dtype='uint8')
            meta.update(nodata=0)
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr

def _check_geom(geom):
    """Check if a geometry is loaded in.
    Returns the geometry if it's a shapely geometry object. If it's a wkt
    string or a list of coordinates, convert to a shapely geometry.
    """
    if isinstance(geom, BaseGeometry):
        return geom
    elif isinstance(geom, str):  # assume it's a wkt
        return loads(geom)
    elif isinstance(geom, list) and len(geom) == 2:  # coordinates
        return Point(geom)

def _check_do_transform(df, reference_im, affine_obj):
    """Check whether or not a transformation should be performed."""
    try:
        crs = getattr(df, 'crs')
    except AttributeError:
        return False  # if it doesn't have a CRS attribute

    if not crs:
        return False  # return False for do_transform if crs is falsey
    elif crs and (reference_im is not None or affine_obj is not None):
        # if the input has a CRS and another obj was provided for xforming
        return True

def geojson_to_px_gdf(geojson:str, im_path:str, geom_col='geometry', precision=None,
                      output_path=None, override_crs=False):

    # get the bbox and affine transforms for the image
    im = rasterio.open(im_path)
    # make sure the geo vector data is loaded in as geodataframe(s)
    gdf = gpd.read_file(geojson)

    if len(gdf):  # if there's at least one geometry
        if override_crs:
            gdf.crs = im.crs 
        overlap_gdf = get_overlapping_subset(gdf, im)
    else:
        overlap_gdf = gdf

    affine_obj = im.transform
    transformed_gdf = affine_transform_gdf(overlap_gdf, affine_obj=affine_obj,
                                           inverse=True, precision=precision,
                                           geom_col=geom_col)
    transformed_gdf['image_fname'] = os.path.split(im_path)[1]

    if output_path is not None:
        if output_path.lower().endswith('json'):
            transformed_gdf.to_file(output_path, driver='GeoJSON')
        else:
            transformed_gdf.to_csv(output_path, index=False)
    return transformed_gdf



def get_overlapping_subset(gdf, im=None, bbox=None, bbox_crs=None):
    """Extract a subset of geometries in a GeoDataFrame that overlap with `im`.
    Notes
    -----
    This function uses RTree's spatialindex, which is much faster (but slightly
    less accurate) than direct comparison of each object for overlap.
    Arguments
    ---------
    gdf : :class:`geopandas.GeoDataFrame`
        A :class:`geopandas.GeoDataFrame` instance or a path to a geojson.
    im : :class:`rasterio.DatasetReader` or `str`, optional
        An image object loaded with `rasterio` or a path to a georeferenced
        image (i.e. a GeoTIFF).
    bbox : `list` or :class:`shapely.geometry.Polygon`, optional
        A bounding box (either a :class:`shapely.geometry.Polygon` or a
        ``[bottom, left, top, right]`` `list`) from an image. Has no effect
        if `im` is provided (`bbox` is inferred from the image instead.) If
        `bbox` is passed and `im` is not, a `bbox_crs` should be provided to
        ensure correct geolocation - if it isn't, it will be assumed to have
        the same crs as `gdf`.
    bbox_crs : int, optional
        The coordinate reference system that the bounding box is in as an EPSG
        int. If not provided, it's assumed that the CRS is the same as `im`
        (if provided) or `gdf` (if not).
    Returns
    -------
    output_gdf : :class:`geopandas.GeoDataFrame`
        A :class:`geopandas.GeoDataFrame` with all geometries in `gdf` that
        overlapped with the image at `im`.
        Coordinates are kept in the CRS of `gdf`.
    """
    if im is None and bbox is None:
        raise ValueError('Either `im` or `bbox` must be provided.')
    # gdf = _check_gdf_load(gdf)
    sindex = gdf.sindex
    if im is not None:
        # im = _check_rasterio_im_load(im)
        # currently, convert CRSs to WKT strings here to accommodate rasterio.
        bbox = transform_bounds(im.crs, _check_crs(gdf.crs, return_rasterio=True),
                                *im.bounds)
        bbox_crs = im.crs
    # use transform_bounds in case the crs is different - no effect if not
    if isinstance(bbox, Polygon):
        bbox = bbox.bounds
    if bbox_crs is None:
        try:
            bbox_crs = _check_crs(gdf.crs, return_rasterio=True)
        except AttributeError:
            raise ValueError('If `im` and `bbox_crs` are not provided, `gdf`'
                             'must provide a coordinate reference system.')
    else:
        bbox_crs = _check_crs(bbox_crs, return_rasterio=True)
    # currently, convert CRSs to WKT strings here to accommodate rasterio.
    bbox = transform_bounds(bbox_crs,
                            _check_crs(gdf.crs, return_rasterio=True),
                            *bbox)
    try:
        intersectors = list(sindex.intersection(bbox))
    except RTreeError:
        intersectors = []

    return gdf.iloc[intersectors, :]

def _check_crs(input_crs, return_rasterio=False):
    """Convert CRS to the ``pyproj.CRS`` object passed by ``solaris``."""
    if not isinstance(input_crs, pyproj.CRS) and input_crs is not None:
        out_crs = pyproj.CRS(input_crs)
    else:
        out_crs = input_crs

    if return_rasterio:
        if LooseVersion(rasterio.__gdal_version__) >= LooseVersion("3.0.0"):
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt())
        else:
            out_crs = rasterio.crs.CRS.from_wkt(out_crs.to_wkt("WKT1_GDAL"))

    return out_crs


def affine_transform_gdf(gdf, affine_obj, inverse=False, geom_col="geometry",
                         precision=None):
    if isinstance(gdf, str):  # assume it's a geojson
        if gdf.lower().endswith('json'):
            gdf = gpd.read_file(gdf)
        elif gdf.lower().endswith('csv'):
            gdf = pd.read_csv(gdf)
        else:
            raise ValueError(
                "The file format is incompatible with this function.")
    if 'geometry' not in gdf.columns:
        gdf = gdf.rename(columns={geom_col: 'geometry'})
    if not isinstance(gdf['geometry'][0], Polygon):
        gdf['geometry'] = gdf['geometry'].apply(shapely.wkt.loads)
    gdf["geometry"] = gdf["geometry"].apply(convert_poly_coords,
                                            affine_obj=affine_obj,
                                            inverse=inverse)
    if precision is not None:
        gdf['geometry'] = gdf['geometry'].apply(
            _reduce_geom_precision, precision=precision)

    # the CRS is no longer valid - remove it
    gdf.crs = None

    return gdf

def convert_poly_coords(geom, raster_src=None, affine_obj=None, inverse=False,
                        precision=None):
    """Georegister geometry objects currently in pixel coords or vice versa.
    Arguments
    ---------
    geom : :class:`shapely.geometry.shape` or str
        A :class:`shapely.geometry.shape`, or WKT string-formatted geometry
        object currently in pixel coordinates.
    raster_src : str, optional
        Path to a raster image with georeferencing data to apply to `geom`.
        Alternatively, an opened :class:`rasterio.Band` object or
        :class:`osgeo.gdal.Dataset` object can be provided. Required if not
        using `affine_obj`.
    affine_obj: list or :class:`affine.Affine`
        An affine transformation to apply to `geom` in the form of an
        ``[a, b, d, e, xoff, yoff]`` list or an :class:`affine.Affine` object.
        Required if not using `raster_src`.
    inverse : bool, optional
        If true, will perform the inverse affine transformation, going from
        geospatial coordinates to pixel coordinates.
    precision : int, optional
        Decimal precision for the polygon output. If not provided, rounding
        is skipped.
    Returns
    -------
    out_geom
        A geometry in the same format as the input with its coordinate system
        transformed to match the destination object.
    """

    if not raster_src and not affine_obj:
        raise ValueError("Either raster_src or affine_obj must be provided.")

    if raster_src is not None:
        affine_xform = get_geo_transform(raster_src)
    else:
        if isinstance(affine_obj, Affine):
            affine_xform = affine_obj
        else:
            # assume it's a list in either gdal or "standard" order
            # (list_to_affine checks which it is)
            if len(affine_obj) == 9:  # if it's straight from rasterio
                affine_obj = affine_obj[0:6]
            affine_xform = list_to_affine(affine_obj)

    if inverse:  # geo->px transform
        affine_xform = ~affine_xform

    if isinstance(geom, str):
        # get the polygon out of the wkt string
        g = shapely.wkt.loads(geom)
    elif isinstance(geom, shapely.geometry.base.BaseGeometry):
        g = geom
    else:
        raise TypeError('The provided geometry is not an accepted format. '
                        'This function can only accept WKT strings and '
                        'shapely geometries.')

    xformed_g = shapely.affinity.affine_transform(g, [affine_xform.a,
                                                      affine_xform.b,
                                                      affine_xform.d,
                                                      affine_xform.e,
                                                      affine_xform.xoff,
                                                      affine_xform.yoff])
    if isinstance(geom, str):
        # restore to wkt string format
        xformed_g = shapely.wkt.dumps(xformed_g)
    if precision is not None:
        xformed_g = _reduce_geom_precision(xformed_g, precision=precision)

    return xformed_g

def _reduce_geom_precision(geom, precision=2):
    geojson = mapping(geom)
    geojson['coordinates'] = np.round(np.array(geojson['coordinates']),
                                      precision)
    return shape(geojson)

def get_geo_transform(raster_src):
    """Get the geotransform for a raster image source.
    Arguments
    ---------
    raster_src : str, :class:`rasterio.DatasetReader`, or `osgeo.gdal.Dataset`
        Path to a raster image with georeferencing data to apply to `geom`.
        Alternatively, an opened :class:`rasterio.Band` object or
        :class:`osgeo.gdal.Dataset` object can be provided. Required if not
        using `affine_obj`.
    Returns
    -------
    transform : :class:`affine.Affine`
        An affine transformation object to the image's location in its CRS.
    """

    if isinstance(raster_src, str):
        affine_obj = rasterio.open(raster_src).transform
    elif isinstance(raster_src, rasterio.DatasetReader):
        affine_obj = raster_src.transform
    elif isinstance(raster_src, gdal.Dataset):
        affine_obj = Affine.from_gdal(*raster_src.GetGeoTransform())

    return affine_obj

def list_to_affine(xform_mat):
    """Create an Affine from a list or array-formatted [a, b, d, e, xoff, yoff]
    Arguments
    ---------
    xform_mat : `list` or :class:`numpy.array`
        A `list` of values to convert to an affine object.
    Returns
    -------
    aff : :class:`affine.Affine`
        An affine transformation object.
    """
    # first make sure it's not in gdal order
    if len(xform_mat) > 6:
        xform_mat = xform_mat[0:6]
    if rasterio.transform.tastes_like_gdal(xform_mat):
        return Affine.from_gdal(*xform_mat)
    else:
        return Affine(*xform_mat)