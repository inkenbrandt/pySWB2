import requests
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from owslib.wms import WebMapService
from rasterio.transform import from_bounds
import numpy as np
from PIL import Image
import io
import ee
import geemap
import os


def clip_nlcd_esri(bbox, output_path, resolution=30):
    """
    Clip NLCD data for a given bounding box and save to file.

    Parameters:
    bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax) in WGS84
    output_path (str): Path to save the output GeoTIFF
    resolution (int): Desired resolution in meters (default 30m for NLCD)
    """
    # NLCD Image Service base URL
    base_url = "https://landscape10.arcgis.com/arcgis/rest/services/USA_NLCD_Land_Cover/ImageServer/exportImage"

    # Calculate image dimensions
    width = int((bbox[2] - bbox[0]) / resolution)
    height = int((bbox[3] - bbox[1]) / resolution)

    # Construct parameters for the export request
    params = {
        'bbox': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        'bboxSR': 4326,  # WGS84 EPSG code
        'size': f"{width},{height}",
        'imageSR': 4326,
        'format': 'tiff',
        'pixelType': '8_BIT_UNSIGNED',
        'noData': None,
        'f': 'json'
    }

    try:
        # Request the image
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        # Get the URL for the exported image
        result = response.json()
        if 'error' in result:
            raise Exception(f"API Error: {result['error']}")

        image_url = result.get('href')
        if not image_url:
            raise Exception("No image URL in response")

        # Download the actual image
        image_response = requests.get(image_url)
        image_response.raise_for_status()

        # Save to file
        with open(output_path, 'wb') as f:
            f.write(image_response.content)

        # Update the georeference information
        with rasterio.open(output_path, 'r+') as dst:
            transform = from_origin(bbox[0], bbox[3], resolution, resolution)
            dst.transform = transform
            dst.crs = CRS.from_epsg(4326)

        print(f"Successfully saved clipped NLCD data to {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")

def download_nlcd_wms(bbox, output_path, width=1000, height=1000, year=2021):
    """
    Download NLCD data from USGS GeoServer WMS.

    Parameters:
    bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax) in EPSG:4326
    output_path (str): Path to save the output GeoTIFF
    width (int): Output image width in pixels
    height (int): Output image height in pixels
    year (int): NLCD year
    """
    # WMS URL
    wms_url = "https://dmsdata.cr.usgs.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/wms"

    try:
        # Connect to WMS
        wms = WebMapService(wms_url, version='1.3.0')

        # Get layer name for specified year
        layer_name = f'mrlc_Land-Cover-Native_conus_{year}_data'

        if layer_name not in wms.contents:
            available_years = [l.split('_')[-2] for l in wms.contents.keys()
                             if l.startswith('mrlc_Land-Cover-Native_conus_')]
            raise ValueError(f"Year {year} not available. Available years: {', '.join(available_years)}")

        # Request the image
        img = wms.getmap(layers=[layer_name],
                        srs='EPSG:4326',
                        bbox=bbox,
                        size=(width, height),
                        format='image/tiff',
                        transparent=True)

        # Save to file and set proper georeference
        with open(output_path, 'wb') as f:
            f.write(img.read())

        # Update georeference information
        with rasterio.open(output_path, 'r+') as dst:
            # Calculate transform
            transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3],
                                  width, height)

            # Update metadata
            dst.transform = transform
            dst.crs = 'EPSG:4326'

        print(f"Successfully downloaded NLCD {year} data to {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")

def download_nlcd_gee(bbox, output_path, year=2019):
    """
    Download NLCD data from Google Earth Engine for a given bounding box.

    Parameters:
    bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax) in WGS84
    output_path (str): Path to save the output GeoTIFF
    year (int): NLCD year (2001, 2004, 2006, 2008, 2011, 2013, 2016, or 2019)
    """
    # Initialize Earth Engine
    try:
        ee.Initialize()
    except Exception as e:
        print("Please authenticate using 'earthengine authenticate' in terminal first")
        raise e

    # Create geometry from bbox
    roi = ee.Geometry.Rectangle(bbox)

    # Get NLCD dataset
    dataset = ee.ImageCollection("USGS/NLCD_RELEASES/2019_REL/NLCD")

    # Filter to get the land cover for specified year
    nlcd = dataset.filter(ee.Filter.eq('system:index', f'L{year}')).first() \
        .select('landcover')

    # Clip to region of interest
    clipped = nlcd.clip(roi)

    # Create download URL
    try:
        url = clipped.getDownloadURL({
            'scale': 30,  # Native NLCD resolution
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF',
            'region': roi
        })

        # Use geemap to download the file
        geemap.download_file(url, output_path)

        print(f"Successfully downloaded NLCD {year} data to {output_path}")

    except Exception as e:
        print(f"Error downloading data: {str(e)}")

def get_nlcd_class_names(value):
    """Return NLCD class name for a given value."""
    classes = {
        11: 'Open Water',
        12: 'Perennial Ice/Snow',
        21: 'Developed, Open Space',
        22: 'Developed, Low Intensity',
        23: 'Developed, Medium Intensity',
        24: 'Developed, High Intensity',
        31: 'Barren Land',
        41: 'Deciduous Forest',
        42: 'Evergreen Forest',
        43: 'Mixed Forest',
        51: 'Dwarf Scrub',
        52: 'Shrub/Scrub',
        71: 'Grassland/Herbaceous',
        72: 'Sedge/Herbaceous',
        73: 'Lichens',
        74: 'Moss',
        81: 'Pasture/Hay',
        82: 'Cultivated Crops',
        90: 'Woody Wetlands',
        95: 'Emergent Herbaceous Wetlands'
    }
    return classes.get(value, 'Unknown')



# Example usage
if __name__ == "__main__":
    # Example bbox for Washington, DC area
    dc_bbox = (-77.12, 38.79, -76.91, 38.995)

    clip_nlcd(
        bbox=dc_bbox,
        output_path="dc_nlcd.tiff",
        resolution=30
    )