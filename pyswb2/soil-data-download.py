from arcgis.gis import GIS
from arcgis.raster import ImageServer
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os

def download_soil_data(bbox, output_file="soil_data.tif", resolution=0.001):
    """
    Download soil water storage data for a specified bounding box.
    
    Parameters:
    bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax) in WGS84
    output_file (str): Name of output GeoTIFF file
    resolution (float): Spatial resolution in degrees
    
    Returns:
    str: Path to saved file
    """
    # Connect to ArcGIS
    gis = GIS()
    
    # Connect to the image service
    url = "https://landscape11.arcgis.com/arcgis/rest/services/USA_Soils_Available_Water_Storage/ImageServer"
    image_service = ImageServer(url)
    
    # Calculate dimensions based on bbox and resolution
    width = int((bbox[2] - bbox[0]) / resolution)
    height = int((bbox[3] - bbox[1]) / resolution)
    
    # Download the data
    raster_data = image_service.export_image(
        bbox=bbox,
        bbox_sr=4326,  # WGS84 EPSG code
        size=[width, height],
        export_format='TIFF',
        pixel_type='F32',  # 32-bit float
        no_data=None
    )
    
    # Get the pixel values
    pixels = raster_data.pixels
    
    # Create the GeoTIFF
    transform = from_origin(bbox[0], bbox[3], resolution, resolution)
    
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=pixels.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(pixels, 1)
        
    return output_file

if __name__ == "__main__":
    # Example usage: Download data for a small area in the USA
    # Coordinates for a sample area (modify as needed)
    sample_bbox = (-98.5, 39.0, -98.4, 39.1)  # Small area in Kansas
    
    try:
        output_path = download_soil_data(
            bbox=sample_bbox,
            output_file="kansas_soil_data.tif",
            resolution=0.001  # Approximately 100m at this latitude
        )
        print(f"Data successfully downloaded and saved to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
