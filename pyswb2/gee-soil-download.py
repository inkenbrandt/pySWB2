import ee
import geemap
import os

def initialize_gee():
    """Initialize Earth Engine"""
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()

def download_soil_data(region, output_dir="soil_data", scale=30):
    """
    Download soil data from Google Earth Engine
    
    Parameters:
    region (ee.Geometry): Region of interest
    output_dir (str): Directory to save the output
    scale (int): Resolution in meters
    
    Returns:
    str: Path to downloaded file
    """
    # Initialize Earth Engine
    initialize_gee()
    
    # Load the NRCS soil dataset
    soil_data = ee.ImageCollection("USDA/NRCS/SSURGO/soil_properties")
    
    # Get available water storage (AWS) at different depths
    aws = soil_data.select(['aws0_5cm', 'aws5_15cm', 'aws15_30cm', 'aws30_60cm'])
    
    # Calculate mean values across the time series
    aws_mean = aws.mean()
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the export parameters
    export_params = {
        'image': aws_mean,
        'description': 'soil_aws_data',
        'folder': output_dir,
        'region': region,
        'scale': scale,
        'crs': 'EPSG:4326',
        'fileFormat': 'GeoTIFF'
    }
    
    # Download the data using geemap
    output_file = os.path.join(output_dir, 'soil_aws_data.tif')
    geemap.ee_export_image(**export_params, filename=output_file)
    
    return output_file

if __name__ == "__main__":
    # Example usage for a rectangular area
    # Define a rectangle in Kansas (modify coordinates as needed)
    bbox = [-98.5, 39.0, -98.4, 39.1]  # [west, south, east, north]
    region = ee.Geometry.Rectangle(bbox)
    
    try:
        # Download the data
        output_path = download_soil_data(
            region=region,
            output_dir="gee_soil_data",
            scale=30  # 30-meter resolution
        )
        print(f"Data successfully downloaded to: {output_path}")
        
        # Create a quick visualization using geemap
        Map = geemap.Map()
        Map.add_basemap('HYBRID')
        Map.centerObject(region, 12)
        Map.addLayer(ee.Image(output_path), {'bands': ['aws0_5cm']}, 'AWS 0-5cm')
        Map.save('soil_map.html')
        print("Map saved as soil_map.html")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
