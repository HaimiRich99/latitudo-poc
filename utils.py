import cv2
import rasterio
import numpy as np 

import matplotlib.pyplot as plt

# Geocoding
import rasterio.windows
from shapely.geometry import box
from geopy.geocoders import Nominatim



def geocode_to_bbox(location: str, buffer_km: float = 5):
    geolocator = Nominatim(user_agent="sentinel-assistant")
    loc = geolocator.geocode(location)
    if not loc:
        raise ValueError(f"Could not geocode location: {location}")
    lat, lon = loc.latitude, loc.longitude
    # Create a small box around the point (roughly 0.05 deg ~ 5 km)
    return (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)

def raster_downloader(mappa,asset_key, channel = 1):
    """Download a raster asset and return it as a numpy array."""
    asset_url = mappa.assets[asset_key].href
    with rasterio.open(asset_url) as src:
        window = rasterio.windows.Window(0, 0, width=100, height=100)
        data = src.read(channel, window=window)  
        print(f"{asset_key} band shape:", data.shape)
    return data.astype(np.float32)

def normalized_index_calculator(band1,band2):
    """
    Calculate a normalized index (e.g., NDVI) from two bands.
    
    Parameters:
        band1 (np.array): First band array (e.g., NIR).
        band2 (np.array): Second band array (e.g., Red).
    
    Returns:
        np.array: Normalized index array.
    """
    # NDVI calculation: (NIR - Red) / (NIR + Red)
    index = (band1 - band2) / (band1 + band2 + 1e-10)  # Add small constant to avoid division by zero
    index = np.clip(index, -1, 1)  # NDVI is typically in [-1, 1]

    '''
    plt.figure(figsize=(8, 8))
    plt.imshow(index, cmap='RdYlGn')
    plt.colorbar(label='NDVI')
    plt.axis('off')
    plt.title("NDVI (Normalized Difference Vegetation Index)")
    plt.show()'''

    return index

import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for resizing

def display_band(band_array, title=None, cmap_color="gray", 
                 vmin_percentile=2, vmax_percentile=98, resize_factor=0.25):
    """
    Display a satellite band as a greyscale image, optionally resized for speed.

    Parameters:
        band_array (np.array): 2D array representing the band.
        title (str): Title for the image display.
        cmap_color (str): Colormap to use (default: 'gray').
        vmin_percentile (float): Lower percentile for normalization (default: 2).
        vmax_percentile (float): Upper percentile for normalization (default: 98).
        resize_factor (float): Scaling factor to downsample the image (default: 0.25).

    Returns:
        figure: Matplotlib figure object displaying the band.
    """
    # Resize using OpenCV if factor < 1
    if resize_factor < 1.0:
        new_height = int(band_array.shape[0] * resize_factor)
        new_width = int(band_array.shape[1] * resize_factor)
        band_array = cv2.resize(band_array, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Normalize based on percentiles
    vmin, vmax = np.percentile(band_array, (vmin_percentile, vmax_percentile))
    band_norm = np.clip((band_array - vmin) / (vmax - vmin), 0, 1)

    # Plotting
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(band_norm, cmap=cmap_color, vmin=0, vmax=1)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig



def display_stacked_bands(band_array_list, titles=None, cmap_color="gray", vmin_percentile=2, vmax_percentile=98):
    """
    Display multiple satellite bands as a stacked image. For example, RGB bands.

    Parameters:
        band_arrays (list of np.array): List of 2D arrays representing the bands.
        titles (list of str): Titles for each band (optional).
        vmin_percentile (float): Lower percentile for normalization (default: 2).
        vmax_percentile (float): Upper percentile for normalization (default: 98).
    """
    stacked_bands = np.stack(band_array_list ,axis=-1).astype(np.float32)
    display_band(stacked_bands,
                 title=titles, cmap_color=cmap_color,
                 vmin_percentile=vmin_percentile, vmax_percentile=vmax_percentile)
    

def decider(mappa, bands, frequency_operation=None):
    """
    Decide how to display the bands based on the number of bands and the operation.
    
    Parameters:
        mappa: The map object containing the assets.
        bands: List of band names to display.
        frequency_operation: Operation to perform on multiple bands (None, "stack", "combine").
        
    Returns:
        Matplotlib figure displaying the appropriate band(s).
    """
    if len(bands) > 1 and frequency_operation == 'combine':
        band1 = raster_downloader(mappa, asset_key=bands[0])
        band2 = raster_downloader(mappa, asset_key=bands[1])
        index = normalized_index_calculator(band1, band2)
        fig = display_band(index)
    elif len(bands) > 1 and frequency_operation == 'stack':
        arrays = [raster_downloader(mappa, asset_key=band) for band in bands]
        fig = display_stacked_bands(arrays)
    else:
        array = raster_downloader(mappa, asset_key=bands[0])
        fig = display_band(array)
    return fig