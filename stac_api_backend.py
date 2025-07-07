# Query
from pystac_client import Client
from shapely.geometry import box

from datetime import datetime
from dateutil.relativedelta import relativedelta

# Functions
from utils import *

import pandas as pd


# Connect to STAC API
api_url = "https://earth-search.aws.element84.com/v1"
catalog = Client.open(api_url)


def mock_sentinel_query(location, start_date, end_date, bands, cloud_cover):
    return {
        "status": "success",
        "message": f"Querying Sentinel-2 for {location} from {start_date} to {end_date or start_date}, bands: {bands}, cloud_cover < {cloud_cover}%"
    }


# Search parameters
def stac_api_query(location, start_date, end_date, cloud_cover):
    """    Query the STAC API for Sentinel-2 images based on location, date range, and cloud cover.
    Args:
        location (str or tuple): Location as a string (e.g., "Milan") or a geojson geometry (e.g., a tuple with min_lon, min_lat, max_lon, max_lat).
        start_date (str): Start date in ISO format (e.g., "2023-07-01").
        end_date (str, optional): End date in ISO format (e.g., "2023-07-31"). Defaults to None.
        cloud_cover (int): Maximum cloud cover percentage (0-100).
    Returns:
        list: List of Sentinel-2 items matching the query. An item is a dictionary containing metadata about the satellite image.
    """
    
    # Location could be a string (e.g. "Milan") or a geojson geometry  (e.g. a tuple with min_lon, min_lat, max_lon, max_lat)
    if type(location)==str:
        aoi = box(*geocode_to_bbox(location, buffer_km=5))
    elif type(location)== tuple:
        aoi = box(location)
    
    # Date Range in cui cercare immagini satellitari
    if end_date:
        dates = f"{start_date}/{end_date}"
    else:
        # add a time window to ensure we get at least one image
        # Note: date is a string in ISO format (e.g. "2023-07-01")
        # Parse the start date from ISO format string
        start_date = datetime.fromisoformat(start_date)

        # No need to replace the day if you're not changing it
        new_start_date = start_date

        # Safely add 2 months (relativedelta handles varying month lengths)
        end_date = start_date + relativedelta(months=+2)

        # Format the date range as a string
        dates = f"{new_start_date.isoformat()}/{end_date.isoformat()}"

    results = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi,
        datetime=dates,
        query={"eo:cloud_cover": {"lt": cloud_cover}},
        limit=5)
    
    print(results)
    # Convert results to a list of items
    if results is None or results.matched == 0:
        return {"status": "error", "message": "No results found for the given query."}
    else:
        results_list = results.items()
        # Invertiamo l'ordine dei risultati per avere le immagini più vecchie per prime
        results_list = list(results.items())[::-1]
        
    return results_list

def results_to_df(results_list, bands):
    results_dict = {}
    for result in results_list:
        date_id = result.datetime.strftime("%Y-%m-%d")
        results_dict[date_id] = {}
        for band in bands:
            results_dict[date_id][band] = result.assets[band].href
    # Create a df with dates as rows and bands as columns
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    df.index.name = 'Date'

    return df
