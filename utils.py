import math
import os
import requests
import pandas as pd
import geopandas as gpd
import geemap
import geemap.colormaps as cm

import sys
import ee

def mask_s2_clouds(image):
  """Masks clouds in a Sentinel-2 image using the QA band.

  Args:
      image (ee.Image): A Sentinel-2 image.

  Returns:
      ee.Image: A cloud-masked Sentinel-2 image.
  """
  qa = image.select('QA60')

  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloud_bit_mask = 1 << 10
  cirrus_bit_mask = 1 << 11

  # Both flags should be set to zero, indicating clear conditions.
  mask = (
      qa.bitwiseAnd(cloud_bit_mask)
      .eq(0)
      .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
  )

  return image.updateMask(mask).divide(10000)


# Calculate Index

#Nari

def calculate_nari(img):
    green = img.select('B3')
    red_edge_1 = img.select('B5')
    nari = ((ee.Image.constant(1).divide(green)).subtract(ee.Image.constant(1).divide(red_edge_1))).divide((ee.Image.constant(1).divide(green)).add(ee.Image.constant(1).divide(red_edge_1)))
    return nari

# Ncri

def calculate_ncri(img):
    red_edge_1 = img.select('B5')
    red_edge_3 = img.select('B7')
    ncri = ((ee.Image.constant(1).divide(red_edge_1)).subtract(ee.Image.constant(1).divide(red_edge_3))).divide((ee.Image.constant(1).divide(red_edge_1)).add(ee.Image.constant(1).divide(red_edge_3)))
    return ncri

def gee_calculate_scrub_index(index: str=None, year: int=None):
    sentinel = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(f'{year}-09-01', f'{year}-11-01')
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 65))
                .map(mask_s2_clouds)
                .select(['B3', 'B5', 'B7'])
    )
    if index =='nari':
        nari = sentinel.map(calculate_nari)
        return nari.median()
    elif index == 'ncri':
        ncri = sentinel.map(calculate_ncri)
        return ncri.median()
    else:
        print('ERROR. No layer returned')
        
def calculate_ndvi(year: int=None):
    sentinel = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(f'{year}-01-01', f'{year+1}-01-01')
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 65))
                .map(mask_s2_clouds)
                .select(['B3', 'B5', 'B7'])
    )
    red = sentinel.select('B4')
    nir = sentinel.select('B8')
    ndvi = (red.subtract(nir)).divide(red.add(nir)).mean()
    return ndvi       

def get_era5_snowfall(date, poi):
    window_end = ee.Date(date)
    window_start = window_end.advance(ee.Number(-4), 'days')
    era5 = (
            ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
            .filterDate(window_start, window_end)
            .select('snowfall_sum')
            .sum()
            .multiply(1000)
        )
    return era5.sampleRegions(
        collection = poi,
        properties = ['name', 'datetime']
    )

def get_era5_snowcover(date, poi):
    window_end = ee.Date(date)
    window_start = window_end.advance(ee.Number(-4), 'days')
    era5 = (
            ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
            .filterDate(window_start, window_end)
            .select('snow_cover')
            .mean()
        )
    return era5.sampleRegions(
        collection = poi,
        properties = ['name', 'datetime']
)

def get_species_data(species_name, country_code, database = 'gbif'):
    """
    Retrieves observational data for a specific species using the GBIF API and returns it as a pandas DataFrame.

    Parameters:
    species_name (str): The scientific name of the species to query.
    country_code (str): The country code of the where the observation data will be queried.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the observational data.
    """
    if(database == 'iNaturalist'): user_id = input('Type in User-ID: ')
    
    match database:
        case 'gbif': base_url = "https://api.gbif.org/v1/occurrence/search"
        case 'iNaturalist': base_url = f"https://api.inaturalist.org/v1/observations?user_id={user_id}&quality_grade=needs_id&rank=genus"
    params = {
        "scientificName": species_name,
        "country": country_code,
        "hasCoordinate": "true",
        "basisOfRecord": "HUMAN_OBSERVATION",
        "limit": 10000,
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an exception for a response error.
        data = response.json()
        occurrences = data.get("results", [])

        if occurrences:  # If data is present
            df = pd.json_normalize(occurrences)
            return gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude), crs="EPSG:4326")[["species", "year", "month", "geometry"]]
        else:
            print("No data found for the given species and country code.")
            return None  # Returns an empty DataFrame
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None  # Returns an empty DataFrame in case of an exception
    
def remove_duplicates(data, grain_size, aoi):
    # Select one occurrence record per pixel at the chosen spatial resolution
    random_raster = ee.Image.random().reproject("EPSG:4326", None, grain_size).clip(aoi)
    rand_point_vals = random_raster.sampleRegions(
        collection=ee.FeatureCollection(data), geometries=True
    )
    return rand_point_vals.distinct("random")

def get_aoi_from_nuts(country_code:str = "AT", county_name:str=None):
    NUTS_0 = gpd.read_file(r"C:\Users\snied\Documents\GIS\general\ref-nuts-2024-01m.geojson\NUTS_RG_01M_2024_4326_LEVL_0.geojson")
    NUTS_2 = gpd.read_file(r"C:\Users\snied\Documents\GIS\general\ref-nuts-2024-01m.geojson\NUTS_RG_01M_2024_4326_LEVL_2.geojson")
    country = geemap.gdf_to_ee(NUTS_0.loc[NUTS_0.CNTR_CODE==country_code])
    if county_name:
        county = geemap.gdf_to_ee(NUTS_2.loc[NUTS_2.NUTS_NAME==county_name])  
    else:
        county = None
    
    return country, county

def get_layer_information(year: int):
    era5 = (
            ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
            .filter(ee.Filter.date(f"{year}-01-01", f"{year+1}-01-01"))
            )
    terrain = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003"))
    layer= {
            "CORINE Land Cover 2018" : ee.Image("COPERNICUS/CORINE/V20/100m/2018").select('landcover'),
            "BIOCLIM": ee.Image("projects/ee-sebasd1991/assets/BioClim"),
            "elevation" : ee.Image("USGS/SRTMGL1_003").select('elevation'),
            "slope" : terrain.select('slope'),
            "aspect" : terrain.select('aspect'),
            "northness" : terrain.select('aspect').multiply(math.pi/180).cos().rename('northness'),
            "eastness" : terrain.select('aspect').multiply(math.pi/180).sin().rename('eastness'),
            "NDVI" : (
                    ee.ImageCollection('LANDSAT/COMPOSITES/C02/T1_L2_8DAY_NDVI')
                    .filterDate(f"{year}-01-01", f"{year+1}-01-01")
                    .mean()
                ),
            "GHMI" : ee.ImageCollection("projects/sat-io/open-datasets/GHM/HM_2022_300M").first().rename('GHMI'),
            
            "SWE" : era5.select(['snow_depth_water_equivalent'], ['swe']).mean().rename('SWE'),
            "snow_depth" : era5.select(['snow_depth'], ['snow_depth']).mean(),
            "snow_cover" : era5.select(['snow_cover'], ['snow_cover']).mean(),
            "snow_albedo" : era5.select(['snow_albedo'], ['snow_albedo']).mean(),
            "CHM" : ee.ImageCollection("projects/sat-io/open-datasets/facebook/meta-canopy-height").mosaic().rename('CHM'),
            "NARI" : gee_calculate_scrub_index('nari', year).rename('NARI'),
            "NCRI" : gee_calculate_scrub_index('ncri', year).rename('NCRI')
        }
    return layer

def get_layer_visualization_params(layer_name: str):
    paletteHM = ['4c6100','adda25','e2ff9b','ffff73','ffe629','ffd37f','ffaa00','e69808','e60000','a80000','730000'];
    vis_params = {
        "elevation": {"min": 0, "max": 4000, "palette": cm.palettes['terrain']},
        "slope": {"min": 0, "max": 60, "palette": cm.palettes['viridis']},
        "NDVI": {"min": -1, "max": 1, "palette": cm.palettes['RdYlGn']},
        "CORINE Land Cover 2018": {"min": 1, "max": 44, "palette": cm.palettes['tab20']},
        "CHM": {"min": 0, "max": 25, "palette": cm.palettes['viridis']},
        "NARI": {"min": -1, "max": 1, "palette": cm.palettes['RdYlGn']},
        "NCRI": {"min": -1, "max": 1, "palette": cm.palettes['RdYlGn']},
        "GHMI": {"min": 0, "max": 1, "palette": paletteHM},
        "SWE": {"min": 0, "max": 500, "palette": cm.palettes['Blues']},
        "snow_depth": {"min": 0, "max": 200, "palette": cm.palettes['Blues']},
        "snow_cover": {"min": 0, "max": 100, "palette": cm.palettes['Blues']},
        "snow_albedo": {"min": 0, "max": 1, "palette": cm.palettes['Greys']},
        "BIOCLIM": {"min": 0, "max": 100, "palette": cm.palettes['viridis']},
        "aspect": {"min": 0, "max": 360, "palette": cm.palettes['hsv']},
        "northness": {"min": -1, "max": 1, "palette": cm.palettes['coolwarm']},
        "eastness": {"min": -1, "max": 1, "palette": cm.palettes['coolwarm']}
    }
    return vis_params.get(layer_name, {})
