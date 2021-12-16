import pandas as pd
import geopandas as gpd
import numpy as np
import math

def sum_points_in_poly(poly_gdf, point_gdf, col_name, groups):
    
    # Set coordinate system for meter distance measures
    poly_gdf = poly_gdf.to_crs("EPSG:4326")
    point_gdf = point_gdf.to_crs("EPSG:4326")

    # Sum of points in polygon
    pointInPoly = gpd.sjoin(point_gdf, poly_gdf, op='within') 

    data = pd.DataFrame(pointInPoly.groupby(groups).size()).rename(columns = {0: f'sum_{col_name}'}).reset_index()
    
    # If grouping by more than polygon level - need to unstack output
    if len(groups) > 1:
        
        groups.reverse()
        data_t = data.set_index(groups).unstack(level = 0)
        data_t.columns = data_t.columns.droplevel(0)
        data_t.replace({np.nan : 0}, inplace = True)
        data_t[f"sum_industrial_sites"] = data_t.sum(axis=1)
        
        return data_t
        
    return data

def sum_lines_in_poly(lines_gdf, poly_gdf):
    
    poly_gdf = poly_gdf[['geometry']].copy()
    
    # Set coordinate system for meter distance measures
    lines_gdf = lines_gdf.to_crs("EPSG:4326")
    
    sjoin = gpd.sjoin(lines_gdf, poly_gdf, how='inner', op='within')
    
    poly_gdf.reset_index(inplace = True)
    
    poly_gdf['lines_within_geoms'] = poly_gdf['index'].apply(lambda x: sjoin[sjoin['index_right'] == x]['geometry'].tolist())
    
    poly_gdf['result'] = None

    for index, row in poly_gdf.iterrows():
        sum_intersections = 0

        for i in range(len(row['lines_within_geoms'])):
            sum_intersections += row['geometry'].intersection(row['lines_within_geoms'][i]).length
        poly_gdf.loc[index, 'result'] = sum_intersections

    poly_gdf.drop(['lines_within_geoms'], axis = 1, inplace = True)
    
    return poly_gdf['result']

## Well exposure feature engineering ###

from requests import get
from pandas import json_normalize
import rasterio

def get_slope_aspect(raster_path, locations_df):
    
    with rasterio.open(raster_path) as src:
        ndvi = src.read(1)
     
    # raster CRS
    crs = str(src.crs)
    
    # set locations df to crs of raster
    locations_df = locations_df.to_crs(crs)
    
    # Get slope aspect for each point
    slope_aspect = []
    for row in locations_df.iterrows():
        r, c = src.index(row[1]['geometry'].x, row[1]['geometry'].y)
        slope_aspect.append(ndvi[r, c])

    return slope_aspect
    
def get_elevation(_lat = None, _long = None):
    '''
        script for returning elevation in m from lat, long
    '''
    if _lat is None or _long is None: return None
    
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={_lat},{_long}')
    
    # Request with a timeout for slow responses
    r = get(query, timeout = 20)

    # Only get the json response in case of 200 or 201
    if r.status_code == 200 or r.status_code == 201:
        elevation = json_normalize(r.json(), 'results')['elevation'].values[0]
    else: 
        elevation = None
    return elevation

def get_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)

    # convert_bearing_to_360_north_circle
    if brng < 0:
        brng += 360
    
    return brng

def slope_phase(soil_phase_types, soil):
    
    slope_phase_types = slope_phase_types.str.lower()
    
    soil['slope_phase'] = np.nan
    for slope in slope_phase_types:
        soil['slope_phase'] = np.where(soil['MUNAME'].str.contains(slope), slope , soil['slope_phase'])
        
    return soil['slope_phase']

def soil_series(soil_series, soil):
    
    soil_series = soil_series.str.lower()
    
    # while soil series is not nan - populate primary, secondary, tertiary, quarternary. 
    soil_split = soil['MUNAME'].str.split('-', expand=True)
    
    primary_series = soil_split[0].apply(lambda x: x.split()[0])
    secondary_series = soil_split[1].replace({None : '0'}).apply(lambda x: x.split()[0])
    tertiary_series = soil_split[2].replace({None : '0'}).apply(lambda x: x.split()[0])
    
    soil['primary_series'] = np.nan
    soil['secondary_series'] = np.nan
    soil['tertiary_series'] = np.nan

    for series in soil_series:
        soil['primary_series'] = np.where(primary_series.str.contains(series), series , soil['primary_series'])
        soil['secondary_series'] = np.where(secondary_series.str.contains(series), series , soil['secondary_series'])
        soil['tertiary_series'] = np.where(tertiary_series.str.contains(series), series , soil['tertiary_series'])
        
    return soil['primary_series'], soil['secondary_series'], soil['tertiary_series']

def soil_texture(soil_texture, soil):
    
    soil_texture = soil_texture.str.lower()
    
    soil_split = soil['MUNAME'].str.split('-', expand=True)
    
    primary_texture = soil_split[0]
    secondary_texture = soil_split[1]
    tertiary_texture = soil_split[2]
    
    soil['primary_texture'] = np.nan
    soil['secondary_texture'] = np.nan
    soil['tertiary_texture'] = np.nan

    for texture in soil_texture:
        soil['primary_texture'] = np.where(primary_texture.str.contains(texture), texture , soil['primary_texture'])
        soil['secondary_texture'] = np.where(secondary_texture.str.contains(texture), texture , soil['secondary_texture'])
        soil['tertiary_texture'] = np.where(tertiary_texture.str.contains(texture), texture , soil['tertiary_texture'])
        
    return soil['primary_texture'], soil['secondary_texture'], soil['tertiary_texture']

def extra_desc(soil_extra_desc, soil):
    
    soil_extra_desc = soil_extra_desc.str.lower()
    
    soil_split_desc = soil['MUNAME'].str.split(',', expand=True)
    
    soil['extra_desc'] = np.nan

    for desc in soil_extra_desc:
        soil['extra_desc'] = np.where(soil_split_desc[1].str.contains(desc), desc , soil['extra_desc'])
        soil['extra_desc'] = np.where(soil_split_desc[2].str.contains(desc), desc , soil['extra_desc'])
        soil['extra_desc'] = np.where(soil_split_desc[3].str.contains(desc), desc , soil['extra_desc'])
    
    return soil['extra_desc']

def map_unit_groups(soil):
    
    map_unit_numbers = soil['MUSYM'].str.extract('(\d+)').replace({np.nan : 0}).astype(int)
    
    func = np.vectorize(conditions)
    soil['map_unit_groups'] = func(map_unit_numbers)
    
    return soil['map_unit_groups']
    
    
def map_unit_conditions(x):
    if (x >= 1) and (x <= 99):
        return "Water, poorly, and very poorly drained soils"
    elif (x >= 100) and (x <= 199):
        return "Mapunits controlled by bedrock"
    elif (x >= 200) and (x <= 299):
        return "Excessively drained to somewhat poorly drained soils formed in glacial outwash or lacustrine sediments"
    elif (x >= 300) and (x <= 399):
        return "Excessively drained to somewhat poorly drained soils formed in dense glacial till"
    elif (x >= 400) and (x <= 599):
        return "Excessively drained to somewhat poorly drained soils formed in glacial till"
    elif (x >= 600) and (x <= 699):
        return "Miscellaneous land types, urban land, and soils above series levels"
    elif (x >= 700) and (x <= 799):
        return "Excessively drained to somewhat poorly drained soils formed in lacustrine or marine material"
    elif (x >= 900) and (x <= 999):
        return "Two or more similar soils mapped in one association"
    else:
        return np.nan