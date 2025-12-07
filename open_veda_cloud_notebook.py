# %%

import datetime as dt

import geopandas as gpd
from owslib.ogcapi.features import Features
import requests
import os

# %%

OGC_URL = "https://openveda.cloud/api/features"

w = Features(url=OGC_URL)
w.feature_collections()

# %%

collection_id = "public.eis_fire_snapshot_perimeter_nrt"
perm = w.collection(collection_id)
perm_q = w.collection_queryables(collection_id)
perm_q["properties"]

# %%

url_example = "https://openveda.cloud/api/features/collections/public.eis_fire_lf_fireline_nrt/items?f=geojson&sortby=-t" 
response = requests.get(url_example)
most_recent_time = response.json()['features'][0]['properties']['t'] # Extracting the most recent time from the json dictionary. 
most_recent_time

# %%

import math

def iter_features_offset(w, collection_id, params=None, page_size=100, max_pages=None, progress=True):
    """
    Paginate through OGC API Features using offset parameter.
    - Uses w.collection_items() with offset increments
    - Default page_size=100
    - Shows progress as pages are fetched
    """
    params = dict(params or {})
    
    # Get total count with minimal data
    meta_params = dict(params)
    meta_params["limit"] = 1
    meta = w.collection_items(collection_id, **meta_params)
    total = meta.get("numberMatched", 0)
    
    if total == 0:
        if progress:
            print("No matching features")
        return []
    # Round up the division here for total number of pages
    pages = math.ceil(total / page_size)

    # Support a user-defined page limit
    if max_pages and max_pages < pages:
        pages = max_pages

    all_features = []
    
    for i in range(pages):
        offset = i * page_size
        page_params = dict(params)
        page_params["limit"] = page_size
        page_params["offset"] = offset
        
        page = w.collection_items(collection_id, **page_params)
        features = page.get("features", [])
        all_features.extend(features)
        
        if progress:
            print(f"Page {i+1}/{pages}: {len(all_features)}/{total} features")
        
        if len(features) < page_size:
            break
    
    return all_features

# %%

## Get 7 days before most recent fire perimeter
most_recent_time = most_recent_time + "+00:00"
now = dt.datetime.strptime(most_recent_time, "%Y-%m-%dT%H:%M:%S+00:00")
last_week = now - dt.timedelta(weeks=1)
last_week = dt.datetime.strftime(last_week, "%Y-%m-%dT%H:%M:%S+00:00")
print("Most Recent Time =", most_recent_time)
print("Last week =", last_week)

# %%

# Using pagination instead of a single large request
params = {
    "bbox": ["-106.8", "24.5", "-72.9", "37.3"],
    "datetime": [last_week + "/" + most_recent_time],
    "filter": "farea>5 AND duration>2",
}

# Fetch features with pagination
features = iter_features_offset(
    w,
    "public.eis_fire_snapshot_perimeter_nrt",
    params=params,
    page_size=100,
    progress=True,
)

# Create results dictionary compatible with existing code
perm_results = {
    "type": "FeatureCollection",
    "features": features,
    "numberMatched": len(features),
    "numberReturned": len(features)
}

# %%

perm_results.keys()

# %%

perm_results["numberMatched"] == perm_results["numberReturned"]

# %%

perm_results["features"][0]["links"][1]["href"]

# %%

fline_q = w.collection_queryables("public.eis_fire_snapshot_fireline_nrt")
fline_collection = w.collection("public.eis_fire_snapshot_fireline_nrt")
fline_q["properties"]

# %%

# Get most recent fire perimeters
params = {
    "datetime": most_recent_time,  # or specify a date like "2025-01-15T00:00:00"
}

# Get perimeters with pagination
features = iter_features_offset(
    w,
    "public.eis_fire_snapshot_perimeter_nrt",
    params=params,
    page_size=100,
    max_pages=1,  # Increase if you want more data
    progress=True,
)

perm_results = {
    "type": "FeatureCollection",
    "features": list(features)
}

perimeters = gpd.GeoDataFrame.from_features(perm_results["features"])
perimeters = perimeters.set_crs("epsg:4326")

# Print info about the data
print(f"Found {len(perimeters)} perimeter features")
print(f"Fire IDs: {perimeters.fireid.unique()[:10]}")
print(f"Bounds: {perimeters.total_bounds}")

# Create interactive map
center_lat = -15
center_lon = -100

m = perimeters.explore(
    zoom_start=2,
    location=(center_lat, center_lon),
    color='red',
    style_kwds={
        'fillOpacity': 0.3,
        'weight': 2
    },
    tooltip=['fireid', 't', 'farea'],
    popup=True,
    legend_name="Fire Perimeters"
)

m.save("map.html")

# %%

# Get Camp Fire area perimeters with pagination
params = {
    "bbox": ["-124.52", "39.2", "-120", "42"],  # North California bounding box
    "datetime": ["2018-11-01T00:00:00+00:00/2018-11-30T12:00:00+00:00"],
}

features = iter_features_offset(
    w,
    "public.eis_fire_lf_perimeter_archive",
    params=params,
    page_size=100,
    progress=True,
)

perimeters_archive_results = {
    "type": "FeatureCollection",
    "features": features
}

perimeters = gpd.GeoDataFrame.from_features(perimeters_archive_results["features"])
perimeters = perimeters.sort_values(by="t", ascending=False)
perimeters = perimeters.set_crs("epsg:4326")

print(perimeters.fireid.unique())
m = perimeters.explore(
    style_kwds={"fillOpacity": 0}, zoom_start=9, location=(39.7, -121.4)
)

m.save("map.html")

# %%

# Get Camp Fire specific perimeters with pagination
params = {
    "filter": "fireid = 'F17028'",
    "datetime": ["2018-01-01T00:00:00+00:00/2018-12-31T12:00:00+00:00"],
}

features = iter_features_offset(
    w,
    "public.eis_fire_lf_perimeter_archive",
    params=params,
    page_size=100,
    progress=True,
)

perimeters_archive_results = {
    "type": "FeatureCollection",
    "features": features
}

perimeters = gpd.GeoDataFrame.from_features(perimeters_archive_results["features"])
perimeters = perimeters.sort_values(by="t", ascending=False)
perimeters = perimeters.set_crs("epsg:4326")

m = perimeters.explore(
    style_kwds={"fillOpacity": 0}, zoom_start=12, location=(39.7, -121.4)
)

m.save("map.html")

# %%

perimeters.to_file('perimeters.geojson', driver='GeoJSON')

# %%

params = {
    "datetime": ["2020-09-27T00:00:00+00:00/2020-10-13T12:00:00+00:00"],
}

features = iter_features_offset(
    w,
    "public.eis_fire_lf_perimeter_archive",
    params=params,
    page_size=100,
    progress=True,
)

perimeters_archive_results = {
    "type": "FeatureCollection",
    "features": features
}

perimeters = gpd.GeoDataFrame.from_features(perimeters_archive_results["features"])
perimeters = perimeters.sort_values(by="t", ascending=False)
perimeters = perimeters.set_crs("epsg:4326")

fires = set()

for i in range(len(perimeters)):
    if 100 < perimeters["farea"][i] and perimeters["farea"][i] < 200 and perimeters["fireid"][i] not in fires:
        fires.add(perimeters["fireid"][i])

for fire in fires:
    perimeters[perimeters["fireid"] == fire].to_file(f'fires/{fire}.geojson', driver='GeoJSON')

# %%

file_path = '2020-2021 fires/F10911.geojson'
gdf = gpd.read_file(file_path)

m = gdf.explore(
    style_kwds={"fillOpacity": 0}
)
m.save("map.html")

# %%

with open("temp.txt", "w") as f:
    for file in os.listdir('fires'):
        gdf = gpd.read_file(f'fires/{file}')
        start_time = gdf['t'].min()
        end_time = gdf['t'].max()
        area = gdf['farea'].max()
        bbox = gdf.total_bounds
        
        growth_frames = 0
        for i in range(len(gdf)):
            if gdf['n_newpixels'][i] > 0:
                growth_frames += 1
    
        if area > 200:
            continue
    
        print(file, start_time, end_time, area, growth_frames, bbox, file=f)

# %%

import pandas as pd
import os

fire_dates = []
for file in os.listdir('fires'):
    if file.endswith('.geojson'):
        gdf = gpd.read_file(os.path.join('fires', file))
        start_time = gdf['t'].min()
        fire_dates.append(pd.to_datetime(start_time))

df = pd.DataFrame(fire_dates, columns=['start_date'])
df['start_date'] = pd.to_datetime(df['start_date'])

df = df.set_index('start_date')

daily_counts = df.resample('D').size()

rolling_counts = daily_counts.rolling(window=30).sum()

max_fires_date = rolling_counts.idxmax()
max_fires_count = rolling_counts.max()

start_date = max_fires_date - pd.Timedelta(days=29)

print(f"The 30-day period with the most fires started on: {start_date.date()}")
print(f"Number of fires in this period: {int(max_fires_count)}")

# %%

