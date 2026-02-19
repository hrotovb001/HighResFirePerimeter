import geopandas as gpd
from owslib.ogcapi.features import Features
import argparse
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
        
        try:
            page = w.collection_items(collection_id, **page_params)
        except Exception as e:
            print(f"Error fetching page {i+1}: {e}")
            continue

        features = page.get("features", [])
        all_features.extend(features)
        
        if progress:
            print(f"Page {i+1}/{pages}: {len(all_features)}/{total} features")
        
        if len(features) < page_size:
            break
    
    return all_features


def main():
    parser = argparse.ArgumentParser(description="Gather fire perimeter data.")
    parser.add_argument("--ogc_url", default="https://openveda.cloud/api/features", help="URL of the OGC API")
    parser.add_argument("--collection", default="public.eis_fire_lf_perimeter_archive", help="Feature collection name")
    parser.add_argument("--datetime", default="2020-09-27T00:00:00+00:00/2020-10-13T12:00:00+00:00", help="Datetime range for filtering")
    parser.add_argument("--page_size", type=int, default=100, help="Page size for fetching features")
    parser.add_argument("--min_area", type=float, default=100, help="Minimum fire area")
    parser.add_argument("--max_area", type=float, default=200, help="Maximum fire area")
    parser.add_argument("--output_dir", default="fires", help="Output directory for GeoJSON files")
    args = parser.parse_args()

    w = Features(url=args.ogc_url)
    w.feature_collections()

    params = {
        "datetime": [args.datetime],
    }

    features = iter_features_offset(
        w,
        args.collection,
        params=params,
        page_size=args.page_size,
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
        if args.min_area < perimeters["farea"][i] < args.max_area and perimeters["fireid"][i] not in fires:
            fires.add(perimeters["fireid"][i])

    for fire in fires:
        perimeters[perimeters["fireid"] == fire].to_file(f'{args.output_dir}/{fire}.geojson', driver='GeoJSON')

if __name__ == "__main__":
    main()
