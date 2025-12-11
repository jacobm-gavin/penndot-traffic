import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import h3


def export_h3_to_shapefile(input_csv, output_shapefile):
    # 1. Load your modeling data
    df = pd.read_csv(input_csv)

    # 2. Define a function to convert H3 ID to Polygon
    def cell_to_poly(cell_id):
        # Get boundary coordinates (lat, lon)
        boundary = h3.cell_to_boundary(cell_id)
        # Flip to (lon, lat) for GIS standards and create Polygon
        return Polygon([(lon, lat) for lat, lon in boundary])

    # 3. Create Geometry Column
    # Ensure H3 IDs are valid before converting
    df['geometry'] = df['H3_R8'].apply(cell_to_poly)

    # 4. Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    # 5. Set Coordinate Reference System (EPSG:4326 is standard Lat/Lon)
    gdf.set_crs(epsg=4326, inplace=True)

    # 6. Export to Shapefile (for ArcGIS Pro)
    # Note: Shapefiles have column name length limits; GeoJSON is safer if you have long names
    gdf.to_file(output_shapefile)
    print(f"Successfully exported {len(gdf)} hexagons to {output_shapefile}")

# Usage
export_h3_to_shapefile("data/aggregates/h3_modeling_dataset_enriched.csv", "data/gis/h3_crash_risk.shp")