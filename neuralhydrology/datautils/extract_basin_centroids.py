"""Extract basin centroids from shapefile geometries."""
import os
from pathlib import Path
from typing import Optional, List

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def load_basin_boundaries(dir_path: str) -> Optional[gpd.GeoDataFrame]:
    """Load basin boundary geometries from all shapefiles in a directory.
    
    Parameters
    ----------
    dir_path : str
        Path to the directory containing shapefiles with basin boundaries
        
    Returns
    -------
    Optional[gpd.GeoDataFrame]
        GeoDataFrame containing basin geometries from all shapefiles, or None if error
    """
    try:
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} not found.")
        
        shapefile_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                          if f.lower().endswith('.shp')]
        
        if not shapefile_paths:
            raise FileNotFoundError(f"No shapefiles found in {dir_path}")
        
        all_basins = []
        
        for file_path in shapefile_paths:
            basins = gpd.read_file(file_path)
            
            if basins.empty:
                continue
                
            # Look for common basin name field variations
            name_fields = ['NAME', 'Name', 'name', 'BASIN_NAME', 'BasinName', 'ID', 'basin_id']
            basin_name_field = None
            
            for field in name_fields:
                if field in basins.columns:
                    basin_name_field = field
                    break
            
            if basin_name_field is None:
                file_prefix = os.path.splitext(os.path.basename(file_path))[0]
                basins['basin_name'] = [f"{file_prefix}_Basin_{i}" for i in range(len(basins))]
            else:
                basins = basins.rename(columns={basin_name_field: 'basin_name'})
                file_prefix = os.path.splitext(os.path.basename(file_path))[0]
                basins['basin_name'] = file_prefix + '_' + basins['basin_name'].astype(str)
            
            all_basins.append(basins)
        
        if not all_basins:
            return None
            
        combined_basins = gpd.GeoDataFrame(pd.concat(all_basins, ignore_index=True))
        
        if combined_basins['basin_name'].duplicated().any():
            mask = combined_basins['basin_name'].duplicated(keep=False)
            dup_indices = [str(i) for i in range(sum(mask))]
            combined_basins.loc[mask, 'basin_name'] = combined_basins.loc[mask, 'basin_name'] + '_' + dup_indices

        return combined_basins
    
    except Exception as e:
        raise RuntimeError(f"Error loading basin boundaries: {e}")


def calculate_basin_centroids(basins: Optional[gpd.GeoDataFrame]) -> Optional[pd.DataFrame]:
    """Calculate centroids for all basin geometries.
    
    Parameters
    ----------
    basins : Optional[gpd.GeoDataFrame]
        GeoDataFrame containing basin geometries
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame containing basin names and centroid coordinates
    """
    if basins is None:
        return None
    
    centroids = pd.DataFrame({
        'basin_name': basins['basin_name'],
        'latitude': [geom.centroid.y for geom in basins.geometry],
        'longitude': [geom.centroid.x for geom in basins.geometry]
    })
    
    return centroids


def plot_basins_and_centroids(basins: Optional[gpd.GeoDataFrame], centroids: Optional[pd.DataFrame]) -> None:
    """Plot basin boundaries and their centroids.
    
    Parameters
    ----------
    basins : Optional[gpd.GeoDataFrame]
        GeoDataFrame containing basin geometries
    centroids : Optional[pd.DataFrame]
        DataFrame containing basin centroid coordinates
    """
    if centroids is None:
        return
        
    print(centroids.head())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if basins is not None:
        basins.plot(ax=ax, color='lightblue', edgecolor='gray', alpha=0.5)
    
    ax.scatter(centroids['longitude'], centroids['latitude'], 
               color='red', marker='o', s=20, label='Centroids')
    
    for idx, row in centroids.sample(min(5, len(centroids))).iterrows():
        ax.annotate(row['basin_name'], 
                    xy=(row['longitude'], row['latitude']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title("Basin Boundaries and Centroids")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def save_basin_centroids(centroids: Optional[pd.DataFrame], output_file: str) -> None:
    """Save basin centroids to a CSV file.
    
    Parameters
    ----------
    centroids : Optional[pd.DataFrame]
        DataFrame with basin centroid data
    output_file : str
        Path to the output CSV file
    """
    if centroids is None:
        raise ValueError("No centroid data to save.")
    
    try:
        centroids.to_csv(output_file, index=False)
    except Exception as e:
        raise RuntimeError(f"Error saving centroids to CSV: {e}")


def main():
    """Main function to run the basin centroid extraction process."""
    basin_dir_path = "../data/basin_shapefiles"
    output_dir = "../data"
    output_file = os.path.join(output_dir, "basin_centroids.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    basins = load_basin_boundaries(basin_dir_path)
    
    if basins is not None:
        centroids = calculate_basin_centroids(basins)
        plot_basins_and_centroids(basins, centroids)
        save_basin_centroids(centroids, output_file)


if __name__ == "__main__":
    main()
