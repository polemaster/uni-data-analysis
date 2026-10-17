# Import necessary libraries
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
import folium


# Step 1: Load GeoJSON Data
geojson_path = "voivodships.geojson"  # Ensure this file exists in your working directory
try:
   polygons = gpd.read_file(geojson_path)
except FileNotFoundError:
   raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")


# Debug: Print available columns in the GeoJSON file
print("Columns in GeoJSON file:", polygons.columns)


# Step 2: Add Population Data
population_data = pd.DataFrame({
   'voivodeship': ["Dolnośląskie", "Kujawsko-Pomorskie", "Lubelskie", "Lubuskie", "Łódzkie", "Małopolskie", "Mazowieckie", "Opolskie", "Podkarpackie", "Podlaskie", "Pomorskie", "Śląskie", "Świętokrzyskie", "Warmińsko-Mazurskie", "Wielkopolskie", "Zachodniopomorskie"],
   'population': [2874496, 1990323, 2003475, 972140, 2354135, 3429685,
                  5510618, 933349, 2067139, 1135201, 2359956, 4305126,
                  1163001, 1353374, 3484177, 1626876],
   'num_cities': [93, 56, 57, 44, 60, 64, 111, 37, 54, 40, 43, 74, 51, 50, 120, 66]
})


# Align GeoJSON column names to match population data
# Debug: Check unique names in the GeoJSON file
print("Unique names in NAME_1 column:", polygons['NAME_1'].unique())


# Step 3: Map GeoJSON Names to Population Data Names
# Complete mapping between GeoJSON names and voivodeship names
name_mapping = { "Lower Silesian": "Dolnośląskie", "Kuyavian-Pomeranian": "Kujawsko-Pomorskie", "Lublin": "Lubelskie", "Lubusz": "Lubuskie", "Łódź": "Łódzkie", "Lesser Poland": "Małopolskie", "Masovian": "Mazowieckie", "Opole": "Opolskie", "Subcarpathian": "Podkarpackie", "Podlachian": "Podlaskie", "Pomeranian": "Pomorskie", "Silesian": "Śląskie", "Świętokrzyskie": "Świętokrzyskie", "Warmian-Masurian": "Warmińsko-Mazurskie", "Greater Poland": "Wielkopolskie", "West Pomeranian": "Zachodniopomorskie" }


# Replace GeoJSON names with mapped names
polygons['NAME_1'] = polygons['NAME_1'].replace(name_mapping)


# Merge GeoJSON data with population data
polygons = polygons.merge(population_data, left_on="NAME_1", right_on="voivodeship", how="inner")


# Debug: Check the merged DataFrame
print("Number of rows after merge:", len(polygons))
print(polygons[['NAME_1', 'voivodeship', 'population', 'num_cities']])


# Step 4: Handle Missing Values
# Check and fill missing values in 'population' if needed
if polygons['population'].isna().sum() > 0:
   print("Missing values found in 'population'. Filling with the column mean...")
   polygons['population'] = polygons['population'].fillna(polygons['population'].mean())


# Standardize the population data for clustering
polygons['population_scaled'] = (
   polygons['population'] - polygons['population'].mean()
) / polygons['population'].std()


# Handle any potential NaN values in 'population_scaled'
polygons['population_scaled'] = polygons['population_scaled'].fillna(0)


# Step 5: Apply K-Means Clustering
# Ensure there are enough rows for clustering
if len(polygons) < 3:
   raise ValueError("Not enough data points for K-Means clustering (minimum 3 required).")


kmeans = KMeans(n_clusters=3, random_state=42)
polygons['cluster'] = kmeans.fit_predict(polygons[['population_scaled']])


# Step 5b: Sort clusters by population and reassign labels
cluster_order = polygons.groupby('cluster')['population'].mean().sort_values().index
cluster_mapping = {old: new for new, old in enumerate(cluster_order)}


# Apply the sorted cluster mapping
polygons['cluster'] = polygons['cluster'].map(cluster_mapping)


# Debug: Check the clustering results
print(polygons[['voivodeship', 'population', 'cluster']])


# Step 6: Create an Interactive Choropleth Map
# Initialize the map
m = folium.Map(location=[52, 19], zoom_start=6)  # Centered on Poland


# Add a choropleth layer
folium.Choropleth(
   geo_data=polygons,  # GeoJSON data
   data=polygons,  # DataFrame with cluster info
   columns=['voivodeship', 'cluster'],  # Columns for the choropleth
   key_on='feature.properties.NAME_1',  # Match GeoJSON property
   fill_color='YlGn',
   fill_opacity=0.7,
   line_opacity=0.2,
   legend_name='Population Clusters'
).add_to(m)


# Add pop-ups for each voivodeship
for _, row in polygons.iterrows():
   # Extract the centroid of each geometry for placing the marker
   centroid = row.geometry.centroid
   folium.Marker(
       location=[centroid.y, centroid.x],
       popup=folium.Popup(
           f"<b>Voivodeship:</b> {row['voivodeship']}<br>"
           f"<b>Population:</b> {row['population']:,}<br>"
           f"<b>Number of Cities:</b> {row['num_cities']}<br>"
           f"<b>Cluster:</b> {row['cluster']}",
           max_width=300
       )
   ).add_to(m)


# Save the map
output_path = "poland_population_clusters_geojson.html"
m.save(output_path)
print(f"Interactive map saved as '{output_path}'")

