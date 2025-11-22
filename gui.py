import streamlit as st
import geemap.foliumap as geemap
import ee
from utils import get_aoi_from_nuts, get_species_data, get_layer_information, get_layer_visualization_params


st.set_page_config(layout="wide")

geemap.ee_initialize(auth_mode='gcloud')

st.title("GUI Toolbox")
st.write("This is a simple GUI toolbox using Streamlit.")

sdm_tab, extract_tab = st.tabs(["SDM", "Extract"])

with sdm_tab:
    st.header("Species Distribution Modeling (SDM)")

    with st.expander("SDM Settings"):
        year_col, species_col, country_col, county_col = st.columns(4, gap="medium")
        with year_col:
            st.number_input("Select Year", min_value=2015, max_value=2024, value=2022, step=1, key="year_input")
        
        with species_col:
            st.text_input("Species Name", value="Lagopus muta", key="species_input")
        
        with country_col:
            st.text_input("Country Code", value="AT", max_chars=2, key="country_input")
            if st.session_state.country_input:
                with county_col:
                    county = st.text_input("County Name (Optional)", value="Tirol", key="county_input")
                country_aoi, county_aoi = get_aoi_from_nuts(st.session_state.country_input, st.session_state.county_input)
        if st.session_state.species_input and st.session_state.country_input:
                st.session_state.species_gdf = get_species_data(st.session_state.species_input, st.session_state.country_input)
        st.multiselect(
            "Select Environmental Features",
            options=["elevation", "slope", "NDVI", "CORINE Land Cover 2018", "CHM", "NARI", "NCRI", "GHMI", "SWE", "snow_depth", "snow_cover", "snow_albedo", "BIOCLIM", "aspect", "northness", "eastness"],
            default=["elevation", "slope", "NDVI"],
            key="features_select"
        )
    
    with st.expander("Stats and Visualizations"):
        st.write("Statistical summaries and visualizations will be displayed here.")
    if "layer" not in st.session_state or st.session_state.layer is None:
        st.session_state.layer = get_layer_information(st.session_state.year_input)
    
    with st.expander("Map View", expanded=True):
        Map = geemap.Map()
        Map.add_basemap("SATELLITE")
        for key, value in st.session_state.layer.items():
            if key in st.session_state.features_select:
                Map.addLayer(value.clip(country_aoi), get_layer_visualization_params(key), key, opacity=.5) 
        Map.addLayer(country_aoi, {'color':'red'}, "Country AOI", opacity=0.1)
        if st.session_state.species_gdf is not None:
            Map.addLayer(geemap.gdf_to_ee(st.session_state.species_gdf), {'color':'red'}, "Species Observations")
        Map.centerObject(country_aoi, 6)
        Map.to_streamlit()

    
    
    
    if st.button("Run SDM"):
        st.write("SDM process started!")




# Add more GUI components as needed
if st.button("Click me"):
    st.write("Button clicked!")
        
        
        
        
        
if __name__ == "__main__":

    pass









