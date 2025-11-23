import streamlit as st
import json
from io import StringIO
import geemap.foliumap
import geemap
import pandas as pd
import geopandas as gpd
import ee
import tempfile
from utils import load_background_data, get_layer_information, get_aoi_from_nuts, get_species_data, get_layer_visualization_params, plot_correlation_heatmap, compute_sdm, plot_hier_clustering, initialize_gee, classify_image_aoi

@st.cache_data
def load_map_layer(country_code):
    Map = geemap.foliumap.Map()
    Map.add_basemap("SATELLITE")
    for key, value in st.session_state.layer.items():
        if key in st.session_state.layers_select:
            Map.addLayer(value.clip(country_aoi), get_layer_visualization_params(key), key, opacity=.5) 
            Map.addLayer(ee.Image().byte().paint(featureCollection=country_aoi, color=1, width=2), {'palette': 'FF0000'}, "Country AOI", opacity=1)
            Map.centerObject(country_aoi, 6)
        
    return Map

st.set_page_config(layout="wide",
                   page_title="SDM Playground",
                   page_icon=":bird:")

if "background_gdf" not in st.session_state:
    st.session_state.background_gdf = load_background_data()

initialize_gee()

st.title("SDM Plyaground")
st.write("This is a simple SDM visualizer using Streamlit.")

sdm_tab, stats_tab, extract_tab = st.tabs(["SDM", "Stats overview", "Extract"])

with sdm_tab:
    st.header("Species Distribution Modeling (SDM)")

    with st.expander("Layer Settings", expanded=True):
        with st.form("layer_form", border=False):
            year_col, species_col, country_col, county_col = st.columns(4, gap="medium")
            with year_col:
                st.number_input("Select Year", min_value=2015, max_value=2024, value=2024, step=1, key="year_input")
                if "layer" not in st.session_state or st.session_state.layer is None:
                    st.session_state.layer = get_layer_information(st.session_state.year_input)
            with species_col:
                st.text_input("Species Name", value="Lagopus muta", key="species_input")
            
            with country_col:
                st.text_input("Country Code", value="AT", max_chars=2, key="country_input")
            with county_col:
                county = st.text_input("County Name (Optional)", value="Tirol", key="county_input")
            country_aoi, county_aoi = get_aoi_from_nuts(st.session_state.country_input, st.session_state.county_input)
            if st.session_state.species_input and st.session_state.country_input:
                    st.session_state.species_gdf = get_species_data(st.session_state.species_input, st.session_state.country_input) 
            
            st.multiselect(
                    "Select Environmental Features",
                    options=list(st.session_state.layer.keys()),
                    default=['NDVI', 'aspect', 'b1', 'b12', 'CHM', 'GHMI', 'NARI', 'northness'],
                    key="layers_select"
                )
            if st.form_submit_button("Update Layers"):
                Map = load_map_layer(st.session_state.country_input)
        
        
    with st.expander("SDM Settings", expanded=False):
        with st.form("sdm_form", border=False):
            model_col, tree_col, tree_depth_col,  train_size_col = st.columns(4, gap="medium")
            with model_col:
                st.selectbox('Select Model', options=["Random Forest", "Maxent", "Gradient Boost"], index=0, key="model_input")
            if st.session_state.model_input in ["Random Forest", "Gradient Boost"]:
                with tree_col:
                    st.number_input('Number of Trees', min_value=10, max_value=500, value=100, step=10, key="n_trees_input")
                with tree_depth_col:
                    st.number_input('Max Tree Depth', min_value=1, max_value=50, value=3, step=1, key="tree_depth_input")
            with train_size_col:
                st.number_input('Train Size (%)', min_value=50, max_value=90, value=75, step=5, key="train_size_input")  
            
            st.multiselect(
                "Select Layer to display",
                options=list(st.session_state.layer.keys()),
                default=['NDVI', 'aspect', 'b1', 'b12', 'CHM', 'GHMI', 'NARI', 'northness'],
                key="features_select"
            )
            
            if st.form_submit_button("Run SDM"):
                with st.spinner("Running SDM..."):
                    st.session_state.rf, st.session_state.results_df, st.session_state.ml_gdf, st.session_state.predictors= compute_sdm(
                                                                                                                        species_gdf=st.session_state.species_gdf,
                                                                                                                        features=list(st.session_state.features_select),
                                                                                                                        prediction_aoi=county_aoi,
                                                                                                                        model_type=st.session_state.model_input, 
                                                                                                                        n_trees=st.session_state.n_trees_input, 
                                                                                                                        tree_depth=st.session_state.tree_depth_input, 
                                                                                                                        train_size=st.session_state.train_size_input/100, 
                                                                                                                        year=st.session_state.year_input
                                                                                                                        )
                                
                    st.success("SDM run completed. Results are displayed on the map below.")
                    st.write("Model Performance Metrics:")
                    st.table(st.session_state.results_df)
                    if (rocauc_mean := st.session_state.results_df['roc_auc'].mean()) > .7:
                        st.success("Mean ROC-AUC: {}".format(rocauc_mean))
                    else:
                        st.error("Mean ROC-AUC: {} < 0.7".format(rocauc_mean))
                
    
    
    with st.expander("Map View", expanded=True):
        if "country_input" in st.session_state:
            Map = load_map_layer(st.session_state.country_input)
            if "classified_img_pr" in st.session_state:
                Map.addLayer(st.session_state.classified_img_pr, {'min': 0, 'max': 1, 'palette': geemap.colormaps.palettes.viridis_r}, 'Habitat suitability')
                Map.add_colorbar({'min': 0, 'max': 1, 'palette': geemap.colormaps.palettes.viridis_r}, label="Habitat suitability",
                                orientation="vertical",
                                position="bottomright",
                                layer_name="Habitat suitability")
            if st.session_state.species_gdf is not None:
                Map.addLayer(geemap.gdf_to_ee(st.session_state.species_gdf), {'color':'red'}, f"Species Observations {st.session_state.species_input}", shown=True)
                Map.addLayer(geemap.gdf_to_ee(st.session_state.background_gdf), {'color':'blue'}, "Background data", shown=False)
            if st.button("Show SDM Prediction") and "rf" in st.session_state:
                # if "classified_img_pr" in st.session_state:
                #     del st.session_state.classified_img_pr
                with st.spinner("Classifying image..."):
                    st.session_state.classified_img_pr = classify_image_aoi(
                        image=st.session_state.predictors,
                        aoi=county_aoi,
                        ml_gdf=st.session_state.ml_gdf,
                        model=st.session_state.rf,
                        features=list(st.session_state.features_select)
                    )
                    st.success("Image classified.")
            else:
                st.error("Run the SDM first to see the prediction.")
            
            Map.to_streamlit()

    
with stats_tab:
    with st.expander("Stats and Visualizations"):
        st.write("Statistical summaries and visualizations will be displayed here.")
        if st.session_state.background_gdf is not None and st.session_state.features_select:
            pass
            # st.write(st.session_state.background_gdf.drop('geometry', axis=1).corr())
            fig, selected_features = plot_hier_clustering(st.session_state.background_gdf)
            st.pyplot(fig=fig)
            st.write(f"Selected features: {list(selected_features)}")   
        
        
        
if __name__ == "__main__":

    pass












