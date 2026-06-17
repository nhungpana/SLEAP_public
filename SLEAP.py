from json import load
import select
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn import metrics

import os
import sys

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent

# PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd() ))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0,PROJECT_ROOT)

print("PROJECT_ROOT:", PROJECT_ROOT)

### ------------------------- Load sidebar ------------------------------ ###
from Streamlit_interface.b_content.e_training_page import TrainModelPage
from mypackage.paths import DATA_DIR
from Streamlit_interface.a_layout.b_sidebar_layout import LoadSidebar


# st.session_state.setdefault("dataset", None)
st.session_state.setdefault("page", "test_yourself")

# if "dataset" not in st.session_state:
#     st.session_state.dataset = "SHHS"

if "page" not in st.session_state:
    st.session_state.page = "home"

# dataset_name = [
#             "Sleep Heart Health Study (SHHS)", 
#             "Wiscosin Sleep Cohort (WSC)"
#             ]

sidebar = LoadSidebar()
sidebar.sidebar_info_v4()

# print(st.session_state.dataset)

# dataset = st.session_state.dataset
page = st.session_state.page

# st.write("Dataset:", dataset)
# st.write("Page:", page)

### ------------------------- Load datasets ------------------------------ ###
from Streamlit_interface.a_layout.a_path_load import LoadData
data_loader = LoadData()
data = data_loader.data

if data is None:
    st.error("Dataset failed to load")
else:
    df_x, df_y = data

# st.dataframe(df_x.head())
# st.dataframe(df_y.head())

# df_x, df_y = data_loader.data

# ### ------------------------- Page content ------------------------------ ###
# from Streamlit_interface.b_content.a_home_page import LoadHomePage
# from Streamlit_interface.b_content.b_dataset_page import LoadDatasetPage
from Streamlit_interface.b_content.c_model_page import LoadModelPage
from Streamlit_interface.b_content.d_selftest_page import LoadSelfTestPage
from Streamlit_interface.b_content.e_training_page import TrainModelPage
# from Streamlit_interface.b_content.f_chat_bot import LoadChatbotPage
# # st.write("Unacceptable")

if page == "home":
    # st.markdown(
    # f"<h3>Dataset: {dataset}</h4>",
    # unsafe_allow_html=True
    # )
    home_page = LoadHomePage(df_x, df_y)
    home_page.display_home_page()
elif page == "datasets":
    st.markdown(
    f"<h3>Dataset: {dataset}</h4>",
    unsafe_allow_html=True
    )
    dataset_page = LoadDatasetPage(df_x, df_y)
    dataset_page.display_dataset_page()
elif page == "processing_pipeline":
    # st.markdown(
    # f"<h3>Dataset: {dataset}</h4>",
    # unsafe_allow_html=True
    # )
    model_page = LoadModelPage(df_x, df_y)
    model_page.display_model_page()
elif page == "test_yourself":
    # st.markdown(f"<h4>Test yourself page.</h4>", unsafe_allow_html=True)
    self_test_page = LoadSelfTestPage(df_x, df_y)
    self_test_page.display_selftest_page()
elif page == "training_model":
    # st.markdown(
    # f"<h3>Dataset: {dataset}</h4>",
    # unsafe_allow_html=True
    # )
    train_model_page = TrainModelPage(df_x, df_y)
    train_model_page.display_training_page()
# elif page == "chatbot":
#     # st.markdown(f"<h4>Chatbot page.</h4>", unsafe_allow_html=True)
#     chatbot_page = LoadChatbotPage()

