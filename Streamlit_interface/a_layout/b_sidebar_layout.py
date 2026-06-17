import streamlit as st

### -------------------------- Sidebar selection ------------------------- ###
class LoadSidebar():
    def __init__(self, dataset_name=None):
        self.dataset_name = dataset_name
        # print("hello")

    def sidebar_selection(self):
        if self.dataset_name is None:
            self.dataset_name = ["Empty"]
        # else:
        #     dataset_name = dataset_name
        return st.sidebar.selectbox(
            "Select a dataset", 
            self.dataset_name,
            index=0,
            label_visibility="visible",
            key="dataset"
            )
        # show_categories = st.sidebar.checkbox("Show categories data",
        #                       help="Check this box to display the dataset below.")
        # return dataset_chosen #, show_categories
    
    def sidebar_info_v4(self):
        # dataset_name = [
        #     "Sleep Heart Health Study (SHHS)", 
        #     "Wiscosin Sleep Cohort (WSC)"
        #     ]
        st.sidebar.title("Menu")
        # self.sidebar_selection()

        # --- Navigation buttons --- #
        # if st.sidebar.button("🏠 Home"):
        #     st.session_state.page = "home"
        # if st.sidebar.button("📊 Datasets"):
        #     st.session_state.page = "datasets"
        if st.sidebar.button("❓ Individual Prediction"):
            st.session_state.page = "test_yourself"
        # if st.sidebar.button("⚙️ Model Information"):
        #     st.session_state.page = "processing_pipeline"
        if st.sidebar.button("⚙️ Training Model"):
            st.session_state.page = "training_model"
        if st.sidebar.button("🤖 Chatbot"):
            st.session_state.page = "chatbot"
        

    # def sidebar_info_v3(self, dataset_name=None):
    #     st.sidebar.title("Menu")

    #     # --- Home button ---
    #     with st.sidebar.expander("🏠 Home", expanded=False):
    #         if st.button("Go to Home"):
    #             st.session_state.page = "home"

    #     # --- Dataset dropdown (expandable) ---
    #     with st.sidebar.expander("📊 Select Dataset", expanded=False):
    #         if st.button("Go to Datasets"):
    #             st.session_state.page = "datasets"
    #         dataset_name = [
    #             "Sleep Heart Health Study (SHHS)", 
    #             "Wiscosin Sleep Cohort (WSC)"
    #             ]
    #         self.sidebar_selection(dataset_name)

    #     # --- Settings button ---
    #     with st.sidebar.expander("⚙️ Settings", expanded=False):
    #         if st.button("Go to Settings"):
    #             st.session_state.page = "settings"
    #         st.selectbox(
    #             "Select settings category",
    #             ["General", "Advanced", "About"],
    #             label_visibility="collapsed",
    #             key="settings"
    #         )
                        