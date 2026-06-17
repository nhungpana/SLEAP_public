import pandas as pd
import streamlit as st
import mypackage
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mypackage.plotting import make_autopct

def plot_SA_severity(df, color_theme="reds",
                     list_AHI_severity=["Normal", "Mild", "Moderate", "Severe"],
                     list_AHI_labels=["AHI_label1", "AHI_label2", "AHI_label3", "AHI_label4"],
                     df_name="SHHS"):
    
    list_color_default  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    list_color_reds     = ["#F28E8C", "#E53935", "#C62828", "#7F1D1D"]
    list_color_blues    = ["#9ACAE6", "#4FA3D1", "#2C6FB7", "#1F3A5F"]
    list_color_greens   = ["#A8D5BA", "#5CBF7F", "#2E8B57", "#1B4D3E"]
    list_color_purples  = ["#CBA3D6", "#A349C7", "#7A2EA3", "#4B1F66"]
    list_color_greys    = ["#B0B0B0", "#7A7A7A", "#4D4D4D", "#2E2E2E"]

    match color_theme:
        case "default":
            list_color = list_color_default
        case "reds":
            list_color = list_color_reds
        case "blues":
            list_color = list_color_blues
        case "greens":
            list_color = list_color_greens
        case "purples":
            list_color = list_color_purples
        case "greys":
            list_color = list_color_greys
        case _:
            print(f"Color theme {color_theme} not recognized. Using default colors.")
            list_color = list_color_default
   

    # list_AHI_severity = ["Normal", "Mild", "Moderate", "Severe"]
    # list_AHI_labels = ["AHI_label1", "AHI_label2", "AHI_label3", "AHI_label4"]


    # print(f"Number of subjects diagnosed with sleep apnea: {df.shape[0]}")
    list_pie_colors = [to_rgba(c, alpha=0.6) for c in list_color[:len(list_AHI_severity)]]


    fig, ax = plt.subplots(1, len(list_AHI_labels), 
                           figsize=(len(list_AHI_labels)*5, 5), 
                           facecolor='white')
    
    for i, label in enumerate(list_AHI_labels):
        temp = (df[label].value_counts())
        if len(temp.index) != len(list_AHI_severity):
            for idx in list_AHI_severity:
                if idx not in temp.index:
                    temp.loc[idx] = 0
        temp = temp.loc[list_AHI_severity]
        radius_value = sum(temp)/df.shape[0]*1.8
        ax[i].pie(temp, 
                labels=None, 
                autopct=make_autopct(temp),
                colors=list_pie_colors,
                counterclock=False,
                radius=min(1, max(0.7, radius_value)),
                startangle=90)
        ax[i].set_title(f"{df_name}{i+1}, N={sum(temp)}")
    plt.legend(list_AHI_severity, loc='center left', bbox_to_anchor=(0.92, 0.5))
    plt.tight_layout()
    return fig, ax

def plot_type_define(self, selected, plot_type):
        alpha = 0.7
        base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(figsize=[10,5])
        match plot_type:
            case "categorical":
                counts = self.df_x[selected].value_counts()
                pie_colors = [to_rgba(c, alpha=alpha) for c in base_colors[:len(counts)]]
                ax.pie(
                    counts,
                    labels=None,
                    autopct=mypackage.make_autopct(counts),
                    counterclock=False,
                    startangle=90,
                    colors=pie_colors)
                ax.legend(
                        counts.index, 
                        loc="center left",
                        bbox_to_anchor=(-0.1, -0.1)
                        )
                ax.set_title(f"Poportion of {selected} \n(nominal categorical)")
            case "high_cardinality_categorical":
                ax.hist(self.df_x[selected].values, 
                            bins=self.df_x[selected].nunique(), 
                            color=base_colors[0], 
                            alpha=alpha )
                ax.set_title(f"Distribution of {selected} \n(ordinal categorical)")
            case "numerical":   
                ax.hist(self.df_x[selected].values, 
                            bins=100,
                            color=base_colors[0], 
                            alpha=alpha )
                ax.set_ylabel("Count")
                ax.set_xlabel(f"{selected} values")
                ax.set_title(f"Distribution of {selected}")
        return fig, ax



class LoadHomePage:
    def __init__(self, dataset_name, df_x, df_y):
        self.dataset_name = dataset_name
        self.df_x = df_x
        self.df_y = df_y
        
    

    def display_home_page(self):
        tab_info, tab_general, tab_note = st.tabs(["Dataset info", "General information", "Developer's note"])

        with tab_info:
            # st.write("**Wiscosin Sleep Cohort (WSC)** dataset selected.")
            # st.markdown(f"<h4>General information about the dataset.</h4>", unsafe_allow_html=True)
            if "SHHS" in self.dataset_name:
                st.write("""
                        The **Sleep Heart Health Study (SHHS)** is a large, multi-center cohort study 
                        led by the National Heart, Lung, and Blood Institute (NHLBI) to investigate 
                        the cardiovascular and systemic consequences of sleep-disordered breathing.

                        The study examines whether sleep-related breathing abnormalities are associated 
                        with an increased risk of coronary heart disease, stroke, hypertension, and 
                        all-cause mortality.

                        Between November 1995 and January 1998, **6,441 adults aged 40 years and older** 
                        were enrolled and underwent overnight polysomnography (SHHS Visit 1). A second 
                        sleep study (SHHS Visit 2) was conducted in **3,295 participants** during exam 
                        cycle 3 (January 2001-June 2003).

                        Cardiovascular outcomes were continuously monitored and adjudicated by the 
                        parent cohorts through 2011.
                            
                        _Data provided via the National Sleep Research Resource._
                        """)
            elif "WSC" in self.dataset_name:
                st.write("""
                        The Wisconsin Sleep Cohort (WSC) is an ongoing longitudinal study of the causes, 
                        consequences, and natural history of sleep disorders, particularly sleep apnea.

                        The WSC uses overnight in-laboratory sleep studies conducted at the University of 
                        Wisconsin-Madison ICTR's CTRC, with a baseline sample of 1,500 Wisconsin state employees, 
                        assessed at four-year intervals.

                        With approximately 100 publications, the WSC has produced seminal findings on:
                        - the high prevalence of sleep apnea
                        - longitudinal associations with cardiovascular, mental health, and mortality outcomes
                        - risk factors such as weight gain, menopause, and alcohol use

                        The WSC maintains international collaborations with Stanford, the University of Chicago, 
                        Harvard, Toronto, the UK, Poland, France, and others.

                        _Data provided via the National Sleep Research Resource._
                        """)
        
        with tab_general:
            # ---- Shared interactive control ----
            selected_display = st.radio(
                "Select display mode",
                ("Show all data", "Show part of data"),
                horizontal=True,
            )
            if selected_display == "Show all data":
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader("Features")
                    st.dataframe(self.df_x)

                with col2:
                    st.subheader("Label")
                    st.dataframe(self.df_y)

            elif selected_display == "Show part of data":
                selected_row = st.selectbox(
                    "Go to row",
                    options=self.df_x.index,
                    index=0
                )

                window = 5
                start = max(0, selected_row - window)
                end = selected_row + window + 1

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.subheader("Features")
                    st.dataframe(self.df_x.iloc[start:end])

                with col2:
                    st.subheader("Label")
                    st.dataframe(self.df_y.iloc[start:end])

        ### ------------------------- Feature explanation ------------------------------ ###
            st.write("Feature explanation:")
            st.selectbox(
                "Select feature to see explanation",
                options=self.df_x.columns[1:],
                key="feature_explanation"
            )
        ### ------------------------- Plot general information ------------------------------ ###
            if "plots" not in st.session_state:
                st.session_state.plots = []

            st.subheader("📊 Plot Explorer")

            # identify categorical columns
            cat_cols        = mypackage.cat_identify(self.df_x)
            low_cat_cols    = [col for col in cat_cols if self.df_x[col].nunique()<10]
            df_cols         = self.df_x.columns[1:]

            

            if len(st.session_state.plots) == 0:
                st.write("No plots to display.")
                # if len(st.session_state.plots) < 0:
                if st.button("➕ Add plot"):
                    st.session_state.plots.append(None)
                    st.rerun()
                

            else:
                if len(st.session_state.plots) > 0:
                    num_subplots = min(2, len(st.session_state.plots))
                    cols = st.columns(num_subplots)
                    for i, feature in enumerate(st.session_state.plots):
                        with cols[i % num_subplots]:
                            with st.container():
                                col1, col2 = st.columns([4, 1])

                                with col1:
                                    selected = st.selectbox(
                                        f"Select feature for plot {i+1}",
                                        df_cols,
                                        key=f"plot_select_{i}",
                                        index=df_cols.get_loc(feature) if feature in df_cols else 0,
                                    )

                                    st.session_state.plots[i] = selected
                                    if selected in low_cat_cols:
                                        plot_type = "categorical"
                                    elif selected in cat_cols:
                                        plot_type = "high_cardinality_categorical"
                                    else:
                                        plot_type = "numerical"
                                    
                                    fig, ax = plot_type_define(self, selected, plot_type)
                                    st.pyplot(fig)

                                with col2:
                                    if st.button("➕", key=f"add_{i}"):
                                            st.session_state.plots.append(None)
                                            st.rerun()
                                    if st.button("❌", key=f"remove_{i}"):
                                            st.session_state.plots.pop(i)
                                            st.rerun()
                                    

                    if st.button("🗑️ Clear all plots"):
                        st.session_state.plots = []
                        st.rerun()
        with tab_note:
            if "SHHS" in self.dataset_name:
                list_AHI_labels=["AHI_label1", "AHI_label2"]
                df_name="SHHS"
            elif "WSC" in self.dataset_name:
                list_AHI_labels=["AHI_label1", "AHI_label2", "AHI_label3", "AHI_label4"]
                df_name="WSC"

            df_total = pd.concat([self.df_x, self.df_y], axis=1)
            # st.dataframe(df_total)
            df_SA_diagnosed = df_total[df_total["apnea_common"] == "Y"]
            # st.dataframe(df_SA_diagnosed)
            fig, ax = plot_SA_severity(df_SA_diagnosed, 
                                       color_theme = "reds", 
                                       list_AHI_labels=list_AHI_labels,
                                       df_name=df_name)
            fig.suptitle("AHI severity distribution\n(Subjects have been diagnosed with sleep apnea)")
            plt.tight_layout()
            st.pyplot(fig)

            df_SA_diagnosed = df_total[df_total["apnea_common"] == "N"]
            # st.dataframe(df_SA_diagnosed)
            fig, ax = plot_SA_severity(df_SA_diagnosed, 
                                       color_theme = "greens", 
                                       list_AHI_labels=list_AHI_labels,
                                       df_name=df_name)
            fig.suptitle("AHI severity distribution\n(Subjects have been diagnosed with sleep apnea)")
            plt.tight_layout()
            st.pyplot(fig)