import pandas as pd
import numpy as np

import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
import re

import shap
import catboost
from sklearn import metrics
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.model_selection import train_test_split

import mypackage 

def select_cutoff(cutoff, visit = 1):
    match cutoff:
        case 5:
            thres = "cut_off_5"
        case 15:
            thres = "cut_off_15"
        case 30:
            thres = "cut_off_30"
    return f"{thres}_AHI{visit}"
        
def plot_AHI(df_y, sub_idx):
    plt.rcParams.update({'font.size': 18})
    fig,ax = plt.subplots(figsize=(10,5))
    df_temp = df_y.iloc[sub_idx]
    ahi_cols = [col for col in df_temp.index if col.startswith("AHI") and "_label" not in col]
    ax.axhspan(0, 5, facecolor='green', alpha=0.2, label='Normal (0-5)')
    ax.axhspan(5, 15, facecolor='yellow', alpha=0.2, label='Mild (5-15)')
    ax.axhspan(15, 30, facecolor='orange', alpha=0.2, label='Moderate (15-30)')
    ax.axhspan(30, 100, facecolor='red', alpha=0.2, label='Severe (30+)')
    ax.plot(df_temp[ahi_cols], "o--", label="AHI value")
    ax.set_title("AHI Classification Zones")
    ax.set_ylabel("AHI")
    ax.set_ylim(0, max(0,max(df_temp[ahi_cols]),45)+5)
    ax.legend(loc='upper left',bbox_to_anchor=(-0.1,1.8))
    ax.grid()
    st.pyplot(fig)

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
        return fig

def plot_grouped_pie_chart(df ,group_name = ["a", "b", "c"]):
    '''This function is used for SHHS and MESA harmonized data only
    a: first priority order
    b: second priority order, usually "Severity"
    c: could be any columns different from a and b, usually "nsrr_ahi_hp3r_aasm15"'''
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    total = df.groupby([group_name[0], group_name[1]])[group_name[2]].count().unstack(group_name[1]).fillna(0)
    total = total.transpose()
    # display(total)
    total = total.loc[['Normal','Mild','Moderate','Severe']]

    def func(pct, allvals):
        '''calculate the percentage of each component 
        and display in format {percentage}% (n = total_number)'''                                                     
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return f"{pct:.1f}%\n(n={absolute:d} )"

    plt.rcParams['font.size']   = 10.0
    fig, ax                     = plt.subplots(1,len(total.columns), figsize=(3*len(total.columns), 5), subplot_kw=dict(aspect="equal"), facecolor='w')
    for i in range(len(total.columns)):
        wedges, texts, autotexts = ax[i].pie(total[total.columns[i]], autopct=lambda pct: func(pct, total[total.columns[i]]),
                                          # pctdistance=1.25, labeldistance=.6,
                                          colors=[to_rgba(c, alpha=0.7) for c in base_colors[:len(total.index)]],
                                          startangle=90, counterclock=False,
                                          wedgeprops=dict(width=0.4),
                                        textprops=dict(color="k"))
        ax[i].set_title(total.columns[i])
      
    plt.figlegend(wedges, total.index,
            title = "OSA severity",
            loc = "center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    # plt.setp(autotexts, size=8, weight="bold")
    plt.suptitle(f"Distribution of OSA severity separate by {group_name[0]}")
    plt.tight_layout()
    # plt.show()

    return fig, ax

# def plot_grouped_bar_chart(df, feature = None):
#     if feature is None:
#         print("Error, no feature selected")
        
#     total = df.groupby([feature, "AHI_label1"])["AHI1"].count().unstack("AHI_label1").fillna(0)
#     fig   = total[['Normal','Mild','Moderate','Severe']].plot(kind='bar', stacked=True, figsize = [10,5])
#     plt.ylabel("Num of subject")
#     plt.title(f"Distribution of OSA severity across different {feature} in SHHS")
#     return fig

def plot_grouped_bar_chart(df, feature=None):
    if feature is None:
        raise ValueError("No feature selected")

    total = (
        df.groupby([feature, "AHI_label1"])["AHI1"]
          .count()
          .unstack("AHI_label1")
          .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    total[['Normal', 'Mild', 'Moderate', 'Severe']].plot(
        kind='bar',
        stacked=True,
        ax=ax
    )

    ax.set_ylabel("Number of subjects")
    ax.set_title(f"Distribution of OSA severity across {feature} (SHHS)")

    fig.tight_layout()
    return fig

class LoadDatasetPage:
    def __init__(self, dataset_name, df_x, df_y):
        self.dataset_name = dataset_name
        self.df_x = df_x
        self.df_y = df_y

    
              
    def display_dataset_page(self):
        ### ------------------------- Display datasets ------------------------------ ###
        st.markdown(f"<h4>Data exploratory.</h4>", unsafe_allow_html=True)
        # st.text_input("Subject ID", key="subject_id_input")
        # st.selectbox(
        #     "Select Subject ID to preview data",
        #     options=df_x['subject_id'].unique(),
        #     key="subject_id"
        # )
        tab_questionnaire, tab_individual = st.tabs(["Questionnaire data", "Individual data preview"])
        with tab_questionnaire:
        ### ------------------------- Feature explanation ------------------------------ ###
            # st.write("Feature explanation:")
            # st.selectbox(
            #     "Select feature to see explanation",
            #     options=self.df_x.columns,
            #     key="feature_explanation"
            # )
        ### ------------------------- Plot general information ------------------------------ ###
            if "plots" not in st.session_state:
                st.session_state.plots = []

            # st.subheader("📊 Plot Explorer")

            # identify categorical columns
            cat_cols        = mypackage.cat_identify(self.df_x)
            low_cat_cols    = [col for col in cat_cols if self.df_x[col].nunique()<10]
            df_cols         = self.df_x.columns[1:]

            

            if len(st.session_state.plots) == 0:
                st.write("No feature selected to display.")
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
                                    st.write(selected)
                                    st.session_state.plots[i] = selected
                                    if selected in low_cat_cols:
                                        plot_type = "categorical"
                                    elif selected in cat_cols:
                                        plot_type = "high_cardinality_categorical"
                                    else:
                                        plot_type = "numerical"

                                    df_total = pd.concat([self.df_x, self.df_y],axis = 1)
                                    if plot_type == "categorical":
                                        fig, ax = plot_grouped_pie_chart(df_total ,
                                                                        group_name = [selected, "AHI_label1", "AHI1"])
                                        fig.tight_layout()
                                        st.pyplot(fig)
                                    elif plot_type == "numerical":
                                        fig = plot_grouped_bar_chart(df_total, 
                                                                     selected)
                                        st.pyplot(fig)

                                    else:
                                        st.write("nothing to plot")
                                    # fig = plot_type_define(self, selected, plot_type)
                                    # st.pyplot(fig)

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

            



        with tab_individual:
            ###---------------------- plot AHI -------------------- ###
            # sub_idx = st.slider("Select an index", 0, len(self.df_y)-1, 0)
            sub_idx = st.selectbox(
                "Select Subject ID to preview data",
                    options=self.df_x.index,
                    index=0
                )
            df_temp = self.df_x.iloc[sub_idx]
            col1, col2 = st.columns([1, 1])

            with col1:
                st.write(df_temp.dropna())
            with col2:
                plot_AHI(self.df_y, sub_idx)

            