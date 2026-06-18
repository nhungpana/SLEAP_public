import re
# from click import group
import matplotlib
from matplotlib.dates import SA
from numpy import sort
import os
# from pathlib import Path

import random
from pygments import highlight
import streamlit as st
import catboost
import shap
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from mypackage.paths import IMAGES_DIR, MODEL_DIR

MODEL_FILENAME = "CatBoost_model_15_seed42.cbm"
SEVERITY_ORDER = ["Normal", "Mild", "Moderate", "Severe"]


@st.cache_resource(show_spinner=False)
def load_prediction_model():
    model = catboost.CatBoostClassifier()
    model.load_model(os.path.join(MODEL_DIR, MODEL_FILENAME))
    return model


@st.cache_resource(show_spinner=False)
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)


def st_shap(plot, height=150):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)


def align_input_columns(input_df, expected_columns):
    expected_columns = list(expected_columns)
    if len(input_df.columns) != len(expected_columns):
        raise ValueError(
            "The user input does not match the model feature count. "
            f"Input has {len(input_df.columns)} columns, model expects {len(expected_columns)}."
        )

    aligned_df = input_df.copy()
    aligned_df.columns = expected_columns
    return aligned_df



### --------------------------------- PDF report generation ------------------------- ###
import io
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import pagesizes
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import PageBreak
from reportlab.lib import utils

def generate_pdf(input_df, prediction_text, force_fig):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=pagesizes.A4)
    elements = []
    styles = getSampleStyleSheet()

    # ---- Title ----
    elements.append(Paragraph("<b>OSA Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    # ---- Prediction ----
    elements.append(Paragraph(prediction_text, styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # ---- Input Data ----
    elements.append(Paragraph("<b>User Input</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for col in input_df.columns:
        value = input_df.iloc[0][col]
        elements.append(
            Paragraph(f"<b>{col}</b>: {value}", styles["Normal"])
        )
        elements.append(Spacer(1, 0.1 * inch))


    # ---- Force Plot Image ----
    force_fig.set_size_inches(8, 2.5)
    img_buffer = io.BytesIO()
    force_fig.savefig(img_buffer, format="png", dpi=200, bbox_inches="tight")
    img_buffer.seek(0)

    image = ImageReader(img_buffer)
    img_width, img_height = image.getSize()

    max_width = 6.5 * inch
    max_height = 4 * inch   # smaller height limit for force plot

    scale = min(max_width / img_width, max_height / img_height)

    img = Image(
        img_buffer,
        width=img_width * scale,
        height=img_height * scale
    )

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("<b>Feature Contribution</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(img)

    doc.build(elements)
    buffer.seek(0)

    return buffer


def generate_pdf_report(user_df, figures, markdown_text):
    """
    user_df: pandas dataframe (user input)
    figures: list of matplotlib figures
    markdown_text: string (your st.markdown content)
    """

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=pagesizes.A4
    )

    elements = []
    styles = getSampleStyleSheet()

    # ---- Title ----
    elements.append(Paragraph("<b>OSA Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    # ---- Markdown Text ----
    elements.append(Paragraph(markdown_text, styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # ---- User DataFrame ----
    elements.append(Paragraph("<b>User Input Data</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    table_data = [user_df.columns.tolist()] + user_df.values.tolist()

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.4 * inch))

    # ---- Figures ----
    elements.append(Paragraph("<b>Model Outputs</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for fig in figures:
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300, bbox_inches="tight")
        img_buffer.seek(0)

        img = Image(img_buffer, width=5.5 * inch, preserveAspectRatio=True)
        elements.append(img)
        elements.append(Spacer(1, 0.4 * inch))

    doc.build(elements)

    buffer.seek(0)
    return buffer

def create_force_plot(explainer, shap_values, input_df):
    fig = plt.figure()

    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values[0].values,
        features=input_df.iloc[0],
        feature_names=input_df.columns,
        matplotlib=True,
        show=False
    )

    return fig

def slider_generate(text, min_val, max_val, col_width = 1, ):
    cols = st.columns([1,col_width])
    with cols[0]:
        st.write(text)
    with cols[1]:
        value = st.slider(
            text,
            min_val, 
            max_val, 
            min_val,
            label_visibility="collapsed"
        )
    return value

def grouped_bar_chart(df, group_name=["feature","AHI_label1","AHI1"], user_age=None):
    if group_name[0] is None:
            raise ValueError("No feature selected")

    df_total = (
        df.groupby([group_name[0], group_name[1]])[group_name[2]]
        .count()
        .unstack(group_name[1])
        .fillna(0)
    )
    df_total = df_total.reindex(columns=SEVERITY_ORDER, fill_value=0)
    return df_total
    
def plot_grouped_bar_chart(df_total, group_name=["feature","AHI_label1","AHI1"], user_age=None):
    
    fig, ax = plt.subplots(figsize=(10, 5))
    df_total[SEVERITY_ORDER].plot(
        kind='bar',
        stacked=True,
        ax=ax
    )
    if user_age in df_total.index:
        age_index = list(df_total.index).index(user_age)

        ### ----------------- hightlight user age bar ----------------- ###
        bottom = 0
        for sev in SEVERITY_ORDER:
            height = df_total.loc[user_age, sev]
            ax.bar(
                age_index,
                height,
                bottom=bottom,
                edgecolor='black',
                linewidth=2,
                fill=False,        # outline only
                width=0.8
            )
            bottom += height

        # total height of stacked bar
        total_height = df_total.loc[user_age].sum()

        ### ----------------- annotate user age bar with an arrow ----------------- ###
        ax.annotate(
            "You",
            xy=(age_index, total_height),          # arrow tip (top of bar)
            xytext=(age_index, total_height + 20), # text position above
            ha='center',
            arrowprops=dict(
                facecolor='blue',
                arrowstyle='->',
                linewidth=2
            ),
            fontsize=12,
            fontweight='bold'
        )

    ax.set_ylabel("Number of subjects")
    ax.set_title(f"Distribution of OSA severity across {group_name[0]} (SHHS)")

    fig.tight_layout()
    return fig, ax

def plot_grouped_pie_chart(df ,
                           group_name = ["a", "b", "c"], 
                           highlight_group=None, 
                           radius=1,
                           ax_title=None):
    '''This function is used for SHHS and MESA harmonized data only
    a: first priority order
    b: second priority order, usually "Severity"
    c: could be any columns different from a and b, usually "nsrr_ahi_hp3r_aasm15"'''
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    total = df.groupby([group_name[0], group_name[1]])[group_name[2]].count().unstack(group_name[1]).fillna(0)
    total = total.transpose()
    # display(total)
    total = total.reindex(SEVERITY_ORDER, fill_value=0)
    # st.dataframe(total)

    def func(pct, allvals):
        '''calculate the percentage of each component 
        and display in format {percentage}% (n = total_number)'''                                                     
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        return f"{pct:.1f}%\n(n={absolute:d} )"

    plt.rcParams['font.size']   = 10.0
    fig, ax                     = plt.subplots(1,len(total.columns), figsize=(3*len(total.columns), 5), subplot_kw=dict(aspect="equal"), facecolor='w')
    if len(total.columns) == 1:
        wedges, texts, autotexts = ax.pie(total[total.columns[0]], autopct=lambda pct: func(pct, total[total.columns[0]]),
                                          # pctdistance=1.25, labeldistance=.6,
                                          colors=[to_rgba(c, alpha=0.7) for c in base_colors[:len(total.index)]],
                                          startangle=90, counterclock=False,
                                          wedgeprops=dict(width=0.6),
                                        textprops=dict(color="k"))
        # ax.set_title(total.columns[0])
    else:
        for i, col in enumerate(total.columns):
            if highlight_group is not None:
                is_highlight = (col == highlight_group)
                # radius = 1.5 if is_highlight else 1
                wedges, texts, autotexts = ax[i].pie(total[total.columns[i]], autopct=lambda pct: func(pct, total[total.columns[i]]),
                                                # pctdistance=1.25, labeldistance=.6,
                                                colors=[to_rgba(c, alpha=0.7) for c in base_colors[:len(total.index)]],
                                                startangle=90, counterclock=False,
                                                wedgeprops=dict(width=0.6,
                                                            edgecolor='black' if is_highlight else 'white',
                                                            linewidth=2.5 if is_highlight else 1),
                                                textprops=dict(color="k"),
                                                radius=radius)
                if ax_title is None:
                    ax[i].set_title(total.columns[i],
                                    fontweight='bold' if is_highlight else 'normal',
                                    color='black')
                else:
                    ax[i].set_title(ax_title[i])
            else:    
                wedges, texts, autotexts = ax[i].pie(total[total.columns[i]], autopct=lambda pct: func(pct, total[total.columns[i]]),
                                                # pctdistance=1.25, labeldistance=.6,
                                                colors=[to_rgba(c, alpha=0.7) for c in base_colors[:len(total.index)]],
                                                startangle=90, counterclock=False,
                                                wedgeprops=dict(width=0.6),
                                                textprops=dict(color="k"),
                                                radius=radius)
            
                ax[i].set_title(ax_title[i] if ax_title is not None else total.columns[i])
      
    plt.figlegend(wedges, total.index,
            title = "OSA severity",
            loc = "center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    # plt.setp(autotexts, size=8, weight="bold")
    plt.suptitle(f"Distribution of OSA severity separate by {group_name[0]}")
    plt.tight_layout()
    # plt.show()

    return fig

class LoadSelfTestPage:
    def __init__(self, df_x, df_y):
        # self.dataset_name = dataset_name
        self.df_x = df_x
        self.df_y = df_y

    def display_selftest_page(self): 
        if "user_input" not in st.session_state:
            st.session_state.user_input = {}

        tabs = st.tabs(["Input", "Output"])
        with tabs[0]:
            st.subheader("Input data")

            with st.expander("Demographics"):
                st.write("""Basic population characteristics such as age, sex, race, education background.   
                        (*inputs are optional)""")
                cols = st.columns(4)
                with cols[0]:
                    sex = st.session_state.user_input["sex"] = st.selectbox(
                                "Sex:", 
                                self.df_x["sex"].unique(),
                                # index=0
                                index=None
                            )
                with cols[1]:
                    age = st.session_state.user_input["age"] = st.number_input(
                                "Age", min_value=18, max_value=150, value=55
                                # value=None
                            )
                with cols[2]:
                    education = st.session_state.user_input["education"] = st.selectbox(
                                "Education level*:",
                                ["Unknown", 
                                "Less than 10 years of education",
                                "11-15 years of education",
                                "16-20 years of education",
                                "more than 20 years of education"],
                                index = 0
                            )
                with cols[3]:
                    race = st.session_state.user_input["race"] = st.selectbox(
                                "Race*:",
                                self.df_x["race"].unique(),
                                index=len(self.df_x["race"].unique())-1
                            )
            
            with st.expander("Anthropometric"):
                st.write("""Physical body measurements such as height, weight, BMI, 
                        neck or waist circumference, used to describe body size and shape.""")
                
                col1, col2 = st.columns(2)
                with col1:
                    # st.write("Be careful with measurement unit!")
                    st.warning("Be careful with measurement unit!")
                    # st.info("Be careful with measurement unit!")
                    # st.error("Be careful with measurement unit!")
                    col_height, col_weight = st.columns(2)
                    with col_height:
                        height = st.session_state.user_input["height"] = st.number_input(
                                        "Height ***(cm)***", 90, 220, 170 #None
                                    )
                    with col_weight:
                        weight = st.session_state.user_input["weight"] = st.number_input(
                                        "Weight ***(kg)***", 18, 200, 70 #None
                                    )
                    if all(v is not None for v in [height, weight]):
                        bmi = st.session_state.user_input["bmi"] = np.round(st.session_state.user_input["weight"] / (
                                        (st.session_state.user_input["height"] / 100) ** 2
                                    ),2)
                        # st.write(f"Calculated BMI: {np.round(bmi,2)}")

                        if bmi < 18.5:
                            bmi_range = "Underweight"
                            st.warning(f"Calculated BMI: {np.round(bmi,2)}     \nBMI range: {bmi_range}")
                        elif bmi < 25:
                            bmi_range = "Healthy weight"
                            st.info(f"Calculated BMI: {np.round(bmi,2)}     \nBMI range: {bmi_range}")
                        elif bmi < 30:
                            bmi_range = "Overweight"
                            st.warning(f"Calculated BMI: {np.round(bmi,2)}     \nBMI range: {bmi_range}")
                        elif bmi < 35:
                            bmi_range = "Obesity I"
                            st.error(f"Calculated BMI: {np.round(bmi,2)}     \nBMI range: {bmi_range}")
                        elif bmi < 40:
                            bmi_range = "Obesity II"
                            st.error(f"Calculated BMI: {np.round(bmi,2)}     \nBMI range: {bmi_range}")
                        else:
                            bmi_range = "Obesity III"
                            st.error(f"Calculated BMI: {np.round(bmi,2)}     \nBMI range: {bmi_range}")

                    

                    neck = st.session_state.user_input["neck"] = st.number_input(
                                    "Neck circumference ***(cm)***", 15, 60, 20 #None
                                )


                with col2:
                    _, center, _ = st.columns([1, 2, 1])
                    with center:
                        st.image(
                            os.path.join(IMAGES_DIR, "Neck1.png"),
                            caption="Neck circumference in ***cm***",
                            width=160
                        )
                        st.image(
                            os.path.join(IMAGES_DIR, "Neck2.png"),
                            caption="Neck circumference in ***inch***",
                            width=170
                        )


            with st.expander("Comorbidities"):
                st.write("""Co-existing medical conditions that occur alongside a primary condition 
                         and may influence health risk or outcomes.""")
                

                st.info("Have you ever been diagnose with any of the following conditions? (select ALL that applied)")
                cols = st.columns([1,4])
                with cols[0]:
                    emphys = st.checkbox("Emphysema", value=False, key="emphys")
                with cols[1]:
                    with st.expander("What is Emphysema?"):
                        st.markdown("""
                                    <p style="margin:0;">
                                    <strong>Emphysema</strong> is a chronic lung disease where the air sacs in the lungs (alveoli) are gradually damaged and destroyed.
                                    Normally, these air sacs are elastic and help oxygen move into your blood.
                                    </p>
                                    """,
                                    unsafe_allow_html=True)
                        
                        cols =st.columns(2)
                        with cols[0]:
                            st.markdown(
                                """
                                <p style="margin:0;"><strong>In emphysema:</strong></p>
                                <ul style="margin-top:0;">
                                <li>The air sacs lose elasticity</li>
                                <li>Air gets trapped in the lungs</li>
                                <li>It becomes harder to breathe out</li>
                                </ul>
                                """,
                                unsafe_allow_html=True
                            )
                        with cols[1]:
                            st.image(
                            os.path.join(IMAGES_DIR, "emphysema.jpg"),                            
                        )
                            st.markdown(
                                '<a href="https://my.clevelandclinic.org/health/diseases/9370-emphysema" '
                                'target="_blank">Learn more about emphysema</a>',
                                unsafe_allow_html=True
                            )
                            

                cols = st.columns([1,4])
                with cols[0]:            
                    asthma = st.checkbox("Asthma", key="asthma")
                with cols[1]:
                    with st.expander("What is Asthma?"):
                        st.markdown("""
                                    <p style="margin:0;">
                                    <strong>Asthma</strong> is a chronic condition of the airways. 
                                    In asthma, the airways become inflamed and swollen, narrow more easily than normal,
                                    and may produce extra mucus
                                    </p>
                                    """,
                                    unsafe_allow_html=True)
                        
                        cols =st.columns(2)
                        with cols[0]:
                            st.markdown(
                                """
                                <p style="margin:0;"><strong>Common symptoms:</strong></p>
                                <ul style="margin-top:0;">
                                <li>Wheezing (a whistling sound when breathing)</li>
                                <li>Shortness of breath</li>
                                <li>Chest tightness</li>
                                <li>Coughing, especially at night or early morning</li>
                                </ul>
                                """,
                                unsafe_allow_html=True
                            )
                        with cols[1]:
                            st.image(
                            os.path.join(IMAGES_DIR, "asthma.jpg"),                            
                            )
                            st.markdown(
                                '<a href="https://aelhi.pitt.edu/what-is-asthma/" '
                                'target="_blank">Learn more about asthma</a>',
                                unsafe_allow_html=True
                            )
                            
                cols = st.columns([1,4])
                with cols[0]:  
                    htnderv = st.checkbox("Hypertension", key="htnderv")
                with cols[1]:
                    with st.expander("What is Hypertension?"):
                        cols =st.columns(2)
                        with cols[0]:
                            st.markdown("""
                                        <p style="margin:0;">
                                        <strong>Hypertension “The silent killer.”</strong>, also called high blood pressure, 
                                        is a condition where the force of blood pushing against the artery walls is consistently too high.
                                        Over time, this extra pressure makes the heart work harder 
                                        and can damage blood vessels and organs such as the heart, brain, kidneys, and eyes.
                                        </p>
                                        """,
                                        unsafe_allow_html=True)
                        

                        with cols[1]:
                            st.image(
                            os.path.join(IMAGES_DIR, "hypertension.jpg"),                            
                        )
                            st.markdown(
                                '<a href="https://my.clevelandclinic.org/health/diseases/4314-hypertension-high-blood-pressure" '
                                'target="_blank">Learn more about hypertension</a>',
                                unsafe_allow_html=True
                            )

                selected = [k for k, v in {
                    "Emphysema": emphys,
                    "Asthma": asthma,
                    "Hypertention": htnderv
                }.items() if v]
                # st.write(selected)

            with st.expander("Sleep Habits"):
                st.write("""These questions related to weekly sleep schedule.""")
                ### ------------------------- sleep duration ----------------------------- ###
                cols = st.columns(3)
                with cols[0]:
                    st.write("How many hours of sleep do you get each night?")
                with cols[1]:
                    hrswd = st.session_state.user_input["hrswd"] = st.number_input(
                        "On **weekdays**",
                        0, 24, 7
                    )
                with cols[2]:
                    hrswe = st.session_state.user_input["hrswe"] = st.number_input(
                        "On **weekends**",
                        0, 24, 8
                    )

                cols = st.columns(3)
                with cols[0]:
                    st.write("On average, how many minutes?")
                with cols[1]:
                    mi2slp = st.session_state.user_input["mi2slp"] = st.number_input(
                        "To fall asleep each night?",
                        0, 8*60, 15
                    )
                with cols[2]:
                    st.session_state.user_input["totminnap"] = st.number_input(
                        "Of nap during the day?",
                        0, 8*60, 0
                    )

                totminnap = st.session_state.user_input["totminnap"]/60

                totsleep = (
                    (st.session_state.user_input["hrswd"] * 5 +
                    st.session_state.user_input["hrswe"] * 2) / 7
                )

                totsleepnap = (
                    (st.session_state.user_input["hrswd"] * 5 +
                    st.session_state.user_input["hrswe"] * 2) / 7 +
                    (st.session_state.user_input["totminnap"] / 60)
                )     

                # ### ---------------------- Ver2 --------------------- ###
                # col_width = 1
                # cols = st.columns([1,col_width])
                # with cols[1]:
                #     st.markdown(
                #         "<div style='display:flex; justify-content:space-between; font-size:14px;'>"
                #         "<span>Never</span>"
                #         "<span>Rarely</span>"
                #         "<span>Sometimes</span>"
                #         "<span>Often</span>"
                #         "<span>Always</span>"
                #         "</div>",
                #         unsafe_allow_html=True
                #     )
                
                # cols = st.columns([1,col_width])
                # with cols[0]:
                #     st.write("Do you have trouble falling asleep?")
                # with cols[1]:
                #     tfa = st.session_state.user_input["tfa"] = st.slider(
                #         "",
                #         1, 5, 3,
                #         label_visibility="collapsed"
                #     )
                
                # cols = st.columns([1,col_width])
                # with cols[0]:
                #     st.write("Do you frequently wake up during the night and have trouble getting back to sleep?")
                # with cols[1]:
                #     wudnrs = st.session_state.user_input["wudnrs"] = st.slider(
                #         "Do you frequently wake up during the night and have trouble getting back to sleep?",
                #         1, 5, 3,
                #         label_visibility="collapsed"
                #     )

                # cols = st.columns([1,col_width])
                # with cols[0]:
                #     st.write("Do you frequently wake up too early in the morning and are unable to resume sleep?")
                # with cols[1]:
                #     wu2em = st.session_state.user_input["wu2em"] = st.slider(
                #         "Do you frequently wake up too early in the morning and are unable to resume sleep?",
                #         1, 5, 3,
                #         label_visibility="collapsed"
                #     )    

                # cols = st.columns([1,col_width])
                # with cols[0]:
                #     st.write("How ofter do you feel unrested during the day, no matter how many hours of sleep you had?")
                # with cols[1]:
                #     funres = st.session_state.user_input["funres"] = st.slider(
                #         "How ofter do you feel unrested during the day, no matter how many hours of sleep you had?",
                #         1, 5, 3,
                #         label_visibility="collapsed"
                #     )

                # cols = st.columns([1,col_width])
                # with cols[0]:
                #     st.write("How ofter do you feel excessively sleepy during the day? ")
                # with cols[1]:
                #     sleepy = st.session_state.user_input["sleepy"] = st.slider(
                #         "How ofter do you feel excessively sleepy during the day? ",
                #         1, 5, 3,
                #         label_visibility="collapsed"
                #     )

                ### ---------------------- Ver3 --------------------- ###
                col_width = 1
                cols = st.columns([1,col_width])
                with cols[1]:
                    st.markdown(
                        "<div style='display:flex; justify-content:space-between; font-size:14px;'>"
                        "<span>Never</span>"
                        "<span>Rarely</span>"
                        "<span>Sometimes</span>"
                        "<span>Often</span>"
                        "<span>Always</span>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown("""
                    <style>
                    /* Hide min and max numbers */
                    div[data-testid="stSlider"] > div div:nth-child(2) {
                        display: none;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                
                min_val = 1
                max_val = 5
                tfa = st.session_state.user_input["tfa"] = slider_generate(
                    "Do you have trouble falling asleep?", 
                    min_val, max_val, col_width)
                
                wudnrs = st.session_state.user_input["wudnrs"] = slider_generate(
                    "Do you frequently wake up during the night and have trouble getting back to sleep?", 
                    min_val, max_val, col_width)
                
                wu2em = st.session_state.user_input["wu2em"] = slider_generate(
                    "Do you frequently wake up too early in the morning and are unable to resume sleep?", 
                    min_val, max_val, col_width)
                
                funres = st.session_state.user_input["funres"] = slider_generate(
                    "How often do you feel unrested during the day, no matter how many hours of sleep you had?", 
                    min_val, max_val, col_width)
                
                sleepy = st.session_state.user_input["sleepy"] = slider_generate(
                    "How often do you feel excessively sleepy during the day? ", 
                    min_val, max_val, col_width)

                           
                
                
            with st.expander("OSA Symptoms"):
                st.write("""**Epworth Sleepiness Scale.**  
                    How likely are you to nod off or fall asleep in the following situations, in contrast to feeling just
                    tired? This refers to your usual way of life in recent times.
                    Even if you haven’t done some of these things recently, try to work out how they would have
                    affected you. It is important that you answer each question as best you can.
                    Use the following scale to choose the most appropriate number for each situation.   
                    """
                    # """
                    #     0 = would never doze  
                    #     1 = slight chance of dozing  
                    #     2 = moderate chance of dozing  
                    #     3 = high chance of dozing  
                    # """
                    )

                ### ---------------------- ESS questionnaire ver1 --------------------- ###
                # st.write("Select the likelihood of dozing in the following situations:")

                # cols = st.columns(2)
                # with cols[0]:
                #     ess1 = st.session_state.user_input["ess1"] = st.selectbox(
                #         "Sitting and reading",
                #         [0,1,2,3]
                #     )
                #     ess2 = st.session_state.user_input["ess2"] = st.selectbox(
                #         "Watching TV",
                #         [0,1,2,3]
                #     )
                #     ess3 = st.session_state.user_input["ess3"] = st.selectbox(
                #         "Sitting inactive in a public place (e.g., a theater or a meeting)",
                #         [0,1,2,3]
                #     )
                #     ess4 = st.session_state.user_input["ess4"] = st.selectbox(
                #         "As a passenger in a car for an hour or more without stopping for a break",
                #         [0,1,2,3]
                #     )

                # with cols[1]:
                    
                #     ess6 = st.session_state.user_input["ess6"] = st.selectbox(
                #         "Sitting and talking to someone",
                #         [0,1,2,3]
                #     )
                #     ess7 = st.session_state.user_input["ess7"] = st.selectbox(
                #         "Sitting quietly after a meal without alcohol",
                #         [0,1,2,3]
                #     )
                #     ess8 = st.session_state.user_input["ess8"] = st.selectbox(
                #         "In a car, while stopped for a few minutes in traffic              \n.",
                #         [0,1,2,3]
                #     )
                #     ess5 = st.session_state.user_input["ess5"] = st.selectbox(
                #         "Lying down to rest in the afternoon when circumstances permit",
                #         [0,1,2,3]
                #     )

                ### ---------------------- ESS questionnaire ver2 --------------------- ###
                col_width = 1
                cols = st.columns([1,col_width])
                with cols[0]:
                    st.write("Select the likelihood of **dozing** in the following situations:")
                with cols[1]:

                    st.markdown("""
                        <div style="display:flex; width:100%; font-size:15px;">
                            <div style="flex:1; text-align:left;">Would never doze</div>
                            <div style="flex:1; text-align:center;">Slight chance</div>
                            <div style="flex:1; text-align:center;">Moderate chance</div>
                            <div style="flex:1; text-align:right;">High chance</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                min_val = 0
                max_val = 3
                ess1 = st.session_state.user_input["ess1"] = slider_generate(
                    "Sitting and reading", 
                    min_val, max_val, col_width)
                
                ess2 = st.session_state.user_input["ess2"] = slider_generate(
                    "Watching TV", 
                    min_val, max_val, col_width)
                
                ess3 = st.session_state.user_input["ess3"] = slider_generate(
                    "Sitting inactive in a public place (e.g., a theater or a meeting)", 
                    min_val, max_val, col_width)
                
                ess4 = st.session_state.user_input["ess4"] = slider_generate(
                    "As a passenger in a car for an hour or more without stopping for a break", 
                    min_val, max_val, col_width)
                
                ess5 = st.session_state.user_input["ess5"] = slider_generate(
                    "Lying down to rest in the afternoon when circumstances permit", 
                    min_val, max_val, col_width)
                
                ess6 = st.session_state.user_input["ess6"] = slider_generate(
                    "Sitting and talking to someone", 
                    min_val, max_val, col_width)
                
                ess7 = st.session_state.user_input["ess7"] = slider_generate(
                    "Sitting quietly after a meal without alcohol", 
                    min_val, max_val, col_width)
                
                ess8 = st.session_state.user_input["ess8"] = slider_generate(
                    "In a car, while stopped for a few minutes in traffic", 
                    min_val, max_val, col_width)
                ### ------------------- calculate ess -------------------------- ###
                ess_score = (ess1 + ess2 + ess3 + ess4 + ess5 + ess6 + ess7 + ess8
                )

                st.write("**OSA Symptoms**")
                hosnr = st.session_state.user_input["hosnr"] = st.selectbox(
                    "How often do you snore?",
                    sorted(self.df_x["hosnr"].dropna().unique())
                )

                loudsn = st.session_state.user_input["loudsn"] = st.selectbox(
                    "How loud is your snoring?",
                    sorted(self.df_x["loudsn"].dropna().unique())
                )

                SAhistory = st.session_state.user_input["SAhistory"] = st.selectbox(
                    "Have you ever been diagnosed with sleep apnea in the past?",
                    ["No", "Yes"]
                )
                match SAhistory:
                    case "No":
                        SAhistory = "N"
                    case "Yes": 
                        SAhistory = "Y"
                
            with st.expander("Daily habits"):
                st.write("""Several lifestyle factors that may affect the risk to developing sleep apnea""")
                
                # st.write(self.df_x["genhth_s1"].unique())
                genhth = st.session_state.user_input["genhth"] = st.selectbox(
                    "How is your general health?", 
                    sorted(self.df_x["genhth"].dropna().unique()),
                    # ["1_Excellent", 
                    #  "2_Very_good",
                    #  "3_Good",
                    #  "4_Fair",
                    #  "5_Poor"],
                    index=2
                )

                cols = st.columns(3)
                with cols[0]:
                    ever_smoker = st.session_state.user_input["ever_smoker"] = st.selectbox(
                        "🚬 Have you ever smoked?",
                        self.df_x["ever_smoker"].unique(),
                        index=1
                    )
                with cols[1]:
                    current_smoker = st.session_state.user_input["current_smoker"] = st.selectbox(
                        "Are you currently smoking?",
                        self.df_x["current_smoker"].unique()
                    )
                with cols[2]:
                    cigday = st.session_state.user_input["cigday"] = st.number_input(
                        "Cigarettes per day (last 3 months)", 0, 60, 0
                    )

                if st.session_state.user_input["current_smoker"] == "Yes":
                    smoker = "Current"
                else:
                    if st.session_state.user_input["ever_smoker"] == "Yes":
                        smoker = "Past"
                    else:
                        smoker = "Never"


                drinker = st.session_state.user_input["drinker"] = st.selectbox(
                    "Do you drink alcohol?",
                    self.df_x["nondrinker"].unique()
                )

                if st.session_state.user_input["drinker"] == "No":
                    nondrinker = "Y"
                else:
                    nondrinker = "N"
                
                st.write("In the past 3 months, how much have you been drinking?")
                cols = st.columns(4)
                with cols[0]:
                    alcoh = st.session_state.user_input["alcoh"] = st.number_input(
                        "🍺Beer/wine per day", 0, 60, 0
                    )
                with cols[1]:
                    coffee = st.session_state.user_input["coffee"] = st.number_input(
                        ":coffee: Cups of coffee per day", 0, 10, 0
                    )
                with cols[2]:
                    soda = st.session_state.user_input["soda"] = st.number_input(
                        "🥤Can of sodas per day", 0, 10, 0
                    )
                with cols[3]:
                    tea = st.session_state.user_input["tea"] = st.number_input(
                        "🍵Cups of tea per day", 0, 10, 0
                    )
            demographics_list = [age, education, sex, race]
            anthropometric_list = [age, height, weight, neck, bmi]
            comorbidities_list = [emphys, asthma, htnderv]
            sleep_habits_list = [tfa, wudnrs, wu2em, funres, sleepy, mi2slp, hrswe, hrswd, totminnap]
            OSA_symptoms_list = [hosnr, loudsn, ess1, ess2, ess3, ess4, ess5, ess6, ess7, ess8, ess_score, SAhistory]
            dailyHabits_list = [genhth, smoker, current_smoker, ever_smoker, nondrinker, cigday, alcoh, coffee, soda, tea]

            total_len= len(demographics_list) + len(anthropometric_list) + len(comorbidities_list) + len(sleep_habits_list) + len(OSA_symptoms_list) + len(dailyHabits_list)
            total_input = sum(v is not None for v in demographics_list + anthropometric_list + comorbidities_list + sleep_habits_list + OSA_symptoms_list + dailyHabits_list)
            
            if total_input < total_len:
                st.caption(f"Progress: {total_input}/{total_len} inputs completed")
                st.progress(total_input/total_len)

                cols = st.columns(4)
                if None in demographics_list:
                    st.warning("Please complete demographics information.")
                if None in anthropometric_list:
                    st.warning("Please complete anthropometric information.")
                if None in comorbidities_list:
                    st.warning("Please complete comorbidities information.")
                if None in dailyHabits_list:
                    st.warning("Please complete daily habits information.")
                if None in OSA_symptoms_list:
                    st.warning("Please complete OSA symptoms information.")
                if None in sleep_habits_list:
                    st.warning("Please complete sleep habits information.")
            elif total_input == total_len:
                st.success("All inputs completed! You can now check the output in the next tab.")
                # st.balloons()
                

            if None not in demographics_list and None not in anthropometric_list:
                input_df = pd.DataFrame({
                    "sleep_apnea_history": [SAhistory],
                    "education_common": [education],
                    "sex_common": [sex],
                    "race_common": [race],
                    "smoker": [smoker],
                    "current_smoker": [current_smoker],
                    "ever_smoker": [ever_smoker],
                    "nondrinker": [nondrinker],
                    "hosnr": [hosnr],
                    "loudsn": [loudsn],
                    "emphys": [emphys],
                    "asthma": [asthma],
                    "htnderv": [htnderv],
                    "genhth": [genhth],
                    "bmi": [bmi],
                    "height": [height],
                    "weight": [weight],
                    "neck": [neck],
                    "age": [age],
                    "cigday": [cigday],
                    "alcoh": [alcoh],
                    "coffee": [coffee],
                    "soda": [soda],
                    "ess1": [ess1],
                    "ess2": [ess2],
                    "ess3": [ess3],
                    "ess4": [ess4],
                    "ess5": [ess5],
                    "ess6": [ess6],
                    "ess7": [ess7],
                    "ess8": [ess8],
                    "ess_score": [ess_score],
                    "tfa": [tfa],
                    "wudnrs": [wudnrs],
                    "wu2em": [wu2em],
                    "funres": [funres],
                    "sleepy": [sleepy],
                    "mi2slp": [mi2slp],
                    "totsleep": [totsleep],
                    "totsleepnap": [totsleepnap],
                    "hrswe": [hrswe],
                    "hrswd": [hrswd],
                    "totminnap": [totminnap]
                })

                ### ------------------------ Summary of user input ------------------------- ###
                with tabs[1]:
                        
                        st.dataframe(input_df.T.rename(columns={0:"Your input"}), use_container_width=True)
                        feature_col = self.df_x.columns
                        # st.write(feature_col[1:])
                        ref_df = self.df_x[feature_col[1:]]
                        try:
                            model_input_df = align_input_columns(input_df, feature_col[1:])
                        except ValueError as error:
                            st.error(str(error))
                            st.stop()
                        label_col = ["Label", "AHI"]
                        ref_df = pd.concat([ref_df, self.df_y[label_col]], axis=1).dropna()
                        # st.write(input_df)
                        # st.write(self.df_x.loc[0])

                    ### ------------------------------- Model prediction and SHAP explanation ------------------------------- ###
                    # Load model and run prediction

                        model = load_prediction_model()
                        y_test_pred = model.predict(model_input_df)
                        y_test_prob = model.predict_proba(model_input_df)[:,1]
                        prediction_label = int(np.asarray(y_test_pred).ravel()[0])
                        prediction_probability = float(y_test_prob[0])

                        # st.write(y_test_pred, y_test_prob)

                        explainer = load_shap_explainer(model)
                        shap_values = explainer(model_input_df)

                        # st.write("Prediction:", model.predict(X_test.iloc[[idx]])[0])
                        if prediction_label == 0:
                            st.write(f"Risk of having moderate sleep apnea: low ({prediction_probability*100:.2f}%)")
                        else:
                            st.write(f"Risk of having moderate sleep apnea: high ({prediction_probability*100:.2f}%)")

                        # st.write(f"Risk of having moderate sleep apnea: {y_test_prob[0]*100:.2f}%")
                        # st.write(shap_values)
                        # Correct force plot usage for SHAP >= 0.20
                        force_plot = shap.plots.force(
                            explainer.expected_value,
                            shap_values[0].values,
                            features=model_input_df.iloc[0],
                            feature_names=model_input_df.columns
                        )
                        st_shap(force_plot)

                        st.markdown(
                                    '<a href="https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html" '
                                    'target="_blank">An introduction to explainable AI with Shapley values</a>',
                                    unsafe_allow_html=True
                                    )


                    ### ------------------------ Display input along with reference distribution ------------------------- ###    
                        st.write(f"Risk of developing sleep apnea is influenced by various demographic and health factors. " \
                        "Below are some of your key characteristics compared to the distribution in our reference population, " \
                        f"including {len(ref_df)} individuals from the Sleep Heart Health Study dataset.")
                        # with st.container():
                        # with st.expander("Your profile compared to reference population"):
                        with st.expander("Sleep Apnea Patterns in Men and Women"):
                            # st.write("**Your profile**")
                            cols = st.columns([1,4], vertical_alignment="center")
                            if sex == "male":
                                st.write("""In this group, only 7.9\% of men had normal results, while 31.5% had mild, 35.1% had moderate, and 25.5% had severe sleep apnea. 
                                        This means that about 60% of men were found to have moderate to severe sleep apnea. 
                                        These findings suggest that men may be at higher risk for more serious forms of sleep apnea. """)
                                        #  If you experience symptoms such as loud snoring, breathing pauses during sleep, or daytime fatigue, 
                                        #  it is especially important to consider evaluation and possible treatment.
                            else: 
                                st.write("""In this group, 23.3% of women had normal results, while 44.7% had mild, 21.9% had moderate, and 10.1% had severe sleep apnea. 
                                         Compared to men, women were more likely to have mild sleep apnea and less likely to have severe forms. 
                                         However, nearly 1 in 3 women still had moderate to severe sleep apnea. 
                                         Even milder forms can affect sleep quality and overall health, so ongoing symptoms like fatigue, poor sleep, or morning headaches should not be ignored.""")

                            with cols[0]:
                                if sex == "male":
                                    st.image(os.path.join(IMAGES_DIR,"Men_color.png"),
                                                use_container_width=True)
                                else:
                                    st.image(os.path.join(IMAGES_DIR,"Women_color.png"),
                                                use_container_width=True)
                            with cols[1]:
                                fig = plot_grouped_pie_chart(ref_df,
                                        group_name = ["sex", "Label", "AHI"])
                                plt.suptitle("Distribution of sleep apnea severity separated by gender.")
                                fig.tight_layout()
                                st.pyplot(fig, use_container_width=True)

                        ref_df["age"] = ref_df["age"].astype(int)
                        df_age = grouped_bar_chart(ref_df, group_name = ["age", "Label", "AHI"])


                        with st.expander("Sleep Apnea Patterns across Age Groups", expanded=True):
                            # ref_df["age_s1"] = pd.cut(ref_df["age_s1"], 
                            #                           bins=[0, 18, 30, 40, 50, 60, 70, 80, 90, 100], 
                            #                           labels=["<18", "18-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "90+"])
                            
                            available_ages = sorted(ref_df["age"].dropna().astype(int).unique())
                            nearest_age = min(available_ages, key=lambda value: abs(value - age))
                            if (
                                "user_select_age" not in st.session_state
                                or st.session_state.user_select_age not in available_ages
                            ):
                                st.session_state.user_select_age = nearest_age
                            cols = st.columns([1,1])

                            with cols[0]:
                                st.write(f"""Sleep apnea is more common in older adults, with prevalence increasing significantly after age 40. In our reference population, 
                                     the proportion of individuals with moderate to severe sleep apnea rises sharply in the 50-60 age group and continues to increase in older age groups. 
                                     However, sleep apnea can affect adults of all ages, including younger individuals, especially those with risk factors such as obesity or a family history of the condition. 
                                     """)
                            with cols[1]:
                                cols = st.columns(2)
                                with cols[0]:
                                    if st.button("Younger Group"):
                                        current_index = available_ages.index(st.session_state.user_select_age)
                                        st.session_state.user_select_age = available_ages[max(0, current_index - 1)]

                                with cols[1]:
                                    if st.button("Older Group"):
                                        current_index = available_ages.index(st.session_state.user_select_age)
                                        st.session_state.user_select_age = available_ages[min(len(available_ages) - 1, current_index + 1)]

                                # st.write("Selected Age:", st.session_state.user_select_age)
                                age_specific_df = ref_df[ref_df["age"] == st.session_state.user_select_age].reset_index(drop=True)
                                # st.dataframe(age_specific_df[["age_s1", "AHI_label1", "AHI1"]])
                                fig = plot_grouped_pie_chart(age_specific_df,
                                                                 group_name = ["age", "Label", "AHI"])
                                plt.suptitle("")
                                plt.title(f"Distribution of sleep apnea severity \nin individuals aged {st.session_state.user_select_age}.")
                                plt.tight_layout()
                                st.pyplot(fig, 
                                        #   use_container_width=True
                                          )
                                
                            fig, ax = plot_grouped_bar_chart(df_age,
                                    group_name = ["age", "Label", "AHI"],
                                    user_age=st.session_state.user_select_age)
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=True)   

                        with st.expander("Sleep Apnea Patterns across BMI Groups", expanded=True):
                            ref_df["bmi"] = pd.cut(ref_df["bmi"], 
                                                      bins=[0, 18.5, 25, 30, 35, 40, 70],
                                                      labels=["Underweight", "Healthy weight", "Overweight", "Obesity I", "Obesity II", "Obesity III"])
                            
                            fig = plot_grouped_pie_chart(ref_df,
                                    group_name = ["bmi", "Label", "AHI"],
                                    highlight_group = bmi_range)
                            plt.suptitle("Distribution of sleep apnea severity separated by BMI groups.")
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=True)                         
                            
                            st.write("""The figure illustrates the distribution of obstructive sleep apnea (OSA) severity across different BMI categories, 
                                     including underweight, healthy weight, overweight, and obesity classes I–III. 
                                     A clear progressive pattern is observed as BMI increases. In the underweight and healthy weight groups, 
                                     normal and mild OSA constitute the majority of cases, while severe OSA remains relatively uncommon. 
                                     As BMI shifts to the overweight category, the proportion of moderate and severe OSA begins to increase, 
                                     accompanied by a reduction in normal cases. This trend becomes more pronounced in Obesity I, 
                                     where moderate and severe OSA together account for a substantial proportion of individuals. 
                                     In Obesity II and III, severe OSA becomes the dominant category, while normal cases nearly disappear. 
                                     Overall, the figure demonstrates a dose–response relationship between BMI and OSA severity, 
                                     with higher BMI strongly associated with increased prevalence of moderate-to-severe OSA.""")

                        with st.expander("Sleep Apnea Patterns across Snoring Frequency", expanded=True):
                            ax_title = ["Never or rarely \nonce or few time every week",
                                        "Sometimes \na few nights per month\n under special circumstances",
                                        "At least once a week \nbut pattern may be irregular",
                                        "3 to 5 nights per week",
                                        "Every night or almost every night",
                                        "Unknown"]
                            fig = plot_grouped_pie_chart(ref_df,
                                    group_name = ["hosnr", "Label", "AHI"],
                                    highlight_group = hosnr,
                                    ax_title = ax_title)
                            plt.suptitle("Distribution of sleep apnea severity separated by snoring frequency.")
                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=True)                         
                            
                            st.write("""The figure illustrates the distribution of obstructive sleep apnea (OSA) severity across different snoring frequency categories, 
                                     including never, rarely, sometimes, often, and almost always. 
                                     A clear progressive pattern is observed as snoring frequency increases. In the never and rarely snoring groups, 
                                     normal and mild OSA constitute the majority of cases, while severe OSA remains relatively uncommon. 
                                     As snoring frequency shifts to sometimes and often categories, the proportion of moderate and severe OSA begins to increase, 
                                     accompanied by a reduction in normal cases. This trend becomes more pronounced in the almost always snoring group, 
                                     where severe OSA becomes the dominant category, while normal cases nearly disappear. 
                                     Overall, the figure demonstrates a dose–response relationship between snoring frequency and OSA severity, 
                                     with higher snoring frequency strongly associated with increased prevalence of moderate-to-severe OSA.""")
                        
                        ### ------------------------------- button to download report as PDF ------------------------------- ###
                        # markdown_text = """
                        # Predicted OSA risk: 68%

                        # This report summarizes the user's predicted risk and feature contributions.
                        # """

                        # pdf_buffer = generate_pdf_report(
                        #     user_df=input_df,
                        #     figures=[fig],
                        #     markdown_text=markdown_text
                        # )

                        # st.download_button(
                        #     label="Download Full Report (PDF)",
                        #     data=pdf_buffer,
                        #     file_name="OSA_report.pdf",
                        #     mime="application/pdf"
                        # )
                        force_fig = create_force_plot(explainer, shap_values, model_input_df)

                        prediction_text = f"""
                        Predicted OSA risk: {prediction_probability*100:.2f}
                        """

                        pdf_buffer = generate_pdf(input_df, prediction_text, force_fig)

                        # st.download_button(
                        #     label="Download Report (PDF)",
                        #     data=pdf_buffer,
                        #     file_name="OSA_report.pdf",
                        #     mime="application/pdf"
                        # )

            
