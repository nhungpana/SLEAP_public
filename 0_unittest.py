import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Simple Streamlit Demo App")

# Sidebar
st.sidebar.header("Controls")
num_points = st.sidebar.slider("Number of points", 10, 200, 50)

# Main content
st.write("This is a simple interactive app.")

# Generate random data
x = np.linspace(0, 10, num_points)
y = np.sin(x)

# Plot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Sine Wave")
ax.set_xlabel("X")
ax.set_ylabel("sin(X)")

st.pyplot(fig)

# Dataframe example
st.subheader("Sample Data")
df = pd.DataFrame({
    "X": x,
    "sin(X)": y
})

st.dataframe(df)

# Button example
if st.button("Say Hello"):
    st.success("Hello Pana 👋")