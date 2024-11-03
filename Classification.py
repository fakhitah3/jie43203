import streamlit as st

import numpy as np
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/fakhitah3/jie43203/refs/heads/main/Iris.csv")

st.write(df)
