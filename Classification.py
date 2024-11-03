import streamlit as st
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/fakhitah3/jie43203/refs/heads/main/Iris.csv")

st.write(df)
