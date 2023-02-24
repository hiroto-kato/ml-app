import streamlit as st

# ここにないと二回読み込まれてエラーになる
st.set_page_config(page_icon="", page_title="ML App (PyCaret)", layout="wide")
from utils.general import *
from utils.lib import *
from pages.learn import *


def main():
    st.title("ML Application (PyCaret)")


if __name__ == "__main__":
    main()
