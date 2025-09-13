import streamlit as st
import time
import base64


def main():
    # RUL Estimation
    st.set_page_config(page_title="Remaining Useful Life Estimation", page_icon="📉")
    st.title("📉 RUL Estimation")
    st.sidebar.markdown("[💻 GitHub Repository](https://github.com/palscruz23/rul-estimation)")

    # Remaining Life vs Sensor Measurement of all units
    st.markdown(
        """
        ### Construct Asset Health Indicator
        """
    )
    st.image("src/figures/plot_health_indicator.png", caption=None,  output_format ="png")

    # Estimation RUL in All Validation Engine
    st.markdown(
        """
        ### Estimation RUL in All Validation Engine
        """
    )
    st.image("src/figures/plot_val.png", caption=None,  output_format ="png")

    # Estimation RUL in Validation Engine
    st.markdown(
        """
        ### Estimation RUL in Validation Engine
        """
    )
    file_ = open("src/figures/RUL/RUL.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="RUL Estimation">',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()