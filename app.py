import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Навигация между страницами
pages = {
    "Анализ и модель": "analysis_and_model",
    "Презентация": "presentation",
}

selected_page = st.sidebar.radio("Выберите страницу", list(pages.keys()))

# Запуск выбранной страницы
if selected_page == "Анализ и модель":
    import analysis_and_model
    analysis_and_model.analysis_and_model_page()

elif selected_page == "Презентация":
    import presentation
    presentation.presentation_page()
