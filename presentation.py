import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Цель: предсказание отказов оборудования
    - Модель: бинарная классификация
    ---
    ## Этапы работы
    1. Загрузка и анализ данных
    2. Предобработка
    3. Обучение моделей
    4. Визуализация и метрики
    ---
    ## Используемые модели
    - Logistic Regression
    - Random Forest
    - XGBoost
    - SVM
    ---
    ## Метрики оценки
    - Accuracy
    - Confusion Matrix
    - ROC-AUC
    ---
    ## Итоги
    - Построено Streamlit-приложение
    - Реализовано предсказание на новых данных
    """

    with st.sidebar:
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "night"])
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom"])

    rs.slides(
        markdown,
        height=500,
        theme=theme,
        config={"transition": transition},
    )
