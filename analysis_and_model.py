import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train_model(X_train, y_train, model_type):
    if model_type == "Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_type == "SVM":
        model = SVC(kernel='linear', random_state=42, probability=True)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True )
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return accuracy, conf_matrix, class_report, roc_auc, y_pred_proba

def analysis_and_model_page():
    st.title("Анализ данных и модель")

    uploaded_file = st.file_uploader("Загрузите файл: data/predictive_maintenance.csv", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Очистка названий столбцов от символов [, ], <, >
        data.columns = data.columns.str.replace(r"[\[\]<>]", "", regex=True)

        # Предобработка
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']

        # Масштабирование числовых признаков
        numerical_features = ['Air temperature K', 'Process temperature K', 'Rotational speed rpm', 'Torque Nm', 'Tool wear min']
        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])

        # Разделение
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Выбор модели
        model_type = st.selectbox("Выберите модель", ["Logistic Regression", "Random Forest", "XGBoost", "SVM"])

        if st.button("Обучить модель"):
            model = train_model(X_train, y_train, model_type)

            # Оценка
            accuracy, conf_matrix, class_report, roc_auc, y_pred_proba = evaluate_model(model, X_test, y_test)

            st.header("Результаты модели")
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"ROC-AUC: {roc_auc:.2f}")

            st.subheader("Матрица ошибок")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            st.subheader("Classification Report")
            report_df = pd.DataFrame(class_report).transpose()
            st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

            st.subheader("ROC-кривая")
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"{model_type} (AUC = {roc_auc:.2f})")
            ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC-кривая')
            ax2.legend()
            st.pyplot(fig2)

            # Сохраняем модель и scaler в session_state
            st.session_state.model = model
            st.session_state.scaler = scaler

    # Предсказание
    st.header("Предсказание на новых данных")
    with st.form("prediction_form"):
        st.write("Введите значения признаков:")
        type_ = st.selectbox("Type", ["L", "M", "H"])
        air_temp = st.number_input("Air temperature K")
        process_temp = st.number_input("Process temperature K")
        rotational_speed = st.number_input("Rotational speed rpm")
        torque = st.number_input("Torque Nm")
        tool_wear = st.number_input("Tool wear min")

        submit = st.form_submit_button("Предсказать")

        if submit:
            if 'model' in st.session_state and 'scaler' in st.session_state:
                type_map = {"L": 0, "M": 1, "H": 2}
                input_data = pd.DataFrame([{
                    'Type': type_map[type_],
                    'Air temperature K': air_temp,
                    'Process temperature K': process_temp,
                    'Rotational speed rpm': rotational_speed,
                    'Torque Nm': torque,
                    'Tool wear min': tool_wear
                }])

                # Масштабирование
                input_data[numerical_features] = st.session_state.scaler.transform(input_data[numerical_features])

                # Предсказание
                prediction = st.session_state.model.predict(input_data)[0]
                prediction_proba = st.session_state.model.predict_proba(input_data)[0, 1]

                st.subheader("Результат:")
                if prediction == 1:
                    st.error(f"Прогнозируется отказ оборудования (вероятность: {prediction_proba:.2%})")
                else:
                    st.success(f"Оборудование в норме (вероятность отказа: {prediction_proba:.2%})")
            else:
                st.warning("Сначала обучите модель.")
