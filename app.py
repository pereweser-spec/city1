import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Настройка страницы
st.set_page_config(page_title="City Predictive Platform", layout="wide")

# Заголовок
st.title("🏙️ Гибридная мультиагентная платформа")
st.caption("Предиктивная аналитика городской инфраструктуры")

# Боковая панель
st.sidebar.header("⚙️ Параметры агентов")
city = st.sidebar.selectbox("Выберите район города", ["Центральный", "Северный", "Южный", "Западный"])
horizon = st.sidebar.slider("Горизонт прогноза (дней)", 7, 90, 30)

# 1. Агент данных (симуляция получения данных)
class DataAgent:
    def get_data(self, city):
        # Генерация временного ряда (имитация данных)
        dates = pd.date_range(end=datetime.now(), periods=365)
        base = np.sin(np.linspace(0, 20, 365)) * 50 + 100
        noise = np.random.normal(0, 10, 365)
        traffic = base + noise
        energy = base * 1.5 + np.random.normal(0, 20, 365)
        return pd.DataFrame({"Дата": dates, "Трафик": traffic, "Энергия": energy}, ).set_index("Дата")

# 2. Агент прогнозирования (простая модель)
class PredictAgent:
    def predict(self, df, days):
        # Простая скользящая средняя для прогноза
        last_values = df['Трафик'].tail(10).mean()
        forecast = [last_values + np.random.normal(0, 5) for _ in range(days)]
        forecast_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=days)
        return pd.DataFrame({"Дата": forecast_dates, "Прогноз_Трафик": forecast}).set_index("Дата")

# 3. Агент отчетов (визуализация)
class ReportAgent:
    def generate_report(self, data, forecast):
        st.subheader("📈 Текущая нагрузка (данные)")
        st.line_chart(data['Трафик'])
        
        st.subheader("🔮 Прогноз нагрузки на инфраструктуру")
        st.line_chart(forecast['Прогноз_Трафик'])
        
        # Расчет рисков
        risk = "Высокий" if forecast['Прогноз_Трафик'].mean() > data['Трафик'].mean() else "Низкий"
        st.metric("Прогнозируемый риск перегрузки", risk)

# Запуск агентов
data_agent = DataAgent()
predict_agent = PredictAgent()
report_agent = ReportAgent()

with st.spinner("Агенты анализируют городские данные..."):
    df = data_agent.get_data(city)
    forecast = predict_agent.predict(df, horizon)
    report_agent.generate_report(df, forecast)

st.success("✅ Все агенты отработали успешно!")
