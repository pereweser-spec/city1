import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import ollama # <-- Библиотека для работы с локальной Llama

# ==========================================
# 1. АГЕНТ ДАННЫХ
# ==========================================
class DataAgent:
    def generate_synthetic_data(self, days=500):
        dates = pd.date_range(end=datetime.now(), periods=days)
        t = np.arange(days)
        
        traffic = 100 + 50 * np.sin(2 * np.pi * t / 365) + np.sin(2 * np.pi * t / 7) * 20 + np.random.normal(0, 5, days)
        energy = traffic * 1.2 + np.random.normal(0, 15, days) + 20
        pollution = traffic * 0.05 + np.random.normal(0, 3, days)
        
        self.raw_data = pd.DataFrame({
            'Дата': dates,
            'Трафик': traffic,
            'Энергия': energy,
            'Загрязнение': pollution
        })
        return self.raw_data

# ==========================================
# 2. АГЕНТ ПРОГНОЗИРОВАНИЯ (RandomForest)
# ==========================================
class PredictionAgent:
    def train_and_predict(self, df, column='Трафик', forecast_days=30):
        df = df.copy()
        df['Time'] = (df['Дата'] - df['Дата'].min()).dt.days
        
        X = df[['Time']].values
        y = df[column].values
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        last_day = df['Time'].max()
        future_days = np.arange(last_day + 1, last_day + forecast_days + 1).reshape(-1, 1)
        predictions = model.predict(future_days)
        
        last_date = df['Дата'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        forecast_df = pd.DataFrame({
            'Дата': forecast_dates,
            f'Прогноз_{column}': predictions
        })
        return forecast_df

# ==========================================
# 3. АГЕНТ-АНАЛИТИК (Llama 3 для текстовых отчетов)
# ==========================================
class LlamaAnalystAgent:
    def __init__(self, model_name='llama3.2:latest'):
        self.model_name = model_name

    def generate_report(self, historical, forecast, metric):
        # Формируем краткую сводку для модели
        avg_hist = historical[metric].mean()
        avg_forecast = forecast[f'Прогноз_{metric}'].mean()
        risk = "высокий" if avg_forecast > avg_hist * 1.2 else ("умеренный" if avg_forecast > avg_hist else "низкий")
        
        prompt = f"""
        Ты — аналитик городской инфраструктуры. 
        Средний исторический показатель '{metric}' равен {avg_hist:.2f}.
        Прогнозируемый показатель на будущий период равен {avg_forecast:.2f}.
        Риск перегрузки инфраструктуры: {risk}.
        
        Напиши краткий отчет (3-4 предложения) для мэрии города с рекомендациями по улучшению ситуации.
        """

        try:
            # Отправляем запрос в локальную Ollama
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response['response'].strip()
        except Exception as e:
            return f"Ошибка при обращении к Llama: {str(e)}. Убедитесь, что Ollama запущена (команда: ollama serve)."

# ==========================================
# ГЛАВНЫЙ ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.set_page_config(page_title="CityPredict AI", layout="wide")

st.title("🏙️ Гибридная мультиагентная платформа")
st.markdown("### Предиктивная аналитика + Генеративный ИИ (Llama)")

# Инициализация агентов
data_agent = DataAgent()
prediction_agent = PredictionAgent()
analyst_agent = LlamaAnalystAgent()

# Боковая панель
with st.sidebar:
    st.header("⚙️ Управление")
    metric = st.selectbox("Что прогнозируем?", ["Трафик", "Энергия", "Загрязнение"])
    forecast_days = st.slider("Дней для прогноза", 7, 90, 30)
    run_button = st.button("🚀 Запустить агентов")

# Основной экран
if run_button:
    with st.spinner("🤖 Агенты собирают данные, обучают модель и вызывают Llama..."):
        df = data_agent.generate_synthetic_data()
        
        st.subheader(f"📊 Данные по показателю: {metric}")
        st.dataframe(df.tail(10))
        
        forecast_df = prediction_agent.train_and_predict(df, column=metric, forecast_days=forecast_days)
        
        # Визуализация
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Дата'], y=df[metric], mode='lines', name='Исторические данные'))
        fig.add_trace(go.Scatter(x=forecast_df['Дата'], y=forecast_df[f'Прогноз_{metric}'], mode='lines', name='Прогноз ИИ', line=dict(dash='dash')))
        fig.update_layout(template='plotly_dark', title=f'Прогноз: {metric}')
        st.plotly_chart(fig, use_container_width=True)
        
        # Генерация текстового отчета от Llama
        st.subheader("🤖 Отчет от Агента-Аналитика (Llama 3)")
        with st.spinner("Llama думает..."):
            report_text = analyst_agent.generate_report(df, forecast_df, metric)
            st.markdown(report_text)

        st.success("✅ Все агенты успешно отработали!")
else:
    st.info("👈 Нажмите кнопку 'Запустить агентов' в боковой панели, чтобы начать анализ.")
