import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. АГЕНТ ДАННЫХ (Data Agent)
# ==========================================
class DataAgent:
    def generate_synthetic_data(self, days=500):
        dates = pd.date_range(end=datetime.now(), periods=days)
        t = np.arange(days)
        
        # Сезонность и тренд
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

    def load_real_data(self, uploaded_file):
        if uploaded_file is not None:
            self.raw_data = pd.read_csv(uploaded_file)
            self.raw_data['Дата'] = pd.to_datetime(self.raw_data['Дата'])
            return self.raw_data
        return None

# ==========================================
# 2. АГЕНТ ПРОГНОЗИРОВАНИЯ (Prediction Agent)
# ==========================================
class PredictionAgent:
    def train_and_predict(self, df, column='Трафик', forecast_days=30):
        # Готовим данные для регрессии
        df = df.copy()
        df['Time'] = (df['Дата'] - df['Дата'].min()).dt.days
        
        X = df[['Time']].values
        y = df[column].values
        
        # Обучаем модель (случайный лес - быстрый и надежный)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Прогноз на будущее
        last_day = df['Time'].max()
        future_days = np.arange(last_day + 1, last_day + forecast_days + 1).reshape(-1, 1)
        predictions = model.predict(future_days)
        
        # Даты прогноза
        last_date = df['Дата'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        forecast_df = pd.DataFrame({
            'Дата': forecast_dates,
            f'Прогноз_{column}': predictions
        })
        
        return forecast_df

# ==========================================
# 3. АГЕНТ ВИЗУАЛИЗАЦИИ (Report Agent)
# ==========================================
class ReportAgent:
    def generate_plots(self, historical, forecast, metric):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical['Дата'], y=historical[metric],
            mode='lines', name='Исторические данные',
            line=dict(color='blue')
        ))
        
        forecast_col = f'Прогноз_{metric}'
        fig.add_trace(go.Scatter(
            x=forecast['Дата'], y=forecast[forecast_col],
            mode='lines', name='Прогноз ИИ',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Прогноз показателя: {metric}',
            xaxis_title='Дата',
            yaxis_title=metric,
            template='plotly_dark'
        )
        return fig
    
    def calculate_risks(self, historical, forecast, metric):
        avg_hist = historical[metric].mean()
        avg_forecast = forecast[f'Прогноз_{metric}'].mean()
        
        if avg_forecast > avg_hist * 1.2:
            risk = "🔴 Высокий риск перегрузки"
        elif avg_forecast > avg_hist:
            risk = "🟡 Умеренный риск"
        else:
            risk = "🟢 Риск минимален"
        
        return risk, avg_hist, avg_forecast

# ==========================================
# ГЛАВНЫЙ ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.set_page_config(page_title="CityPredict AI", layout="wide")

st.title("🏙️ Гибридная мультиагентная платформа")
st.markdown("### Предиктивная аналитика городской инфраструктуры")

# Инициализация агентов
data_agent = DataAgent()
prediction_agent = PredictionAgent()
report_agent = ReportAgent()

# Боковая панель
with st.sidebar:
    st.header("⚙️ Управление платформой")
    
    data_source = st.radio("Источник данных:", ["Синтетические данные (демо)", "Загрузить CSV файл"])
    
    uploaded_file = None
    if data_source == "Загрузить CSV файл":
        uploaded_file = st.file_uploader("Загрузите CSV", type=['csv'])
    
    # Выбор метрики
    metric = st.selectbox("Что прогнозируем?", ["Трафик", "Энергия", "Загрязнение"])
    
    # Горизонт прогноза
    forecast_days = st.slider("Дней для прогноза", 7, 90, 30)
    
    # Кнопка запуска
    run_button = st.button("🚀 Запустить агентов")

# Основной экран
if run_button:
    with st.spinner("🤖 Агенты собирают данные, обучают модель и строят прогноз..."):
        # Получаем данные
        if uploaded_file is not None:
            df = data_agent.load_real_data(uploaded_file)
        else:
            df = data_agent.generate_synthetic_data()
        
        st.subheader(f"📊 Данные по показателю: {metric}")
        st.dataframe(df.tail(10))
        
        # Прогнозирование
        forecast_df = prediction_agent.train_and_predict(df, column=metric, forecast_days=forecast_days)
        
        # Визуализация
        fig = report_agent.generate_plots(df, forecast_df, metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # Анализ рисков
        risk, avg_hist, avg_forecast = report_agent.calculate_risks(df, forecast_df, metric)
        st.markdown(f"### Анализ рисков: {risk}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Среднее историческое", round(avg_hist, 2))
        with col2:
            st.metric("Среднее прогнозное", round(avg_forecast, 2))
        with col3:
            st.metric("Изменение", f"{((avg_forecast - avg_hist) / avg_hist) * 100:.1f}%")
        
        st.success("✅ Все агенты успешно отработали! Прогноз готов.")

else:
    st.info("👈 Нажмите на кнопку 'Запустить агентов' в боковой панели, чтобы начать анализ и прогнозирование.")
