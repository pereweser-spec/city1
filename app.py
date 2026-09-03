import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# 1. АГЕНТ ДАННЫХ (Data Agent)
# ==========================================
class DataAgent:
    def __init__(self):
        self.raw_data = None
    
    def generate_synthetic_data(self, days=500):
        # Генерация правдоподобных данных городской инфраструктуры
        dates = pd.date_range(end=datetime.now(), periods=days)
        
        # Сезонность и тренд для трафика
        t = np.arange(days)
        traffic = 100 + 50 * np.sin(2 * np.pi * t / 365) + np.sin(2 * np.pi * t / 7) * 20 + np.random.normal(0, 5, days)
        
        # Потребление энергии (зависит от трафика)
        energy = traffic * 1.2 + np.random.normal(0, 15, days) + 20
        
        # Уровень загрязнения
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
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
    
    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_and_predict(self, df, column='Трафик', forecast_days=30):
        # Подготовка данных для нейросети
        data = df[column].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        # Создание последовательностей
        seq_length = 60
        x_train, y_train = [], []
        for i in range(seq_length, len(scaled_data)):
            x_train.append(scaled_data[i-seq_length:i, 0])
            y_train.append(scaled_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Обучение модели
        self.model = self.build_lstm_model((x_train.shape[1], 1))
        self.model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
        
        # Прогнозирование на будущее
        last_60_days = scaled_data[-seq_length:]
        curr_seq = last_60_days.reshape(1, seq_length, 1)
        
        predictions = []
        for _ in range(forecast_days):
            pred = self.model.predict(curr_seq, verbose=0)[0][0]
            predictions.append(pred)
            curr_seq = np.append(curr_seq[:, 1:, :], [[[pred]]], axis=1)
        
        # Обратное масштабирование
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Даты прогноза
        last_date = df['Дата'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        forecast_df = pd.DataFrame({
            'Дата': forecast_dates,
            f'Прогноз_{column}': predictions.flatten()
        })
        
        return forecast_df

# ==========================================
# 3. АГЕНТ ВИЗУАЛИЗАЦИИ И ОТЧЕТОВ (Report Agent)
# ==========================================
class ReportAgent:
    def generate_plots(self, historical, forecast, metric):
        fig = go.Figure()
        
        # Исторические данные
        fig.add_trace(go.Scatter(
            x=historical['Дата'], y=historical[metric],
            mode='lines', name='Исторические данные',
            line=dict(color='blue')
        ))
        
        # Прогноз
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
st.markdown("### Предиктивная аналитика городской инфраструктуры на базе ИИ")

# Инициализация агентов
data_agent = DataAgent()
prediction_agent = PredictionAgent()
report_agent = ReportAgent()

# Боковая панель
with st.sidebar:
    st.header("⚙️ Управление платформой")
    
    # Выбор источника данных
    data_source = st.radio("Источник данных:", ["Синтетические данные (демо)", "Загрузить CSV файл"])
    
    uploaded_file = None
    if data_source == "Загрузить CSV файл":
        uploaded_file = st.file_uploader("Загрузите CSV", type=['csv'])
        if uploaded_file is not None:
            df = data_agent.load_real_data(uploaded_file)
        else:
            st.warning("Файл не загружен, использую синтетику")
            df = data_agent.generate_synthetic_data()
    else:
        df = data_agent.generate_synthetic_data()
    
    # Выбор метрики
    metric = st.selectbox("Что прогнозируем?", ["Трафик", "Энергия", "Загрязнение"])
    
    # Горизонт прогноза
    forecast_days = st.slider("Дней для прогноза", 7, 90, 30)
    
    # Кнопка запуска
    run_button = st.button("🚀 Запустить агентов")

# Основной экран
if run_button:
    with st.spinner("🤖 Агенты собирают данные, обучают нейросеть и строят прогноз..."):
        # 1. Агент данных работает
        st.subheader(f"📊 Данные по показателю: {metric}")
        st.dataframe(df.tail(10))
        
        # 2. Агент прогнозирования обучает LSTM
        forecast_df = prediction_agent.train_and_predict(df, column=metric, forecast_days=forecast_days)
        
        # 3. Агент визуализации строит график
        fig = report_agent.generate_plots(df, forecast_df, metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Агент рисков
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
