import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


def DataPreProcessing(df):
    df1 = df.reset_index()['Close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    return df1, scaler


def SplittingDataSet(df1, scaler):
    train_size = int(len(df1) * 0.80)
    train_data, test_data = df1[0:train_size, :], df1[train_size:len(df1), :1]
    time_step = 100
    X_train, Y_train = create_dataset(df1, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    with st.spinner("Carregando..."):
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64, verbose=1)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    st.success("Completo!")
    return train_predict, test_predict, model, df1, test_data, X_test


def newGraph(model, df1, scaler, test_data, X_test):
    x_input = test_data[X_test.shape[0] + 1:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    lst_output = []
    n_steps = 100
    i = 0
    mainVal = 0
    while (i < 15):
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:])
            print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i, yhat))
            inverse_data = scaler.inverse_transform(yhat)
            if (i == 14):
                mainVal = inverse_data
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            inverse_data = scaler.inverse_transform(yhat)
            if (i == 14): 
                mainVal = inverse_data
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i += 1

    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 116) 
    
    start_date = datetime.now() - timedelta(weeks=10)
    end_date = datetime.now() + timedelta(days=15)

    total_days = (end_date - start_date).days
    interval_days = total_days // 3 

    equidistant_dates = [start_date + timedelta(days=i*interval_days) for i in range(4)]

    formatted_dates = [date.strftime('%d-%m-%Y') for date in equidistant_dates]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 116) 
    ax.plot(day_new, scaler.inverse_transform(df1[len(df1) - 100:]), label='Dados Originais', linewidth=5)
    ax.plot(day_pred, scaler.inverse_transform(lst_output), label='Predições', linewidth=5)

    x_positions = np.linspace(0, 114, 4)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(formatted_dates, rotation=45)

    ax.set_facecolor("black")
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Value', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.legend()

    col1, col2 = st.columns(2)
    col1.title("Previsão:")
    col1.write(
        f"<span style=' font-size: 20px;'>O preço aproximado será de: R$</span> <span style='color: green; font-size: 30px; font-weight: bold; font-style: italic;'>{mainVal[0][0]:.2f}</span>",
        unsafe_allow_html=True)
    col2.pyplot(fig)


def filter_companies(search_term, company_dict):
    filtered_companies = [code for code in company_dict.keys() if
                          search_term.upper() in code or search_term.upper() in company_dict[code].upper()]
    return filtered_companies


def main():
    st.set_page_config(
        page_title="Predição de Valores de Ativos",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )
    company_codes = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'JPM', 'WMT', 'NVDA', 'FB', 'V', 'MA', 'BAC', 'RIL',
                     'TCS', 'TMUS', 'JNJ', 'PG', 'BABA', 'NFLX', 'VZ', 'INTC', 'DIS', 'CSCO', 'PFE', 'HD', 'KO', 'PEP',
                     'NKE', 'ADBE', 'MCD', 'PYPL', 'ABT', 'CRM', 'ORCL', 'NVO', 'CVX', 'XOM', 'CMCSA', 'ASML', 'TM',
                     'ABBV', 'NVS', 'AMGN', 'COST', 'AVGO', 'TMO', 'MRK', 'UNH', 'LIN', 'BHP', 'SBUX', 'BMY', 'DHR',
                     'HDB', 'QCOM', 'TXN', 'NEE', 'ACN', 'LLY', 'LMT', 'NOW', 'LOW', 'AMT', 'NOC', 'SNE', 'UNP', 'UPS',
                     'CHTR', 'RTX', 'PDD', 'RIL', 'TCS', 'HDB', 'HINDUNILVR', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN',
                     'AXISBANK', 'KOTAKBANK', 'ITC', 'LT', 'M&M', 'BHARTIARTL', 'HCLTECH', 'POWERGRID', 'BAJAJFINSV',
                     'BAJFINANCE', 'HINDALCO', 'GRASIM', 'JSWSTEEL', 'INDUSINDBK', 'ADANIPORTS', 'DRREDDY',
                     'BAJAJ-AUTO', 'ULTRACEMCO', 'UPL', 'TATAMOTORS', 'HDFCLIFE', 'HEROMOTOCO', 'TECHM', 'ONGC', 'IOC',
                     'POWERINDIA', 'BPCL', 'SIEMENS', 'HAVELLS', 'GAIL', 'MRF', 'ADANIGREEN', 'AUROPHARMA', 'TVSMOTOR',
                     'JUBLFOOD', 'SBILIFE', 'CHOLAFIN', 'NAM-INDIA', 'DLF', 'JINDALSTEL', 'LUPIN', 'NMDC', 'SRF',
                     'VOLTAS', 'PNB', 'SUNTV', 'PIDILITIND', 'MOTHERSUMI', 'BANKBARODA', 'INDIGO', 'AMBUJACEM',
                     'LALPATHLAB', 'IRCTC', 'BERGEPAINT', 'CADILAHC', 'COLPAL', 'ICICIGI', 'APOLLOHOSP', 'CHOLAHLDNG',
                     'MCDOWELL-N', 'BEL']
    company_names = [
        'Apple Inc.', 'Microsoft Corporation', 'Amazon.com Inc.', 'Alphabet Inc. (Google)', 'Tesla Inc.',
        'JPMorgan Chase & Co.',
        'Walmart Inc.', 'NVIDIA Corporation', 'Facebook Inc.', 'Visa Inc.', 'Mastercard Incorporated',
        'Bank of America Corporation',
        'Reliance Industries Limited', 'Tata Consultancy Services Limited', 'T-Mobile US Inc.', 'Johnson & Johnson',
        'Procter & Gamble Company',
        'Alibaba Group Holding Limited', 'Netflix Inc.', 'Verizon Communications Inc.', 'Intel Corporation',
        'The Walt Disney Company',
        'Cisco Systems Inc.', 'Pfizer Inc.', 'The Home Depot Inc.', 'The Coca-Cola Company', 'PepsiCo Inc.',
        'Nike Inc.', 'Adobe Inc.',
        'McDonald\'s Corporation', 'PayPal Holdings Inc.', 'Abbott Laboratories', 'Salesforce.com Inc.',
        'Oracle Corporation',
        'Novo Nordisk A/S', 'Chevron Corporation', 'Exxon Mobil Corporation', 'Comcast Corporation',
        'ASML Holding N.V.', 'Toyota Motor Corporation',
        'AbbVie Inc.', 'Novartis AG', 'Amgen Inc.', 'Costco Wholesale Corporation', 'Broadcom Inc.',
        'Thermo Fisher Scientific Inc.',
        'Merck & Co. Inc.', 'UnitedHealth Group Incorporated', 'Linde plc', 'BHP Group', 'Starbucks Corporation',
        'Bristol-Myers Squibb Company',
        'Danaher Corporation', 'HDFC Bank Limited', 'Qualcomm Incorporated', 'Texas Instruments Incorporated',
        'NextEra Energy Inc.',
        'Accenture plc', 'Eli Lilly and Company', 'Lockheed Martin Corporation', 'ServiceNow Inc.',
        'Lowe\'s Companies Inc.', 'American Tower Corporation',
        'Northrop Grumman Corporation', 'Sony Corporation', 'Union Pacific Corporation', 'United Parcel Service Inc.',
        'Charter Communications Inc.',
        'Raytheon Technologies Corporation', 'Pinduoduo Inc.', 'Reliance Industries Limited',
        'Tata Consultancy Services Limited', 'HDFC Bank Limited',
        'Hindustan Unilever Limited', 'Infosys Limited', 'Housing Development Finance Corporation Limited',
        'ICICI Bank Limited',
        'State Bank of India', 'Axis Bank Limited', 'Kotak Mahindra Bank Limited', 'ITC Limited',
        'Larsen & Toubro Limited', 'Mahindra & Mahindra Limited',
        'Bharti Airtel Limited', 'HCL Technologies Limited', 'Power Grid Corporation of India Limited',
        'Bajaj Finserv Limited', 'Bajaj Finance Limited',
        'Hindalco Industries Limited', 'Grasim Industries Limited', 'JSW Steel Limited', 'IndusInd Bank Limited',
        'Adani Ports and Special Economic Zone Limited',
        'Dr. Reddy\'s Laboratories Limited', 'Bajaj Auto Limited', 'UltraTech Cement Limited', 'UPL Limited',
        'Tata Motors Limited', 'HDFC Life Insurance Company Limited',
        'Hero MotoCorp Limited', 'Tech Mahindra Limited', 'Oil and Natural Gas Corporation Limited',
        'Indian Oil Corporation Limited', 'Power Grid Corporation of India Limited',
        'Bharat Petroleum Corporation Limited', 'Siemens Limited', 'Havells India Limited', 'GAIL (India) Limited',
        'MRF Limited', 'Adani Green Energy Limited',
        'Aurobindo Pharma Limited', 'TVS Motor Company Limited', 'Jubilant Foodworks Limited',
        'SBI Life Insurance Company Limited', 'Cholamandalam Investment and Finance Company Limited',
        'Nippon Life India Asset Management Limited', 'DLF Limited', 'Jindal Steel & Power Limited', 'Lupin Limited',
        'NMDC Limited', 'SRF Limited',
        'Voltas Limited', 'Punjab National Bank', 'Sun TV Network Limited', 'Pidilite Industries Limited',
        'Motherson Sumi Systems Limited', 'Bank of Baroda',
        'InterGlobe Aviation Limited', 'Ambuja Cements Limited', 'Dr. Lal PathLabs Limited',
        'Indian Railway Catering and Tourism Corporation Limited',
        'Berger Paints India Limited', 'Cadila Healthcare Limited', 'Colgate-Palmolive (India) Limited',
        'ICICI Lombard General Insurance Company Limited',
        'Apollo Hospitals Enterprise Limited', 'Cholamandalam Financial Holdings Limited', 'United Spirits Limited',
        'Bharat Electronics Limited'
    ]

    company_dict = dict(zip(company_codes, company_names))
    st.title("Previsão de Ativos em Tempo Real")

    company = st.selectbox("Selecione a Empresa", company_dict)
    with st.form("my-form"):
        color = st.select_slider(
            'Selecione as semanas de treinamento',
            options=['5', '6', '7', '8', '9', '10'])
        submitted = st.form_submit_button("Treinar")
        if submitted:
            col1, col2, col3, col4 = st.columns(4)
            col1.title(company_dict[company])
            col2.title("")
            stock = yf.Ticker(company)
            hist = stock.history(period=(color + 'y'))

            df = hist.copy()

            df.fillna(method='ffill', inplace=True)

            df1, scaler = DataPreProcessing(df)
            train_predict, test_predict, model, df1, test_data, X_test = SplittingDataSet(df1, scaler)

            newGraph(model, df1, scaler, test_data, X_test)


if __name__ == "__main__":
    main()
