import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

from scipy import stats
from statistics import mode

import warnings
warnings.filterwarnings('ignore')

import pandas_datareader.data as web
import datetime
from sklearn.model_selection import train_test_split

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

import mplfinance as mpf

import statsmodels.api as sm

import plotly.subplots as ms
import plotly.graph_objects as go

def captura_datos (activo,start,period):
    tipo = 0
    start = datetime.datetime.strptime(start,"%Y, %m, %d" )
    end = datetime.datetime.now()
    df = web.get_data_yahoo(activo, start, end, interval=period)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df["timeIndex"] = pd.Series(np.arange(len(df['Close'])), index=df.index)
    #display(df.head(10))
    return df

def graficar_datos(df):
    # genero los gráficos con la serie de tiempo, la estacionalidad y el componente aleatorio
    plt.figure()
    plt.rcParams.update({'figure.figsize': (15,10)})
    y=pd.DataFrame(df.Close)
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',freq=12)
    decomposition.plot()
    plt.show()
    
    # grafico la serie de tiempo, medias moviles y volumen
    #mpf.plot(df,type='candle',mav=(3,6,9),volume=True,style='yahoo',figratio=(11,8),figscale=2)
    
    # Dibuja gráfico interactivo con plotly
    #Make Subplot of 2 rows to plot 2 graphs sharing the x axis
    fig1 = ms.make_subplots(rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0)

    #Add Candlstick Chart to Row 1 of subplot
    fig1.add_trace(go.Scatter(x=df.index, 
                         y=df['MA5'], 
                         opacity=0.7, 
                         line=dict(color='blue', width=1), 
                         name='MA 5'))
    fig1.add_trace(go.Scatter(x=df.index, 
                         y=df['MA20'], 
                         opacity=0.7, 
                         line=dict(color='orange', width=1), 
                         name='MA 20'))
    fig1.add_trace(go.Candlestick(x = df.index,
    low = df["Low"],
    high = df["High"],
    close = df["Close"],
    open = df["Open"],
    increasing_line_color = "green",
    decreasing_line_color = "red"),
    row=1,
    col=1)

    #Add Volume Chart to Row 2 of subplot
    colors = ['red' if row['Open'] - row['Close'] >= 0 
          else 'green' for index, row in df.iterrows()]
    fig1.add_trace(go.Bar(x=df.index, 
                     y=df['Volume'],
                     marker_color=colors
                    ), row=2, col=1)


    #Update Price Figure layout
    fig1.update_layout(title = "Interactive CandleStick & Volume Chart",
    yaxis1_title = "Stock Price ($)",
    yaxis2_title = "Volume (M)",
    xaxis2_title = "Time",width=910, height=600)
    fig1['layout']['yaxis1'].update(domain=[0.2,1])
    fig1['layout']['yaxis2'].update(domain=[0,0.2])

    #Agregar botones
    fig1.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
    buttons=list([
    dict(count=3, label="trim", step="month", stepmode="backward"),
    dict(count=6, label="semes", step="month", stepmode="backward"),
    dict(count=1, label="1 year", step="year", stepmode="backward"),
    dict(count=5, label="5 years", step="year", stepmode="backward"),
    dict(step="all")
    ])
    ),
    row=1, col=1)
    fig1.show()
    
def RMSE(predicted, actual):
    mse = (predicted - actual) ** 2
    rmse = np.sqrt(mse.sum() / mse.count())
    return rmse

def seasonality(df):
    df['month'] = [d.strftime('%b') for d in df.index]
    dummies_mes = pd.get_dummies(df['month'],drop_first=True)
    df = df.join(dummies_mes)
    string = 'Aug + Dec + Feb + Jan + Jul + Jun + Mar + May + Nov + Oct + Sep '
    return (df,string)

def eleccion_modelo(model,df,estacionalidad):
    if model == 'linear':
        if estacionalidad:
            df, est = seasonality(df)
            string = 'Close ~ timeIndex + Volume' + ' + ' + est
            return(df,string)
        
        string = 'Close ~ timeIndex + Volume'
        return (df,string)
    
    elif model == 'quad':
        df["timeIndex_sq"] = df["timeIndex"]**2
        if estacionalidad:
            df, est = seasonality(df)
            string = 'Close ~ timeIndex + timeIndex_sq + Volume' + ' + ' + est
            return(df,string)
        
        string = 'Close ~ timeIndex + timeIndex_sq + Volume'
        return(df,string)
    
    elif model == 'log':
        df['log_value'] = np.log(df['Close'])
        if estacionalidad:
            df, est = seasonality(df)
            string = 'log_value ~ timeIndex + Volume' + ' + ' + est
            return(df,string)
        
        string = 'log_value ~ timeIndex + Volume'
        return(df,string)
    
def evaluacion (df,pred_size,lags,model,estacionalidad):
    tipo = 0
    #start = datetime.datetime.strptime(start,"%Y, %m, %d" )
    #end = datetime.datetime.now()
    #df = web.get_data_yahoo(activo, start, end, interval="m")
    df = df[["Close","Volume"]]
    df["timeIndex"] = pd.Series(np.arange(len(df['Close'])), index=df.index)
    
    df, string = eleccion_modelo(model=model,df=df,estacionalidad=estacionalidad)
    
    df_train, df_test = train_test_split(df, test_size=pred_size, random_state=42, shuffle=False)
    
    modelo = smf.ols(string, data = df_train).fit()
    
    if estacionalidad:
        if model == 'quad':
            
            df_train['model'] = modelo.predict(df_train[["timeIndex","timeIndex_sq","Volume",\
                                        "Aug", "Dec", "Feb", "Jan","Jul", "Jun", "Mar", "May",\
                                                "Nov", "Oct", "Sep"]])
        
            df_test['model'] = modelo.predict(df_test[["timeIndex","timeIndex_sq", "Volume",\
                                            "Aug", "Dec", "Feb", "Jan","Jul", "Jun", "Mar", "May",\
                                                "Nov", "Oct", "Sep"]])
        else:   
            df_train['model'] = modelo.predict(df_train[["timeIndex", "Volume",\
                                            "Aug", "Dec", "Feb", "Jan","Jul", "Jun", "Mar", "May",\
                                                "Nov", "Oct", "Sep"]])
        
            df_test['model'] = modelo.predict(df_test[["timeIndex", "Volume",\
                                            "Aug", "Dec", "Feb", "Jan","Jul", "Jun", "Mar", "May",\
                                                "Nov", "Oct", "Sep"]])
    else:
        if model == 'quad':
            df_train['model'] = modelo.predict(df_train[["timeIndex", "timeIndex_sq","Volume"]])
            df_test['model'] = modelo.predict(df_test[["timeIndex", "timeIndex_sq","Volume"]])
        else:
            df_train['model'] = modelo.predict(df_train[["timeIndex", "Volume"]])
            df_test['model'] = modelo.predict(df_test[["timeIndex", "Volume"]])

    
    
    if model == 'log':
        df_train['back_model_log_est'] = np.exp(df_train['model'])
        df_test['back_model_log_est'] = np.exp(df_test['model'])
        res_model = df_train['Close'] - df_train['back_model_log_est']
        res_log_est = df_train['log_value'] - df_train['model']
        result = adfuller(res_model)
        result1 = adfuller(res_log_est)
    else:
        res_model = df_train['Close'] - df_train['model']
        result = adfuller(res_model)
        result1 = result
        
    if result[1]<0.05:
        print("elegimos el residuo del precio de cierre - exp(log est) - p=",result[1])
        modelo = res_model
        tsplot (res_model,lags=lags)
        tipo = 1
    elif result1[1]<0.05:
        print("elegimos el residuo del log precio de cierre - log est - p=",result1[1])
        modelo = res_log_est
        tsplot (res_log_est,lags=lags)
        tipo = 2
    else:
        return print("la serie no es estacionaria, hacer mas diferenciación")
    return (modelo,df_train,df_test,tipo,model)

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """ 
        Plotea la serie de tiempo, el ACF y PACF y el test de Dickey–Fuller
        
        y - serie de tiempo
        lags - cuántos lags incluir para el cálculo de la ACF y PACF
        
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        
        # definimos ejes
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        
        # obtengo el p-value con h0: raiz unitaria presente
        p_value = sm.tsa.stattools.adfuller(y)[1]
        
        ts_ax.set_title('Análisis de la Serie de Tiempo\n Dickey-Fuller: p={0:.5f}'\
                        .format(p_value))
        
        # plot de autocorrelacion
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        # plot de autocorrelacion parcial
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
def arima (modelo,p,d,q,alpha):
    model_ARIMA = ARIMA(modelo[0], order=(p,d,q))
    results_ARIMA = model_ARIMA.fit()
    res_ARIMA =  results_ARIMA.fittedvalues - modelo[0]
    predictions_ARIMA, se, conf = results_ARIMA.forecast(len(modelo[2]['Close']), alpha=alpha)
    if modelo[3] == 1 and modelo[4] == 'log':
        modelo[1]['model_ARIMA'] = modelo[1]['back_model_log_est'] + results_ARIMA.fittedvalues
        modelo[2]['model_ARIMA'] = modelo[2]['back_model_log_est'] + predictions_ARIMA
        
    elif modelo[3] == 2 and modelo[4] == 'log':
        modelo[1]['model_ARIMA'] = np.exp(modelo[1]['model'] + results_ARIMA.fittedvalues)
        modelo[2]['model_ARIMA'] = np.exp(modelo[2]['model'] + predictions_ARIMA)
        
    else:
        modelo[1]['model_ARIMA'] = modelo[1]['model'] + results_ARIMA.fittedvalues
        modelo[2]['model_ARIMA'] = modelo[2]['model'] + predictions_ARIMA
    modelo[1].plot(kind = "line", y = ['Close', 'model_ARIMA']);
    modelo[2].plot(kind = "line", y = ['Close', 'model_ARIMA']);
    