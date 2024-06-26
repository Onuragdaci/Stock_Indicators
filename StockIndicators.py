from scipy.signal import argrelextrema
from scipy import stats
import numpy as np
import pandas as pd
from tradingview_screener import get_all_symbols
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()

#Stocks for BIST or BINANCE
def Stocks(name):
    Stock_names = ''
    if name == 'BIST':
        Stock_names = get_all_symbols(market='turkey')
        Stock_names = [symbol.replace('BIST:', '') for symbol in Stock_names]
        Stock_names = sorted(Stock_names)
    if name == 'BINANCE':
        Stock_names = get_all_symbols(market='crypto')
        Stock_names = [symbol.replace('BINANCE:', '') for symbol in Stock_names if symbol.startswith('BINANCE:') and symbol.endswith('USDT')]
        Stock_names = sorted(Stock_names)
    return Stock_names

#Download Stocks from #BIST or BINANCE
def TVGet(name,exchange,interval, nbars=100):
    interval_mapping = {
        '1m': Interval.in_1_minute,
        '3m': Interval.in_3_minute,
        '5m': Interval.in_5_minute,
        '15m': Interval.in_15_minute,
        '30m': Interval.in_30_minute,
        '45m': Interval.in_45_minute,
        '1h': Interval.in_1_hour,
        '2h': Interval.in_2_hour,
        '3h': Interval.in_3_hour,
        '4h': Interval.in_4_hour,
        '1D': Interval.in_daily,
        '1W': Interval.in_weekly,
        '1M': Interval.in_monthly,
    }

    if interval in interval_mapping:
        mapped_interval = interval_mapping[interval]
        retries = 3  # Number of retries
        while retries > 0:
            try:
                data = tv.get_hist(symbol=name, exchange=exchange, interval=mapped_interval, n_bars=nbars)
                data = data.reset_index()
                return data
            except Exception as e:
                retries -= 1
                print(f"An error occurred: {e}. Retrying {retries} more times.")
        raise ValueError("Failed to retrieve data after multiple attempts.")
    else:
        raise ValueError("Invalid interval provided.")

#Return Series
def sma(series, length):
    """
    Calculate the Simple Moving Average (SMA) for a given series.
    """
    return series.rolling(window=length).mean()

def ema(series, length):
    """
    Calculate the Exponential Moving Average (EMA) for a given series.
    """
    return series.ewm(span=length, adjust=False).mean()

def smma(values, period):
    """
    Calculates the Smoothed Moving Average (SMMA).

    Args:
    values (pd.Series or list): The input values for which the SMMA is to be calculated.
    period (int): The period over which the SMMA is calculated.

    Returns:
    pd.Series: The SMMA values.
    """
    smma = [np.nan] * len(values)
    smma[period - 1] = np.mean(values[:period])
    
    for i in range(period, len(values)):
        smma[i] = (smma[i - 1] * (period - 1) + values[i]) / period

    smma = pd.Series(smma, index=values.index if isinstance(values, pd.Series) else None)    
    return smma

def rma(series, length=None):
    """
    Calculates the Relative Moving Average (RMA) of a given close price series.

    Parameters:
    - series: pandas Series containing price data.
    - length (int): The number of periods to consider. Default is 10.
    - offset (int): The offset from the current period. Default is None.

    Returns:
    - pandas.Series: The Relative Moving Average (RMA) values.
    """
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    alpha = (1.0 / length) if length > 0 else 0.5

    # Calculate Result
    rma = series.ewm(alpha=alpha, min_periods=length).mean()
    return rma

def kama(series, length=21, fast_end=0.666, slow_end=0.0645, offset=None):
    """
    Calculates the Kaufman Adaptive Moving Average (KAMA) of a given price series.

    Parameters:
    - series: pandas Series containing price data.
    - length (int): The number of periods to consider for the efficiency ratio. Default is 21.
    - fast_end (float): The smoothing constant for the fastest EMA. Default is 0.666.
    - slow_end (float): The smoothing constant for the slowest EMA. Default is 0.0645.
    - offset (int): The offset from the current period. Default is None.

    Returns:
    - pandas.Series: The Kaufman Adaptive Moving Average (KAMA) values.
    """
    # Validate Arguments
    length = int(length) if length and length > 0 else 21
    fast_end = float(fast_end) if fast_end else 0.666
    slow_end = float(slow_end) if slow_end else 0.0645
    offset = int(offset) if offset else 0

    # Calculate Efficiency Ratio (ER)
    price_diff = series.diff(1).abs()
    signal = series.diff(length).abs()
    noise = price_diff.rolling(window=length).sum()
    er = signal / noise
    er.replace([np.inf, -np.inf], 0, inplace=True)  # Handle division by zero

    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_end - slow_end) + slow_end) ** 2

    # Calculate KAMA
    kama = pd.Series(np.zeros(len(series)), index=series.index)
    kama.iloc[length - 1] = series.iloc[length - 1]  # Set initial value

    for i in range(length, len(series)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i - 1])

    # Apply offset if needed
    if offset != 0:
        kama = kama.shift(offset)

    return kama

def alma(series, window=20, sigma=6, offset=0.85):
    """
    Calculate the Arnaud Legoux Moving Average (ALMA) for a given series.

    :param series: pandas Series of prices.
    :param window: int, window length for the moving average.
    :param sigma: float, standard deviation for the Gaussian distribution.
    :param offset: float, offset for the Gaussian distribution.
    :return: pandas Series with the ALMA values.
    """
    m = (window - 1) * offset
    s = window / sigma

    def gaussian_weight(x, m, s):
        return np.exp(-((x - m) ** 2) / (2 * s ** 2))

    weights = np.array([gaussian_weight(i, m, s) for i in range(window)])
    weights /= np.sum(weights)

    alma = series.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)
    return alma

def salma(series, length=10, smooth=3, mult=0.3, sd_len=5):
    """
    Calculate the Smoothed Adaptive Linear Moving Average (SALMA) for a given series.

    :param series: pandas Series of prices.
    :param length: int, length of the baseline and upper/lower bands.
    :param smooth: int, smoothing parameter for SALMA.
    :param mult: float, multiplier for the standard deviation.
    :param sd_len: int, length of the standard deviation calculation.
    :return: pandas Series with the SALMA values.
    """
    baseline = wma(series, sd_len)
    dev = mult * stdev(series, sd_len)
    upper = baseline + dev
    lower = baseline - dev
    cprice = np.where(series > upper, upper, np.where(series < lower, lower, series))
    cprice = pd.Series(cprice)
    salma = wma(wma(cprice, length), smooth)
    return salma

def wma(series, length):
    """
    Calculate the Weighted Moving Average (WMA) for a given series.

    :param series: pandas Series of prices.
    :param length: int, length of the moving average.
    :return: pandas Series with the WMA values.
    """
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def hull_ma(series, length=9):
    """
    Calculate the Hull Moving Average (HMA) for a given series.
    
    The Hull Moving Average is a fast and smooth moving average that 
    reduces lag while improving smoothness.
    
    Parameters:
    - series: pandas.Series of prices
    - length: integer, the period for HMA calculation
    
    Returns:
    - pandas.Series containing the HMA values
    """
    
    half_length = int(length / 2)
    wma_half = wma(series, half_length)
    wma_full = wma(series, length)
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, int(np.sqrt(length)))
    
    return hma

def xsa(src, length, weight):
    """
    Calculate the XSA (X Super Smoother) for a given series.

    :param src: numpy array or pandas Series of prices.
    :param length: int, length of the moving average.
    :param weight: float, weight for the XSA calculation.
    :return: numpy array with the XSA values.
    """
    sumf = np.zeros_like(src)
    ma = np.zeros_like(src)
    xsa = np.zeros_like(src)

    for i in range(length, len(src)):
        sumf[i] = np.nan_to_num(sumf[i - 1]) - np.nan_to_num(src[i - length]) + src[i]
        ma[i] = np.nan if np.isnan(src[i - length]) else sumf[i] / length
        xsa[i] = ma[i] if np.isnan(xsa[i - 1]) else (src[i] * weight + xsa[i - 1] * (length - weight)) / length

    return xsa

def relative_volume(series, length=10, offset=None):
    """
    Calculates the Relative Volume (RV) based on a given volume series.

    Parameters:
    - volume_series: pandas Series containing volume data.
    - length (int): The number of periods to consider. Default is 10.
    - offset (int): The offset from the current period. Default is None.

    Returns:
    - pandas.Series: The Relative Volume (RV) values.
    """
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    alpha = (1.0 / length) if length > 0 else 0.5

    # Calculate Result
    rv = series.ewm(alpha=alpha, min_periods=length).mean()
    rv = series / (rv+0.0001)
    return rv

def vwap(high, low, close, volume):
    """
    Calculates the Volume Weighted Average Price (VWAP).

    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Closing prices.
    volume (pd.Series): Volume data.
    length (int): Window size for calculating VWAP (default is 20).

    Returns:
    pd.Series: VWAP values.
    """
    length=1
    tp = (high + low + close) / 3
    vwap = (tp * volume).rolling(window=length).sum() / volume.rolling(window=length).sum()
    return vwap

def hlc3(high, low, close):
    """
    Calculate the HLC (High-Low-Close) for the given high, low, and close series.

    :param high: pandas Series or numpy array of high prices.
    :param low: pandas Series or numpy array of low prices.
    :param close: pandas Series or numpy array of close prices.
    :return: pandas Series or numpy array with the HLC3 values.
    """
    hlc3 = (high + low + close) / 3
    return hlc3

def mfi(high, low, close, volume, window=14):
    """
    Calculates the Money Flow Index (MFI).

    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Closing prices.
    volume (pd.Series): Volume data.
    window (int): Window size for calculating MFI (default is 14).

    Returns:
    pd.Series: Money Flow Index values.
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_money_flow = (money_flow * (close > close.shift(1))).rolling(window=window).sum()
    negative_money_flow = (money_flow * (close < close.shift(1))).rolling(window=window).sum()
    money_ratio = positive_money_flow / negative_money_flow
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi

def cmf(high, low, close, volume, length=20):
    """
    Calculates the Chaikin Money Flow (CMF).

    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Closing prices.
    volume (pd.Series): Volume data.
    window (int): Window size for calculating CMF (default is 20).

    Returns:
    pd.Series: CMF values.
    """
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    money_flow_volume = money_flow_multiplier * volume
    cmf = money_flow_volume.rolling(window=length).sum() / volume.rolling(window=length).sum()
    return cmf

def williams_r(high, low, close, window=14):
    """
    Calculates the Williams %R (W.R) indicator.

    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Closing prices.
    window (int): Window size for calculating W.R (default is 14).

    Returns:
    pd.Series: Williams %R values.
    """
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r

def momentum(close, window=14):
    """
    Calculates the Momentum indicator.

    Args:
    close (pd.Series): Closing prices.
    window (int): Window size for calculating momentum (default is 14).

    Returns:
    pd.Series: Momentum values.
    """
    momentum = close.diff(window)
    return momentum

def obv(close, volume):
    """
    Calculates the On-Balance Volume (OBV) indicator.

    Args:
    close (pd.Series): Closing prices.
    volume (pd.Series): Volume data.

    Returns:
    pd.Series: OBV values.
    """
    obv = pd.Series(index=close.index)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    
    return obv

def rsi(series, length=14):
    """
    Calculate the Relative Strength Index (RSI) for a given series.

    Parameters:
    - series: pandas Series containing price data.
    - length: Length of the RSI period (default is 14).
    - scalar: Scalar factor to adjust RSI values (default is 100).
    - drift: Number of periods for price changes (default is 1).

    Returns:
    - pandas Series containing RSI values calculated based on the input parameters.
    """
    # Calculate price changes
    scalar=100 
    drift=1
    negative = series.diff(drift)
    positive = negative.copy()

    # Make negatives 0 for the positive series
    positive[positive < 0] = 0
    # Make positives 0 for the negative series
    negative[negative > 0] = 0

    # Calculate average gains and losses
    positive_avg = rma(positive, length=length)
    negative_avg = rma(negative, length=length)

    # Calculate RSI
    rsi = scalar * positive_avg / (positive_avg + negative_avg.abs())
    return rsi

def cci(high, low, close, window=20, constant=0.015):
    """
    Calculates the Commodity Channel Index (CCI).

    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Closing prices.
    window (int): Window size for calculating CCI (default is 20).
    constant (float): Constant multiplier (default is 0.015).

    Returns:
    pd.Series: CCI values.
    """
    typical_price = (high + low + close) / 3
    mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (typical_price - typical_price.rolling(window=window).mean()) / (constant * mean_deviation)
    return cci

def psar(high, low, close, initial_af=0.02, step_af=0.02, max_af=0.2):
    """
    Calculates the Parabolic SAR (PSAR).

    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Close prices.
    initial_af (float, optional): The initial acceleration factor. Default is 0.02.
    step_af (float, optional): The step acceleration factor. Default is 0.02.
    max_af (float, optional): The maximum acceleration factor. Default is 0.2.

    Returns:
    pd.Series: The PSAR values.
    """
    psar = pd.Series(index=high.index)
    psar[0] = close[0]
    uptrend = True
    af = initial_af
    ep = high[0]
    for i in range(1, len(high)):
        if uptrend:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if low[i] < psar[i]:
                uptrend = False
                psar[i] = ep
                af = initial_af
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step_af, max_af)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if high[i] > psar[i]:
                uptrend = True
                psar[i] = ep
                af = initial_af
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step_af, max_af)
    return psar

def lpsar(high, low, close, initial_af=0.02, step_af=0.02, max_af=0.2):
    """
    Calculates the Lucid Parabolic SAR (Lucid PSAR).

    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Close prices.
    initial_af (float, optional): The initial acceleration factor. Default is 0.02.
    step_af (float, optional): The step acceleration factor. Default is 0.02.
    max_af (float, optional): The maximum acceleration factor. Default is 0.2.

    Returns:
    pd.Series: The Lucid PSAR values.
    """
    lpsar = pd.Series(index=high.index)
    lpsar[0] = close[0]
    uptrend = True
    af = initial_af
    ep = high[0]

    for i in range(1, len(high)):
        previous_lpsar = lpsar[i - 1]
        previous_high = high[i - 1]
        previous_low = low[i - 1]

        if uptrend:
            lpsar[i] = previous_lpsar + af * (ep - previous_lpsar)
            if low[i] < lpsar[i]:
                uptrend = False
                lpsar[i] = ep
                af = initial_af
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step_af, max_af)
                lpsar[i] = min(lpsar[i], previous_low, low[i])
        else:
            lpsar[i] = previous_lpsar + af * (ep - previous_lpsar)
            if high[i] > lpsar[i]:
                uptrend = True
                lpsar[i] = ep
                af = initial_af
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step_af, max_af)
                lpsar[i] = max(lpsar[i], previous_high, high[i])

    return lpsar

def tr(high, low, close):
    """
    Calculates the True Range (TR).

    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Closing prices.

    Returns:
    pd.Series: True Range values.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(high, low, close, period=14):
    """
    Calculates the Average True Range (ATR) using high, low, and close prices.

    Args:
    high (pd.Series or list): The high prices.
    low (pd.Series or list): The low prices.
    close (pd.Series or list): The close prices.
    period (int, optional): The period over which the ATR is calculated. Default is 14.

    Returns:
    pd.Series: The Average True Range (ATR) values.
    """
    # Calculate true range (TR)
    true_range = tr(high, low, close)
    atr = rma(true_range,period)
    return atr

def stdev(series, length):
    """
    Calculates the rolling standard deviation of a series.

    Args:
    series (pd.Series): The input series for which the rolling standard deviation is calculated.
    length (int): The window length for the rolling standard deviation calculation.

    Returns:
    pd.Series: The rolling standard deviation values.
    """
    deviation = series.rolling(window=length).std()
    return deviation

def ao(high, low, fast=5, slow=34):
    """
    Calculate the Awesome Oscillator (AO) using the High and Low prices and specified lengths for SMAs.
    
    :param high: pandas Series of high prices
    :param low: pandas Series of low prices
    :param fast: short period for SMA calculation (default: 5)
    :param slow: long period for SMA calculation (default: 34)
    :return: pandas Series with AO values
    """
    midpoints = (high + low) / 2
    fastsma = sma(midpoints, fast)
    slowsma = sma(midpoints, slow)
    ao = fastsma - slowsma
    return ao

def mfi(high, low, close, volume, period=14):
    """
    Calculates the Money Flow Index (MFI).

    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    close (pd.Series): Close prices.
    volume (pd.Series): Volume.
    period (int, optional): The period over which to calculate MFI. Default is 14.

    Returns:
    pd.Series: The Money Flow Index values.
    """
    typical_price = hlc3(high, low, close)
    raw_money_flow = typical_price * volume
    positive_flow = (raw_money_flow * (typical_price > typical_price.shift(1))).rolling(window=period).sum()
    negative_flow = (raw_money_flow * (typical_price < typical_price.shift(1))).rolling(window=period).sum()
    money_flow_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_flow_ratio))
    return mfi

def ewo(series, short_period=5, long_period=34):
    """
    Calculate Elliott Wave Oscillator (EWO)
    EWO = EMA(close, short_period) - EMA(close, long_period)
    
    :param data: pandas DataFrame with a 'close' column
    :param short_period: short period for EMA calculation (default: 5)
    :param long_period: long period for EMA calculation (default: 35)
    :return: pandas Series with EWO values
    """
    short_ema = ema(series, short_period)
    long_ema = ema(series, long_period)
    ewo = short_ema - long_ema
    return ewo

def dss_bresser_scalper(high, low, close, ema_period=8, stoc_period=13):
    """
    DSS Bresser Scalper Improved Indicator Calculation

    Parameters:
    high (pd.Series): High prices
    low (pd.Series): Low prices
    close (pd.Series): Close prices
    ema_period (int): EMA period for smoothing (default: 8)
    stoc_period (int): Stochastic period for DSS calculation (default: 13)

    Returns:
    pd.Series: DSS Bresser Scalper values
    """
    smooth_coeff = 2.6 / (1 + ema_period)
    
    highrange = high.rolling(window=stoc_period).max()
    lowrange = low.rolling(window=stoc_period).min()
    delta = close - lowrange
    rrange = highrange - lowrange

    MIT = delta / rrange * 100
    mitbuffer = MIT.ewm(alpha=smooth_coeff).mean()

    highrange_ = mitbuffer.rolling(window=stoc_period).max().fillna(0)
    lowrange_ = mitbuffer.rolling(window=stoc_period).min().fillna(stoc_period * 1000)
    delta_ = mitbuffer - lowrange_
    DSS = delta_ / (highrange_ - lowrange_) * 100
    dssbuffer = DSS.ewm(alpha=smooth_coeff).mean()
    
    return dssbuffer

def bcwsma(series, length, multiplier):
    """
    Calculate the smoothed moving average (BCWSMA).

    Args:
    series (pd.Series): The input series to be smoothed.
    length (int): The length of the smoothing period.
    multiplier (int): The multiplier used in the smoothing formula.

    Returns:
    pd.Series: The smoothed moving average series.
    """
    bcwsma_series = pd.Series(index=series.index, dtype=float)
    first_valid_index = series.first_valid_index()
    
    if first_valid_index is not None:
        bcwsma_series.iloc[first_valid_index] = series.iloc[first_valid_index]
        for i in range(first_valid_index + 1, len(series)):
            if pd.isna(series.iloc[i]):
                bcwsma_series.iloc[i] = bcwsma_series.iloc[i-1]
            else:
                bcwsma_series.iloc[i] = (multiplier * series.iloc[i] + (length - multiplier) * bcwsma_series.iloc[i-1]) / length
    return bcwsma_series

def kdj_indicator(high, low, close, ilong=9, isig=3):
    """
    Calculate the KDJ indicator.

    Args:
    high (pd.Series): The high prices series.
    low (pd.Series): The low prices series.
    close (pd.Series): The close prices series.
    ilong (int): The period length for the KDJ indicator calculation.
    isig (int): The signal period length for the KDJ indicator calculation.

    Returns:
    pd.Series: The KDJ indicator series.
    """
    c = close
    h = high.rolling(ilong).max()
    l = low.rolling(ilong).min()
    rsv = 100 * ((c - l) / (h - l))
    pk = bcwsma(rsv, isig, 1)
    pd = bcwsma(pk, isig, 1)
    pj = 3 * pk - 2 * pd
    kdj = pj - pd
    return kdj

def TKE(data, period=14, emaperiod=5, novolumedata=False):
    """
    Calculate Technical Knowledge Extract (TKE) score based on multiple technical indicators.

    Parameters:
    - data (DataFrame): Input data containing 'close', 'high', 'low', 'volume' columns.
    - period (int): Period parameter used for calculating various indicators.
    - emaperiod (int): EMA period parameter (currently not used in the function).
    - novolumedata (bool): Flag to exclude volume-based indicators if True.

    Returns:
    - tke (Series): Series containing the calculated TKE scores.
    """
    df = data.copy()

    # Calculate various technical indicators
    mom_ = (df['close'] / df['close'].shift(period)) * 100
    cci_ = cci(df['high'], df['low'], df['close'], period)
    rsi_ = rsi(df['close'], period)
    willr_ = williams_r(df['high'], df['low'], df['close'], period)
    stoch_ = stochastic(df['high'], df['low'], df['close'], period)
    
    if novolumedata:
        # Exclude volume-based indicators
        tke = (cci_ + rsi_ + willr_ + stoch_) / 4
    else:
        # Include volume-based indicators
        mfi_ = mfi(df['high'], df['low'], df['close'], df['volume'], period)
        ultimate_ = ultimate_oscillator(df['high'], df['low'], df['close'], 7, 14, 28)
        tke = (ultimate_ + mfi_ + mom_ + cci_ + rsi_ + willr_ + stoch_) / 7
    
    return tke

def chandelier_exit(high, low, close, length=22, mult=3.0):
    """
    Calculates the Chandelier Exit indicator using high, low, and close prices.

    Args:
    high (pd.Series or list): The high prices.
    low (pd.Series or list): The low prices.
    close (pd.Series or list): The close prices.
    length (int, optional): The period over which the ATR is calculated. Default is 22.
    mult (float, optional): The multiplier for ATR. Default is 3.0.
    useClose (bool, optional): Whether to use the close price for extremums. Default is True.

    Returns:
    pd.Series: The Chandelier Exit values.
    """
    # Calculate ATR
    atr_value = atr(high, low, close, period=length)

    long_stop = close.rolling(window=length).max() - atr_value * mult
    long_stop_prev = long_stop.shift(1)
    long_stop = np.where(close.shift(1) > long_stop_prev, np.maximum(long_stop, long_stop_prev), long_stop)

    short_stop = close.rolling(window=length).min() + atr_value * mult
    short_stop_prev = short_stop.shift(1)
    short_stop = np.where(close.shift(1) < short_stop_prev, np.minimum(short_stop, short_stop_prev), short_stop)

    dir = 1
    dir= np.where(close > short_stop_prev, 1, np.where(close < long_stop_prev, -1, dir))

    for value in dir:
        if value == 1:
            chandelier = long_stop  # Set chandelier to longstop value or function
        elif value == -1:
            chandelier = short_stop  # Set chandelier to shortstop value or function
        return chandelier

    
#Return Dataframes
def bollinger_bands(series, length=20, std_multiplier=2):
    """
    Calculates the Bollinger Bands.

    Args:
    close (pd.Series): Closing prices.
    window (int): Window size for calculating moving average (default is 20).
    std_multiplier (int): Standard deviation multiplier for bands width (default is 2).

    Returns:
    pd.DataFrame: DataFrame with 'upper_band', 'middle_band', 'lower_band' columns.
    """
    middle_band = sma(series,length)
    std = stdev(series,length)
    upper_band = middle_band + std * std_multiplier
    lower_band = middle_band - std * std_multiplier
    bands = pd.DataFrame({'upper_band': upper_band, 'middle_band': middle_band, 'lower_band': lower_band})
    return bands

def nadaraya_watson_envelope(data, bandwidth, mult=3.0):
    """
    Calculate the Nadaraya-Watson envelope for a given time series data.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the time series data. Must contain a 'close' column.
    bandwidth (float): The bandwidth parameter for the Gaussian kernel.
    mult (float, optional): The multiplier for the scaled absolute error (SAE) to calculate the envelope width. Default is 3.0.

    Returns:
    pd.DataFrame: The DataFrame with the original data and three additional columns:
                  'Mid' - The Nadaraya-Watson estimator (smoothed center line)
                  'Lower' - The lower envelope line
                  'Upper' - The upper envelope line
    """
    
    # Make a copy of the data to avoid modifying the original DataFrame
    df = data.copy()
    
    # Define the Gaussian kernel function
    def gaussian_kernel(x, bandwidth):
        return np.exp(-0.5 * (x / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
    
    n = len(df)
    
    # Initialize the weights matrix
    weights = np.zeros((n, n))
    
    # Calculate the weights using the Gaussian kernel
    for i in range(n):
        for j in range(n):
            weights[i, j] = gaussian_kernel(i - j, bandwidth)
    
    # Normalize the weights so that each row sums to 1
    weights /= weights.sum(axis=1)[:, None]
    
    # Calculate the Nadaraya-Watson estimator
    nw_estimator = np.dot(weights, df['close'].values)
    nw_estimator_series = pd.Series(nw_estimator, index=df.index)
    
    # Calculate the scaled absolute error (SAE)
    sae = (df['close'] - nw_estimator_series).abs().rolling(window=n, min_periods=1).mean() * mult
    
    # Calculate the upper and lower envelope lines
    envelope_upper = nw_estimator_series + sae
    envelope_lower = nw_estimator_series - sae
    
    # Add the Nadaraya-Watson estimator and envelope lines to the DataFrame
    df['Mid'] = nw_estimator_series
    df['Lower'] = envelope_lower
    df['Upper'] = envelope_upper
    
    return df

def stochastic(high, low, close, length=14):
    """
    Calculate the Stochastic Oscillator.

    Parameters:
    high (pd.Series): High prices
    low (pd.Series): Low prices
    close (pd.Series): Close prices
    length (int): Lookback period for the Stochastic Oscillator (default: 14)

    Returns:
    pd.Series: Stochastic Oscillator values
    """
    # Calculate the highest high over the lookback period
    highest_high = high.rolling(window=length).max()
    
    # Calculate the lowest low over the lookback period
    lowest_low = low.rolling(window=length).min()
    
    # Calculate the Stochastic Oscillator
    stoch = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    return stoch

def stoch_rsi(series, length_rsi=14, length_stochrsi=14, k=3, d=3):
    """
    Calculate the Stochastic RSI (SRSI).

    Parameters:
    - series (numpy array): Time series data.
    - length_rsi (int): Length of RSI window.
    - length_stochrsi (int): Length of Stochastic RSI window.
    - k (int): Window size for %K calculation.
    - d (int): Window size for %D calculation.

    Returns:
    - srsi_df (DataFrame): DataFrame containing %K and %D values for SRSI.
    """
    # Assuming the rsi function is implemented elsewhere
    rsi_values = rsi(series, length_rsi)
    
    rsi_min = rsi_values.rolling(length_stochrsi).min()
    rsi_max = rsi_values.rolling(length_stochrsi).max()
    rsi_range = rsi_max - rsi_min

    stoch = 100 * (rsi_values - rsi_min)
    stoch /= rsi_range

    k_values = sma(stoch, length=k)
    d_values = sma(k_values, length=d)

    srsi = pd.DataFrame({'fast': k_values, 'slow': d_values})

    return srsi

def ultimate_oscillator(high, low, close, short_period=7, medium_period=14, long_period=28):
    """
    Calculate the Ultimate Oscillator

    Parameters:
    high (pd.Series): High prices
    low (pd.Series): Low prices
    close (pd.Series): Close prices
    short_period (int): Look-back period for short-term (default is 7)
    medium_period (int): Look-back period for medium-term (default is 14)
    long_period (int): Look-back period for long-term (default is 28)

    Returns:
    pd.Series: Ultimate Oscillator values
    """
    # Calculate True Low (TL) and True High (TH)
    true_low = pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    true_high = pd.concat([high, close.shift(1)], axis=1).max(axis=1)
    
    # Calculate Buying Pressure (BP)
    buying_pressure = close - true_low
    
    # Calculate True Range (TR)
    true_range = true_high - true_low
    
    # Calculate the average BP and TR for each period
    avg_bp_short = buying_pressure.rolling(window=short_period).sum()
    avg_tr_short = true_range.rolling(window=short_period).sum()
    
    avg_bp_medium = buying_pressure.rolling(window=medium_period).sum()
    avg_tr_medium = true_range.rolling(window=medium_period).sum()
    
    avg_bp_long = buying_pressure.rolling(window=long_period).sum()
    avg_tr_long = true_range.rolling(window=long_period).sum()
    
    # Calculate the Ultimate Oscillator
    ult_osc = 100 * ((4 * avg_bp_short / avg_tr_short) + (2 * avg_bp_medium / avg_tr_medium) + (avg_bp_long / avg_tr_long)) / (4 + 2 + 1)
    
    return ult_osc

def macd(series, fast=12, slow=26, signal=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given series and return a DataFrame.
    
    Parameters:
    - series: pandas Series containing price data.
    - fast (int): The period for the fast EMA. Default is 12.
    - slow (int): The period for the slow EMA. Default is 26.
    - signal (int): The period for the Signal line EMA. Default is 9. 
    
    Returns:
    - macd_df: pandas DataFrame with columns 'close', 'macd_line', 'signal_line', 'macd_histogram'.
    """
    emafast = ema(series, fast)
    emaslow = ema(series, slow)
    macd_line = emafast - emaslow
    signal_line = ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    
    macd_df = pd.DataFrame({
        'macd': macd_line,
        'macd_histogram': macd_histogram,
        'macd_signal': signal_line,
    })
    
    return macd_df

def dm(high, low, length=14):
    """
    Calculates the positive and negative directional movements (DM).

    Args:
    high (pd.Series): High prices.
    low (pd.Series): Low prices.
    length (int, optional): Length of the resulting Series. Default is 14.

    Returns:
    pd.DataFrame: DataFrame containing the positive DM (DMP) and negative DM (DMN).
    """
    up = high - high.shift(1)
    dn = low.shift(1) - low

    pos_ = ((up > dn) & (up > 0)) * up
    neg_ = ((dn > up) & (dn > 0)) * dn

    pos_ = pos_.apply(lambda x: 0 if x < 0 else x)
    neg_ = neg_.apply(lambda x: 0 if x < 0 else x)

    pos = rma(pos_, length)
    neg = rma(neg_, length)

    dm_df = pd.DataFrame({
        'DMP': pos,
        'DMN': neg,
    })
    
    return dm_df

def adx(high, low, close, period=14):
    """
    Calculates the Average Directional Index (ADX).

    Args:
    high (pd.Series or list): High prices of the asset.
    low (pd.Series or list): Low prices of the asset.
    close (pd.Series or list): Closing prices of the asset.
    period (int, optional): The period over which the ADX is calculated. Default is 14.

    Returns:
    pd.Series: The ADX values.
    """
    avtr = atr(high, low, close, period)

    dm_df =dm(high,low,period)

    dmp = 100 * (dm_df['DMP'] / avtr)
    dmn = 100 * (dm_df['DMN'] / avtr)

    dx = 100 * (dmp - dmn).abs() / (dmp + dmn)
    adx = rma(dx, length=period)

    adx_df = pd.DataFrame({
        'ADX': adx,
        'DMP': dmp,
        'DMN': dmn
    })    
    return adx_df

def Ichimoku_cloud(data, n1=9, n2=26, n3=52, n4=26, n5=26):
    df = data.copy()
    
    # Conversion Line (Tenkan-sen)
    high1 = df['high'].rolling(window=n1).max()
    low1 = df['low'].rolling(window=n1).min()
    df['conversion_line'] = (high1 + low1) / 2                                    
    
    # Base Line (Kijun-sen)
    high2 = df['high'].rolling(window=n2).max()
    low2 = df['low'].rolling(window=n2).min()
    df['baseline'] = (high2 + low2) / 2                                     
    
    # Leading Span A (Senkou Span A)
    df['Leading_A'] = ((df['conversion_line'] + df['baseline']) / 2).shift(n2)

    # Leading Span B (Senkou Span B)
    high3 = df['high'].rolling(window=n3).max()
    low3 = df['low'].rolling(window=n3).min()
    df['Leading_B'] = ((high3 + low3) / 2).shift(n4)
    return df

#Return Signals
def Donchian_Channel_Signal(data, window=20):
    """
    Calculates the Donchian Channel breakout signals.

    Args:
    data (pd.DataFrame): Input DataFrame containing columns 'high', 'low', and 'close'.
    window (int): Window size for the Donchian Channel.

    Returns:
    pd.DataFrame: DataFrame with added columns 'Upper Channel', 'Lower Channel', 'Entry', and 'Exit'.
    """
    df = data.copy()
    df['Upper Channel'] = data['high'].rolling(window=window).max()
    df['Lower Channel'] = data['low'].rolling(window=window).min()
    df['Entry'] = df['close'] > df['Upper Channel'].shift(1)
    df['Exit'] = df['close'] < df['Lower Channel'].shift(1)
    return df

def Keltner_Signal(data, window=20, mult=2):
    """
    Calculates the Keltner Channel breakout strategy signals.

    Args:
    data (pd.DataFrame): The input DataFrame containing columns like 'Upper Channel', 'Lower Channel', 'close', 'high', and 'low'.
    window (int): The window size for the Exponential Moving Average (EMA) calculation.
    mult (float): The multiplier for the True Range (TR) to determine the width of the Keltner Channel.

    Returns:
    pd.DataFrame: DataFrame with Keltner Channel breakout signals added.
    """
    df = data.copy()
    df['hlc3'] = hlc3(df['high'], df['low'], df['close'])
    df['tr'] = tr(df['high'], df['low'], df['close'])
    df['atr'] = atr(df['high'], df['low'], df['close'],window)
    df['ema'] = df['hlc3'].ewm(span=window, adjust=False).mean()
    
    df['Upper Channel'] = df['ema'] + df['atr'] * mult
    df['Lower Channel'] = df['ema'] - df['atr'] * mult

    df['Entry'] = (df['close'] > df['Lower Channel']) & (df['close'].shift(1) < df['Lower Channel'])
    df['Exit'] = (df['close'] > df['Upper Channel']) & (df['close'].shift(1) < df['Upper Channel'])
    return df

def SqueezeMomentum(data, mult=2, length=20, multKC=1.5, lengthKC=20):
    """
    Calculates the Squeeze Momentum indicator signals.

    Args:
    data (pd.DataFrame): The input DataFrame containing columns like 'close', 'high', and 'low'.
    mult (float): The multiplier for the standard deviation to determine the upper and lower Bollinger Bands.
    length (int): The window size for the Simple Moving Average (SMA) used in Bollinger Bands.
    multKC (float): The multiplier for the range moving average to determine the upper and lower Keltner Channels.
    lengthKC (int): The window size for the range moving average used in Keltner Channels.

    Returns:
    pd.DataFrame: DataFrame with Squeeze Momentum signals added.
    """
    df = data.copy()
    df['basis'] = sma(data['close'], length)
    df['dev'] = multKC * stdev(data['close'], length)
    df['upperBB'] = df['basis'] + df['dev']
    df['lowerBB'] = df['basis'] - df['dev']
    df['ma'] = sma(df['close'], lengthKC)
    df['tr0'] = abs(df["high"] - df["low"])
    df['tr1'] = abs(df["high"] - df["close"].shift())
    df['tr2'] = abs(df["low"] - df["close"].shift())
    df['range'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    df['rangema'] = sma(df['range'], lengthKC)
    df['upperKC'] = df['ma'] + df['rangema'] * multKC
    df['lowerKC'] = df['ma'] - df['rangema'] * multKC
    df['Squeeze'] = (df['lowerBB'] < df['lowerKC']) & (df['upperBB'] > df['upperKC'])
    return df

def Awesome_Oscillator_Signal(data, fast=5, slow=35):
    """
    Generates trading signals based on the Awesome Oscillator (AO).

    Args:
    data (pd.DataFrame): The input DataFrame containing columns like 'high' and 'low'.
    fast (int): The fast period for AO calculation.
    slow (int): The slow period for AO calculation.

    Returns:
    pd.DataFrame: DataFrame with Awesome Oscillator signals added.
    """
    df = data.copy()
    df['Awesome'] = ao(df['high'], df['low'], fast, slow)
    df['Entry'] = df['Awesome'] > 0
    df['Exit'] = df['Awesome'] < 0
    return df

def Macd_Signals(data, fast=12, slow=26, signal=9):
    """
    Generates trading signals based on the Moving Average Convergence Divergence (MACD).

    Args:
    data (pd.DataFrame): The input DataFrame containing columns like 'close'.
    fast (int): The fast period for MACD calculation.
    slow (int): The slow period for MACD calculation.
    signal (int): The signal period for MACD calculation.

    Returns:
    pd.DataFrame: DataFrame with MACD signals added.
    """    
    macd_df = macd(data['close'], fast=fast, slow=slow, signal=signal)
    macd_df['Entry'] = macd_df['macd'] > macd_df['macd_signal']
    macd_df['Exit'] = macd_df['macd'] < macd_df['macd_signal']
    result_df = pd.concat([data, macd_df], axis=1)
    return result_df

def DI_Signal(data,length=14):
    """
    Generates trading signals based on the Directional Movement Index (DMI).

    Args:
    data (pd.DataFrame): The input DataFrame containing columns like 'high', 'low', and 'close'.
    p (int): The period for ADX calculation.

    Returns:
    pd.DataFrame: DataFrame with DMP, DMN, Entry, and Exit columns.
    """    
    df=data.copy()
    df_adx=adx(data['high'],data['low'],data['close'],length)
    df['DMP']=df_adx['DMP']
    df['DMN']=df_adx['DMN']
    df['Entry']=df['DMP']>df['DMN']
    df['Exit']=df['DMP']<df['DMN']
    return df

def WaveTrend_Signal(data,n1=10,n2=21):
    """
    Calculate the WaveTrend signal for a given DataFrame.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame containing 'high', 'low', and 'close' columns.
    n1 (int, optional): The length of the first EMA. Default is 10.
    n2 (int, optional): The length of the second EMA. Default is 21.
    
    Returns:
    pd.DataFrame: The DataFrame with the WaveTrend calculations and Entry/Exit signals.
    """
    df=data.copy()
    df['ap'] = hlc3(df['high'], df['low'], df['close'])
    df['esa'] = ema(df['ap'], length=n1)
    df['d'] = ema((df['ap'] - df['esa']).abs(), length=n1)
    df['ci'] = (df['ap'] - df['esa']) / (0.015 * df['d'])
    df['tci1'] = ema(df['ci'], length=n2)
    df['tci2'] = sma(df['tci1'], 4)
    df['Entry'] = (df['tci1'] > df['tci2'])
    df['Exit'] = (df['tci1'] < df['tci2'])
    return df

def RSI_Cross_Signal(data,length_1=14,length_2=34,length_3=68):
    """
    Calculate RSI crosses and entry/exit signals based on RSI values.

    Parameters:
    - data: DataFrame containing 'close' prices.
    - length_1: Length for the first RSI calculation (default is 14).
    - length_2: Length for the second RSI calculation (default is 34).
    - length_3: Length for the third RSI calculation (default is 68).

    Returns:
    - DataFrame with additional columns:
        - 'RSI', 'RSI1', 'RSI2': RSI values calculated with different lengths.
        - 'Entry': True if RSI1 is greater than RSI2, indicating a potential entry point.
        - 'Exit': True if RSI1 is less than RSI2, indicating a potential exit point.
    """    
    df=data.copy()
    df['RSI']=rsi(df['close'],length_1)
    df['RSI1']=rsi(df['close'],length_2)
    df['RSI2']=rsi(df['close'],length_3)
    df['Entry']=(df['RSI1']>df['RSI2'])        
    df['Exit']=(df['RSI1']<df['RSI2']) 
    return df

def Psar_Signal(data,initial_af=0.02,step_af=0.02,max_af=0.2):
    df=data.copy()
    #Parabolic Stop and Reverse (PSAR)
    parabolic=psar(df['high'],df['low'],df['close'],initial_af,step_af,max_af)
    df['Entry'] = df['close'] > parabolic
    df['Exit'] = df['close'] < parabolic
    return df

def TillsonT3_Signal(data, Length=14, vf=0.7,app=2):
    df = data.copy()
    ema_first_input = (df['high'] + df['low'] + 2 * df['close']) / 4
    e1 = ema(ema_first_input, Length)
    e2 = ema(e1, Length)
    e3 = ema(e2, Length)
    e4 = ema(e3, Length)
    e5 = ema(e4, Length)
    e6 = ema(e5, Length)

    c1 = -1 * vf * vf * vf
    c2 = 3 * vf * vf + 3 * vf * vf * vf
    c3 = -6 * vf * vf - 3 * vf - 3 * vf * vf * vf
    c4 = 1 + 3 * vf + vf * vf * vf + 3 * vf * vf
    df['T3'] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    df['Entry'] = False
    df['Exit'] = False
    df['Entry']=df['T3'] > df['T3'].shift(1)
    df['Exit']=df['T3'] < df['T3'].shift(1)

    df['Entry']=(df['Entry'] & df['Entry'].shift(app))
    df['Exit']=(df['Exit'] & df['Exit'].shift(app))
    return df

def Supertrend_Signal(data,SENSITIVITY = 3,ATR_PERIOD = 14):
    df=data.copy()
    df['xATR'] = atr(data['high'], data['low'], data['close'],ATR_PERIOD)
    df['nLoss'] = SENSITIVITY * df['xATR']
    # Filling ATRTrailing Variable
    df['ATRTrailing'] = [0.0] + [np.nan for i in range(len(df) - 1)]

    for i in range(1, len(df)):
        if (df.loc[i, 'close'] > df.loc[i - 1, 'ATRTrailing']) and (df.loc[i - 1, 'close'] > df.loc[i - 1, 'ATRTrailing']):
            df.loc[i, 'ATRTrailing'] = max(df.loc[i - 1, 'ATRTrailing'],df.loc[i, 'close']-df.loc[i,'nLoss'])

        elif (df.loc[i, 'close'] < df.loc[i - 1, 'ATRTrailing']) and (df.loc[i - 1, 'close'] < df.loc[i - 1, 'ATRTrailing']):
            df.loc[i, 'ATRTrailing'] = min(df.loc[i - 1, 'ATRTrailing'],df.loc[i, 'close']+df.loc[i,'nLoss'])

        elif df.loc[i, 'close'] > df.loc[i - 1, 'ATRTrailing']:
            df.loc[i, 'ATRTrailing']=df.loc[i, 'close']-df.loc[i,'nLoss']
        else:
            df.loc[i, 'ATRTrailing']=df.loc[i, 'close']+df.loc[i,'nLoss']

    # Calculating signals
    ema_ = ema(df['close'], 1)
    df['Above'] = ema_ > (df['ATRTrailing'])
    df['Below'] = ema_ < (df['ATRTrailing'])
    df['Entry'] = (df['close'] > df['ATRTrailing']) & (df['Above']==True)
    df['Exit'] = (df['close'] < df['ATRTrailing']) & (df['Below']==True)
    return df

def Coral_Trend_Signal(data, sm=21, cd=0.4):
    df=data.copy()
    close=df['close'].values
    di = (sm - 1.0) / 2.0 + 1.0
    c1 = 2 / (di + 1.0)
    c2 = 1 - c1
    c3 = 3.0 * (cd * cd + cd * cd * cd)
    c4 = -3.0 * (2.0 * cd * cd + cd + cd * cd * cd)
    c5 = 3.0 * cd + 1.0 + cd * cd * cd + 3.0 * cd * cd

    i1 = np.zeros_like(close)
    i2 = np.zeros_like(close)
    i3 = np.zeros_like(close)
    i4 = np.zeros_like(close)
    i5 = np.zeros_like(close)
    i6 = np.zeros_like(close)

    for i in range(len(close)):
        i1[i] = c1 * close[i] + c2 * (i1[i - 1] if i > 0 else 0)
        i2[i] = c1 * i1[i] + c2 * (i2[i - 1] if i > 0 else 0)
        i3[i] = c1 * i2[i] + c2 * (i3[i - 1] if i > 0 else 0)
        i4[i] = c1 * i3[i] + c2 * (i4[i - 1] if i > 0 else 0)
        i5[i] = c1 * i4[i] + c2 * (i5[i - 1] if i > 0 else 0)
        i6[i] = c1 * i5[i] + c2 * (i6[i - 1] if i > 0 else 0)

    df['bfr'] = -cd * cd * cd * i6 + c3 * i5 + c4 * i4 + c5 * i3
    df['Entry'] = df['bfr']>df['bfr'].shift(1)
    df['Exit'] = df['bfr']<df['bfr'].shift(1)
    return df

def MOST_Signal(data, percent=2, n1=14):
    df = data.copy()
    percent = percent / 100
    df['EMA'] = ema(df['close'], length=n1)
    df['tempema'] = 0.0
    df['trend'] = -1
    df['MOST'] = 0.0
    df = df.dropna()
    df = df.reset_index()

    for i in range(1, len(df)):
        if df['trend'][i - 1] == 1:
            df.loc[i, 'tempema'] = max(df['tempema'][i - 1], df['EMA'][i])
        elif df['trend'][i - 1] == -1:
            df.loc[i, 'tempema'] = min(df['tempema'][i - 1], df['EMA'][i])

        if df['EMA'][i] >= df['MOST'][i - 1] and df['trend'][i - 1] == 1:
            df.loc[i, 'trend'] = 1
            df.loc[i, 'MOST'] = df['tempema'][i] * (1 - percent)
        elif df['EMA'][i] <= df['MOST'][i - 1] and df['trend'][i - 1] == -1:
            df.loc[i, 'trend'] = -1
            df.loc[i, 'MOST'] = df['tempema'][i] * (1 + percent)
        elif df['EMA'][i] >= df['MOST'][i - 1] and df['trend'][i - 1] == -1:
            df.loc[i, 'trend'] = 1
            df.loc[i, 'MOST'] = df['tempema'][i] * (1 - percent)
        elif df['EMA'][i] <= df['MOST'][i - 1] and df['trend'][i - 1] == 1:
            df.loc[i, 'trend'] = -1
            df.loc[i, 'MOST'] = df['tempema'][i] * (1 + percent)

    df['Entry'] = df['trend']==1
    df['Exit'] = df['trend']==-1
    return df

def OTT_Signal(data, prt, prc):
    df = data.copy()
    pds = prt
    percent = prc
    alpha = 2 / (pds + 1)

    df['ud1'] = np.where(df['close'] > df['close'].shift(1), (df['close'] - df['close'].shift()), 0)
    df['dd1'] = np.where(df['close'] < df['close'].shift(1), (df['close'].shift() - df['close']), 0)
    df['UD'] = df['ud1'].rolling(9).sum()
    df['DD'] = df['dd1'].rolling(9).sum()
    df['CMO'] = ((df['UD'] - df['DD']) / (df['UD'] + df['DD'])).fillna(0).abs()

    df['Var'] = 0.0
    for i in range(pds, len(df)):
        df['Var'].iat[i] = (alpha * df['CMO'].iat[i] * df['close'].iat[i]) + (1 - alpha * df['CMO'].iat[i]) * df['Var'].iat[i - 1]

    df['fark'] = df['Var'] * percent * 0.01
    df['newlongstop'] = df['Var'] - df['fark']
    df['newlongstop'] = df['newlongstop'].astype('int64')
    df['newshortstop'] = df['Var'] + df['fark']
    df['newshortstop'] = df['newshortstop'].astype('int64')
    df['longstop'] = 0.0
    df['shortstop'] = 999999999999999999

    for i in df['UD']:
        def maxlongstop():
            df.loc[(df['newlongstop'] > df['longstop'].shift(1)), 'longstop'] = df['newlongstop']
            df.loc[(df['longstop'].shift(1) > df['newlongstop']), 'longstop'] = df['longstop'].shift(1)
            return df['longstop']

        def minshortstop():
            df.loc[(df['newshortstop'] < df['shortstop'].shift(1)), 'shortstop'] = df['newshortstop']
            df.loc[(df['shortstop'].shift(1) < df['newshortstop']), 'shortstop'] = df['shortstop'].shift(1)
            return df['shortstop']

        df['longstop'] = np.where(((df['Var'] > df['longstop'].shift(1))), maxlongstop(), df['newlongstop'])
        df['shortstop'] = np.where(((df['Var'] < df['shortstop'].shift(1))), minshortstop(), df['newshortstop'])

    df['xlongstop'] = np.where(((df['Var'].shift(1) > df['longstop'].shift(1)) & (df['Var'] < df['longstop'].shift(1))), 1, 0)
    df['xshortstop'] = np.where(((df['Var'].shift(1) < df['shortstop'].shift(1)) & (df['Var'] > df['shortstop'].shift(1))), 1, 0)

    df['trend'] = 0
    df['dir'] = 0

    for i in df['UD']:
        df['trend'] = np.where(((df['xshortstop'] == 1)), 1, (np.where((df['xlongstop'] == 1), -1, df['trend'].shift(1))))
        df['dir'] = np.where(((df['xshortstop'] == 1)), 1, (np.where((df['xlongstop'] == 1), -1, df['dir'].shift(1).fillna(1))))

    df['MT'] = np.where(df['dir'] == 1, df['longstop'], df['shortstop'])
    df['OTT'] = np.where(df['Var'] > df['MT'], (df['MT'] * (200 + percent) / 200), (df['MT'] * (200 - percent) / 200))

    df = df.round(2)
    df['OTT2'] = df['OTT'].shift(2)
    df['OTT3'] = df['OTT'].shift(3)
    df['Entry'] = df['Var'] > df['OTT2']
    df['Exit'] = df['Var'] < df['OTT2']
    return df

def Banker_Fund_Trend(data):
    df = data.copy()
    close = df['close']
    low = df['low']
    high = df['high']
    open_ = df['open']
    fundtrend =np.zeros_like(close)
    part_1 = (close - low.rolling(window=27).min()) / (high.rolling(window=27).max() - low.rolling(window=27).min()) * 100
    fundtrend = ((3 * xsa(part_1, 5, 1) - 2 * xsa(xsa(part_1, 5, 1), 3, 1) - 50) * 1.032 + 50)

    typ = (2 * close + high + low + open_) / 5
    lol = low.rolling(window=34).min()
    hoh = high.rolling(window=34).max()

    bullbearline = ema((typ - lol) / (hoh - lol) * 100, 13)
    bankerentry = (fundtrend > bullbearline) & (bullbearline < 25)

    df['fundtrend'] = fundtrend
    df['bullbearline'] = bullbearline
    df['bankerentry'] = bankerentry.astype(int)  # Convert to integer

    return df

def TSI_Signal(data, short_len, long_len, signal_len):
    df = data.copy()
    diff = df['close'].diff()
    abs_diff = diff.abs()

    short_ema_diff = ema(diff, short_len)
    long_ema_diff = ema(short_ema_diff, long_len)

    short_ema_abs_diff = ema(abs_diff, short_len)
    long_ema_abs_diff = ema(short_ema_abs_diff, long_len)

    df['tsi'] = 100 * (long_ema_diff / long_ema_abs_diff)
    df['tsi_signal'] = ema(df['tsi'], signal_len)

    df['Entry'] = df['tsi'] > df['tsi_signal']
    df['Exit'] = df['tsi'] < df['tsi_signal']
    return df

def Alpha_Trend_Signal(data, mult=1, n1=14):
    df = data.copy()
    df['TR'] = tr(df['high'], df['low'], df['close'])
    df['ATR'] = sma(df['TR'], length=n1)
        
    df['mfi'] = mfi(df['high'], df['low'], df['close'], df['volume'], n1)
    df['upT'] = df['low'] - df['ATR'] * mult
    df['downT'] = df['high'] + df['ATR'] * mult
    df['AlphaTrend'] = 0.0
    alpha_trend_values = [0.0]

    for i in range(1, len(df)):
        if df['mfi'].iloc[i] >= 50:
            alpha_trend_values.append(max(df['upT'].iloc[i], alpha_trend_values[i-1]))
        else:
            alpha_trend_values.append(min(df['downT'].iloc[i], alpha_trend_values[i-1]))

    df['AlphaTrend'] = alpha_trend_values
    df['Entry'] = False
    prev_signal = False
    for i in range(2, len(df)):
        if df.loc[i, 'AlphaTrend'] > df.loc[i-2, 'AlphaTrend']:
            df.loc[i, 'Entry'] = True
            prev_signal = True
        elif df.loc[i, 'AlphaTrend'] == df.loc[i-2, 'AlphaTrend'] and prev_signal:
            df.loc[i, 'Entry'] = True
        else:
            prev_signal = False
    df['Exit'] = ~df['Entry']
    return df

def EWO_Signal(data, short_period, long_period):
    df = data.copy()
    df['Ewo'] = ewo(df['close'],short_period,long_period)
    df['Entry'] = df['Ewo'] > 0
    df['Exit'] = df['Ewo'] < 0
    return df

def Ichimoku_Signal(data, n1=9, n2=26, n3=52, n4=26, n5=26):
    df = data.copy()
    
    # Conversion Line (Tenkan-sen)
    high1 = df['high'].rolling(window=n1).max()
    low1 = df['low'].rolling(window=n1).min()
    df['tenkansen'] = (high1 + low1) / 2                                    
    
    # Base Line (Kijun-sen)
    high2 = df['high'].rolling(window=n2).max()
    low2 = df['low'].rolling(window=n2).min()
    df['kijunsen'] = (high2 + low2) / 2                                     
    
    # Leading Span A (Senkou Span A)
    df['senkou_A'] = ((df['tenkansen'] + df['kijunsen']) / 2).shift(n2)

    # Leading Span B (Senkou Span B)
    high3 = df['high'].rolling(window=n3).max()
    low3 = df['low'].rolling(window=n3).min()
    df['senkou_B'] = ((high3 + low3) / 2).shift(n4)
    
    # Lagging Span (Chikou Span)
    df['chikou'] = df['close'].shift(-n5)

    df['Entry'] = (df['close'] > df['senkou_A']) & (df['close'] > df['senkou_B']) & (df['close'] > df['kijunsen']) &  df['chikou'] > df['close']
    df['Exit'] = (df['close'] < df['senkou_A']) | (df['close'] < df['senkou_B']) | (df['close'] < df['kijunsen']) | (df['chikou'] < df['close'])

    return df

def Relative_Volume_Signal(data,length=10,limitl=0.9,limith=1.3):
    df=data.copy()
    df['RV'] = relative_volume(df['volume'],length)
    df['Entry'] = df['RV'] > limith
    df['Exit'] = df['RV'] < limitl
    return df

def Divergence(data,DivCheck,order=3):
    df = data.copy()  
    df['Divcheck']=DivCheck
    data = data.reset_index()
    hh_pairs=argrelextrema(df['close'].values, comparator=np.greater, order=order)[0]
    hh_pairs=[hh_pairs[i:i+2] for i in range(len(hh_pairs)-1)]
    
    ll_pairs=argrelextrema(df['close'].values, comparator=np.less, order=order)[0]
    ll_pairs=[ll_pairs[i:i+2] for i in range(len(ll_pairs)-1)]
    
    bear_div=[]
    bull_div=[]

    for p in hh_pairs:
        x_price=p
        y_price=[df['close'].iloc[p[0]], df['close'].iloc[p[1]]]
        slope_price=stats.linregress(x_price, y_price).slope
        x_divcheck=p
        y_divcheck=[df['Divcheck'].iloc[p[0]], df['Divcheck'].iloc[p[1]]]
        slope_divcheck=stats.linregress(x_divcheck, y_divcheck).slope
        
        if slope_price>0:
            if np.sign(slope_price)!=np.sign(slope_divcheck):
                bear_div.append(p)
       
    for p in ll_pairs:
        x_price=p
        y_price=[df['close'].iloc[p[0]], df['close'].iloc[p[1]]]
        slope_price=stats.linregress(x_price, y_price).slope
        x_divcheck=p
        y_divcheck=[df['Divcheck'].iloc[p[0]], df['Divcheck'].iloc[p[1]]]
        slope_divcheck=stats.linregress(x_divcheck, y_divcheck).slope
        
        if slope_price<0:
            if np.sign(slope_price)!=np.sign(slope_divcheck):
                bull_div.append(p)    
    
    bear_points=[df.index[a[1]] for a in bear_div]
    bull_points=[df.index[a[1]] for a in bull_div]
    df['position']=0
    pos=[]
    
    for idx in df.index:
        if idx in bear_points:
            pos.append(-1)
        elif idx in bull_points:
            pos.append(1)
        else:
            pos.append(0)
    
    df['position']=pos
    return df
