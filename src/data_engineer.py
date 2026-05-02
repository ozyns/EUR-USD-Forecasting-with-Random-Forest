import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings


warnings.filterwarnings('ignore')

class EnhancedDataEngineer:

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        

    def collect_data(self):
        print("Collecting data...")

        data = yf.download('EURUSD=X', start='2023-01-01', end='2023-12-31')
        print(data.head())

        try:
            eurusd = yf.download('EURUSD=X', start=self.start_date, end=self.end_date, progress=False)
            if len(eurusd) > 0:
                self.data = pd.DataFrame({'EUR_USD': eurusd.loc[:, ('Close', 'EURUSD=X')]})
                print(f"✓ EUR/USD: {len(eurusd)} rows")
            else:
                raise ValueError("No data")
        except:
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            np.random.seed(42)
            rates = 1.10 + np.cumsum(np.random.normal(0, 0.002, len(dates)))
            self.data = pd.DataFrame({'EUR_USD': rates}, index=dates)
            print(f"✓ Using synthetic EUR/USD: {len(self.data)} rows")


        # Add Gold and Oil
        try:
            gold = yf.download('GC=F', start=self.start_date, end=self.end_date, progress=False)
            if len(gold) > 0:
                self.data['Gold'] = gold['Close']
                print(f"✓ Gold added")
        except:
            print("  Gold skipped")

        try:
            oil = yf.download('CL=F', start=self.start_date, end=self.end_date, progress=False)
            if len(oil) > 0:
                self.data['Oil'] = oil['Close']
                print(f"✓ Oil added")
        except:
            print("  Oil skipped")

        self.data = self.data.ffill().bfill().dropna(axis=1, how='all')
        print(f"✓ Dataset ready: {len(self.data)} rows, {len(self.data.columns)} columns")
        return self.data


    def create_enhanced_features(self):
        """BETTER features for higher accuracy"""
        print("Creating enhanced features...")

        close = self.data['EUR_USD']

        # 1. RETURNS (better than raw prices!)
        self.data['return_1d'] = close.pct_change(1)
        self.data['return_3d'] = close.pct_change(3)
        self.data['return_7d'] = close.pct_change(7)

        # 2. VOLATILITY (critical for forex!)
        self.data['volatility_7'] = close.pct_change().rolling(7).std()
        self.data['volatility_14'] = close.pct_change().rolling(14).std()

        # 3. Moving averages + CROSSOVERS
        self.data['MA_7'] = close.rolling(7).mean()
        self.data['MA_21'] = close.rolling(21).mean()
        self.data['MA_50'] = close.rolling(50).mean()
        self.data['MA_cross_7_21'] = self.data['MA_7'] / self.data['MA_21']
        self.data['MA_cross_21_50'] = self.data['MA_21'] / self.data['MA_50']

        # 4. MOMENTUM indicators
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        self.data['RSI'] = 100 - (100 / (1 + rs))
        self.data['RSI_signal'] = (self.data['RSI'] > 70).astype(int) - (self.data['RSI'] < 30).astype(int)

        # 5. LAG features (multiple timeframes)
        for lag in [1, 2, 3, 5, 7, 14]:
            self.data[f'lag_{lag}'] = close.shift(lag)
            self.data[f'return_lag_{lag}'] = close.pct_change().shift(lag)

        # 6. ROLLING statistics
        self.data['high_7d'] = close.rolling(7).max()
        self.data['low_7d'] = close.rolling(7).min()
        self.data['range_7d'] = self.data['high_7d'] - self.data['low_7d']

        # 7. If we have Gold/Oil, add their features
        for col in ['Gold', 'Oil']:
            if col in self.data.columns:
                self.data[f'{col}_return'] = self.data[col].pct_change()
                self.data[f'{col}_MA_7'] = self.data[col].rolling(7).mean()

        # 8. Temporal features
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['quarter'] = self.data.index.quarter

        print(f"✓ Created {len(self.data.columns)} features total")
        return self.data

    def process_all(self):
        self.collect_data()
        self.create_enhanced_features()

        initial_rows = len(self.data)
        self.data = self.data.dropna()
        dropped_rows = initial_rows - len(self.data)

        print(f"✓ Dropped {dropped_rows} rows with NaN")
        print(f"✓ Final: {len(self.data)} rows, {len(self.data.columns)} features")

        self.data.to_csv('processed_data.csv')
        print(f"✓ Saved to 'processed_data.csv'")

        return self.data