"""AquaPulse backend: hot-swap data pipeline + LSTM forecasting engine.

Heavy dependencies (TensorFlow) are imported lazily inside lstm_engine
functions, so importing this package is always fast and safe.
"""
__version__ = "1.0.0"
