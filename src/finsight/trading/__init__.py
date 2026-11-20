"""FinSight's standalone PAPER-ONLY research and execution simulator."""
from .core import OrderResult, PaperLedger, PaperQuote, RiskLimits
from .signals import StrategySignal, sma_crossover

__all__ = ["OrderResult", "PaperLedger", "PaperQuote", "RiskLimits", "StrategySignal", "sma_crossover"]
