"""Quantlib research utilities.

Submodules are intentionally not imported eagerly: several integrations have
optional third-party dependencies, while ``Quantlib.backtest`` should remain
usable in a lightweight research environment.
"""

__all__ = []
