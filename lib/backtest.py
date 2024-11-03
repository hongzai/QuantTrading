class BacktestResult:
    def __init__(self, positions, close_prices, trading_fee=0):
        self.positions = positions
        self.close_prices = close_prices
        self.trading_fee = trading_fee
        
        # 计算各项指标
        self.pnl = self._calculate_pnl()
        self.cumulative_pnl = self.pnl.cumsum()
        self.sharpe_ratio = self._calculate_sharpe_ratio()
        self.mdd = self._calculate_mdd()  # 确保这里计算MDD
        
    def _calculate_mdd(self):
        # 使用累计收益计算MDD
        cumulative_returns = (1 + self.pnl).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        mdd = drawdowns.min()
        return abs(mdd)  # 返回正值 