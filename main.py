from AI import oracle

oracle(
    portfolio=["TSLA", "AAPL", "NVDA", "NFLX"],
    start_date="2020-01-01",
    weights=[0.3, 0.2, 0.3, 0.2],  # allocate 30% to TSLA and 20% to AAPL...(equal weighting  by default)
    prediction_days=2  # number of days you want to predict
)
