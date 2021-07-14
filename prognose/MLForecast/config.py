config = {
    "method": "random",
    "parameters": {
        "epochs": {
            "value": 200
        },
        "early_stop": {
            "value": 20
        },
        "loss": {
            "values": ["mae", "mse"]
        },
        "x_width": {
            "value": 96
        },
        "y_width": {
            "value": 12
        },
        "structure": {
            "values": [
                "LSTM_x+DROP_d+LSTM_x+DROP_d+DENSE_y+OUT",
                "LSTM_x+DROP_d+LSTM_x+OUT",
                "LSTM_x+DROP_d+DENSE_x+OUT",
                "LSTM_x+DROP_d+LSTM_y+DROP_d+DENSE_y+OUT"
            ]
        },
        "x": {
            "values": [32, 64, 128, 256]
        },
        "d": {
            "values": [0.15, 0.25, 0.35]
        },
        "l2": {
            "values": [1e-3, 5e-5, 1e-6, 0]
        }
    }
}
