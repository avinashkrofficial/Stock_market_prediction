# Stock Market Prediction Using ML, nselib, and LSTM

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Model Architecture](#model-architecture)
8. [Results](#results)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [License](#license)

---

## Introduction
This project focuses on predicting stock market prices using machine learning techniques, specifically leveraging Long Short-Term Memory (LSTM) neural networks. It utilizes the **nselib** library to fetch data from the Indian stock market (Nifty 50) and employs Exponential Moving Average (EMA) as a key feature for trend analysis. The goal is to provide an accurate model that can predict future stock prices based on historical data.

---

## Features
- Fetch live stock market data using **nselib**.
- Process historical data for feature extraction and analysis.
- Use **LSTM** for time-series prediction.
- Incorporate **EMA** for trend smoothing and prediction improvement.
- Visualize historical data and predictions.

---

## Technologies Used
- **Python**: Core programming language.
- **TensorFlow/Keras**: Framework for building and training the LSTM model.
- **nselib**: For accessing Nifty 50 stock market data.
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib/Seaborn**: For data visualization.

---

## Installation
Follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-market-prediction.git
   cd stock-market-prediction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have access to the **nselib** library. For installation instructions, refer to the [nselib documentation](https://pypi.org/project/nselib/).

---

## Usage
1. Run the script to fetch and preprocess the data:
   ```bash
   python data_preprocessing.py
   ```

2. Train the LSTM model:
   ```bash
   python train_model.py
   ```

3. Generate predictions and visualize results:
   ```bash
   python predict_and_visualize.py
   ```

---

## Dataset
The dataset is fetched using **nselib** from Nifty 50 historical data. Key features include:
- Open, High, Low, Close prices.
- Volume.
- Exponential Moving Average (EMA).

---

## Model Architecture
The project employs a stacked **LSTM** architecture:
- Input Layer: Accepts historical stock price features.
- Hidden Layers: Multiple LSTM layers for capturing temporal dependencies.
- Output Layer: Predicts the next price value.

The model is trained using Mean Squared Error (MSE) as the loss function and Adam optimizer.

---

## Results
- The model achieves high accuracy for predicting short-term trends.
- Visualizations include predicted vs. actual prices and trend lines based on EMA.

Example prediction plot:

![Prediction Plot](assets/prediction_plot.png)

---

## Future Work
- Improve accuracy for long-term predictions.
- Incorporate additional features such as sentiment analysis from news headlines.
- Deploy the model as a web application using Flask or Django.
- Add support for more stock indices and global markets.

---

## Contributing
Contributions are welcome! If you have ideas to improve this project, please:
1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

