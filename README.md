# Time-Series-Forecasting

    def prepare_data_prophet(self, target_col):
        logging.info("Preparing data for Prophet model")
        data_prophet = self.data.reset_index().rename(columns={'Date': 'ds', target_col: 'y'})
        return data_prophet

from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from joblib import dump, load

class ProphetModel:
    def __init__(self, data, regressors=None):
        self.data = data
        self.regressors = regressors
        self.model_uni = Prophet(daily_seasonality=True)
        self.model_multi = Prophet(daily_seasonality=True) if regressors else None
        self.forecast_uni = None
        self.forecast_multi = None

    def train_model(self):
        logging.info("Training Prophet model")
        
        # Univariate Model
        self.model_uni.fit(self.data)
        logging.info("Univariate model trained")
        
        # Multivariate Model
        if self.regressors:
            for regressor in self.regressors:
                self.model_multi.add_regressor(regressor)
            self.model_multi.fit(self.data)
            logging.info("Multivariate model trained")

    def make_predictions(self, periods):
        logging.info("Making predictions with Prophet model")
        future_uni = self.model_uni.make_future_dataframe(periods=periods)
        self.forecast_uni = self.model_uni.predict(future_uni)
        
        if self.regressors:
            future_multi = self.model_multi.make_future_dataframe(periods=periods)
            self.forecast_multi = self.model_multi.predict(future_multi)

    def evaluate_model(self):
        if not self.forecast_uni:
            logging.error("No predictions made yet.")
            return

        metrics = {'RMSE': mean_squared_error, 'MAE': mean_absolute_error, 'R2 Score': r2_score}
        evaluation_uni = {}
        
        for name, metric in metrics.items():
            evaluation_uni[name] = metric(self.data['y'], self.forecast_uni['yhat'][:len(self.data)])

        logging.info(f"Univariate model evaluation: {evaluation_uni}")
        
        if self.regressors:
            evaluation_multi = {}
            for name, metric in metrics.items():
                evaluation_multi[name] = metric(self.data['y'], self.forecast_multi['yhat'][:len(self.data)])
            logging.info(f"Multivariate model evaluation: {evaluation_multi}")

    def plot_predictions(self):
        if not self.forecast_uni:
            logging.error("No predictions made yet.")
            return

        logging.info("Plotting predictions")
        self.model_uni.plot(self.forecast_uni)

        if self.regressors:
            self.model_multi.plot(self.forecast_multi)

    def save_model(self):
        model_uni_name = "prophet_uni_model.joblib"
        model_uni_path = os.path.join("trained_models", model_uni_name)
        dump(self.model_uni, model_uni_path)
        logging.info(f"Univariate model saved at {model_uni_path}")

        if self.regressors:
            model_multi_name = "prophet_multi_model.joblib"
            model_multi_path = os.path.join("trained_models", model_multi_name)
            dump(self.model_multi, model_multi_path)
            logging.info(f"Multivariate model saved at {model_multi_path}")

    @staticmethod
    def load_model(model_path):
        loaded_model = load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return loaded_model
