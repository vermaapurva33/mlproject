import os 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path =os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split training and test input data")

            X_train, y_train, X_test,y_test= (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighbors Classifier": KNeighborsRegressor(),
                "CatBoosting Classifier":CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            # params= {
            #     "Decision Tree": {
            #         'criterion': ['absolute_error', 'friedman_mse', 'absolute_error','poisson'],

            #     }
            # }

            model_report:dict =evaluate_models(X_train=X_train,y_train=y_train,X_test =X_test,y_test=y_test,models=models)

            # to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # to get best model name 

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model =models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e, sys)   