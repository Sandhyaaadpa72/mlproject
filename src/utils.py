import os
import sys
import pickle
import dill  # Use dill if you are working with complex models
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
import logging

# Save the model to disk using pickle or dill
def save_object(file_path, obj):
    try:
        # Get the directory path from the file path
        dir_path = os.path.dirname(file_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Save the object to the specified file path
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)  # Using dill instead of pickle (use pickle if not dealing with complex models)

    except Exception as e:
        # Raise custom exception if any error occurs
        raise CustomException(e, sys)

# Function to evaluate models using GridSearchCV
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        # Dictionary to store model names and their test score
        report = {}

        # Iterate over all models
        for i in range(len(models)):
            model_name = list(models.keys())[i]  # Get model name
            model = models[model_name]           # Get model object
            para = param.get(model_name, {})    # Get hyperparameters for the model

            # Perform GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3, scoring='r2')  # Specify the scoring metric
            gs.fit(X_train, y_train)  # Fit the grid search on the training data

            # Set the best parameters obtained from GridSearchCV
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Fit the model with the best parameters

            # Make predictions on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R-squared score for train and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Log the results
            logging.info(f"Model: {model_name}, Train Score: {train_model_score}, Test Score: {test_model_score}")

            # Store the test score in the report dictionary
            report[model_name] = test_model_score

        # Return the report containing model names and their test scores
        return report

    except Exception as e:
        # Raise custom exception if any error occurs
        raise CustomException(e, sys)

# Load the saved object from disk using pickle or dill
def load_object(file_path):
    try:
        # Open the file in binary read mode
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)  # Use dill instead of pickle (use pickle if not dealing with complex models)

    except Exception as e:
        # Raise custom exception if any error occurs
        raise CustomException(e, sys)
