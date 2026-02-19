import numpy as np
import pandas as pd
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

def load_data(filepath  : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading from {filepath} : {e}")
# test_data= pd.read_csv('./data/processed/test_processed.csv')

#fetching independent data
#X_test= test_data.iloc[:, 0:-1].values
#fetching dependent data
#y_test=test_data.iloc[:,-1].values

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns= ['Potability'])
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

def load_model(filepath: str):
    try:
        with open(filepath, "rb") as file:
            model= pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}:{e}")
        
#model=pickle.load(open("model.pkl","rb")) #rb- read binary mode

def evaluation_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        y_pred= model.predict(X_test)

        accuracy= accuracy_score(y_test, y_pred)
        precision= precision_score(y_test, y_pred)
        recall= recall_score(y_test, y_pred)
        f1score= f1_score(y_test, y_pred)

        #storing data in json file
        metrics_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall': recall,
            'f1_score':f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")

def save_metrics(metrics_dict: dict, filepath: str) -> None:
    try: 
        with open('metrics.json','w') as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {filepath}: {e}")
    
def main():
    try:
        test_data_path="./data/processed/test_processed.csv"
        model_path ="model.pkl"
        metrics_path = "metrics.json"
        
        test_data = load_data(test_data_path)
        X_test, y_test =  prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f"An error occured: {e}")
    
if __name__=="__main__":
    main()
    
    
    
    
