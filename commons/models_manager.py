import numpy as np
from datetime import datetime
import seaborn as sns
sns.set_style("darkgrid")
class SingleModelData():
    def __init__(self, model, set_time_start = False):
        self.model = model
        self.model_id = None
        self.model_architecture = None
        self.model_full_description = None
        self.loss = []
        self.val_loss = []
        self.test_loss = None
        self.train_RMSE = []
        self.val_RMSE = []
        self.test_RMSE = None
        self.epochs = 0
        if set_time_start:
            self.time_start = datetime.now()
        else:
            self.time_start = None
        self.time_end = None   
        self.total_params = None
        self.layers_cnt = None
        self.total_units = None
        self.df_results = None 

    def __str__(self):
        import json
        return(
            json.dumps(
                {
                'loss':self.loss[-1],
                'val_loss':self.val_loss[-1],
                'test_loss':self.test_loss,
                'train_RMSE':self.train_RMSE[-1],
                'val_RMSE':self.val_RMSE[-1],
                'test_RMSE':self.test_RMSE,
                'epochs':self.epochs,
                'time_end':self.time_end,
                'total_params':self.total_params,
                'layers_cnt':self.layers_cnt,
                'total_units':self.total_units,
                },
                indent=4,
                default=str
            )
        )
    
    def plot_rmse(self):
        import matplotlib.pyplot as plt
        plt.plot(self.train_RMSE, label='Train RMSE')
        plt.plot(self.val_RMSE, label='Validation RMSE')
        plt.axhline(y=self.test_RMSE, linestyle='--', color='violet', label='Test RMSE at last epoch')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.ylim([0, None]) #Make y_axis start on 0 
        plt.title('Train and validation RMSE during each epoch')
        plt.legend() 


class MultiModelsData:
    def __init__(self):
        self.models = {}
        self.max_index = 0

    def add_model(self, model, set_time_start=True):
        model_id = f"model{self.max_index}"
        self.models[model_id] = SingleModelData(model, set_time_start=set_time_start)
        self.max_index += 1

        return model_id


    def fill_data(self, model_id, model, training_history, test_loss, set_time_end=True, pat="unkwnon", bs="unkwnon", comment=""):
        from commons.model_functions import get_model_total_layers_params_and_units, get_model_description, get_model_layers_info
        model_full_description = get_model_description(model, pat="unknown", bs="unknown", comment="")
        model_architecture = get_model_layers_info(model, ignore_last_layer=True)
        total_params, layers_cnt, total_units = get_model_total_layers_params_and_units(model)

        self.models[model_id].model = model
        self.models[model_id].model_full_description = model_full_description
        self.models[model_id].model_architecture = model_architecture

        self.models[model_id].loss = training_history.history['loss']
        self.models[model_id].val_loss = training_history.history['val_loss']
        self.models[model_id].test_loss = test_loss
        self.models[model_id].train_RMSE = np.sqrt(self.models[model_id].loss)
        self.models[model_id].val_RMSE = np.sqrt(self.models[model_id].val_loss)
        self.models[model_id].test_RMSE = np.sqrt(self.models[model_id].test_loss)
        self.models[model_id].epochs = len(self.models[model_id].loss)
        if set_time_end:
            self.models[model_id].time_end = datetime.now()
        else:
            self.models[model_id].time_end = None
        self.models[model_id].total_params = total_params
        self.models[model_id].layers_cnt = layers_cnt
        self.models[model_id].total_units = total_units

    def generate_results_dataframe(self):
        import pandas as pd
        df_results = pd.DataFrame(columns=['Model id', 'Model architecture', 'Model full description', 'Train RMSE', 'Val RMSE', 'Test RMSE', 'Training_time', 'epochs', 'total_params'])

        for model_id, model_data in self.models.items():
            #Calculate duration
            if (model_data.time_end is not None) and (model_data.time_start is not None):
                training_time = model_data.time_end - model_data.time_start
            else:
                training_time = None

            #Generate single row of result frame
            df_single_row = pd.DataFrame(
                data = {
                    'Model id' : model_id,
                    'Model architecture' : model_data.model_architecture,
                    'Model full description' : model_data.model_full_description,
                    'Train RMSE' : model_data.train_RMSE[-1],
                    'Val RMSE' : model_data.val_RMSE[-1],
                    'Test RMSE' : model_data.test_RMSE,
                    'Training_time' : training_time,
                    'epochs' : model_data.epochs,
                    'total_params' : model_data.total_params,
                }, 
                index=[0])
            
            #Add row to result frame
            df_results = pd.concat([df_results, df_single_row]).reset_index(drop=True)

        self.df_results = df_results
        return df_results
    
    def plot_rmse_subplots(self, print_best_train_rmse=False, print_best_val_rmse=False, print_best_test_rmse=True, x_min=None, x_max=None, y_min=0, y_max=None, num_cols=3):
        """
        Prints RMSE for each model in models dict.
        INPUTS:
        print_best_train_rmse (False) - Adds horozintal line for best value of train_rmse in all models
        print_best_val_rmse (False) - Adds horozintal line for best value of validation_rmse in all models
        print_best_test_rmse (True) - Adds horozintal line for best value of test_rmse in all models
        x_min (None) - Used to override startin point on x_axis
        x_max (None) - Used to override ending point on x_axis
        y_min (0) - Used to override starting point on y_axis
        y_max (None) - Used to override ending point on y_axis
        num_cols (3) - determines number of plots for each row
        """
        import matplotlib.pyplot as plt
        num_rows = (len(self.models) + num_cols - 1) // num_cols
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(28, 5*num_rows))
        for i, (model_id, model_data) in enumerate(self.models.items()):        
            row_idx = i // num_cols
            col_idx = i % num_cols
            ax = axs[row_idx, col_idx]
            ax.plot(model_data.train_RMSE, label='Train RMSE')
            ax.plot(model_data.val_RMSE, label='Validation RMSE')
            ax.axhline(y=model_data.test_RMSE, linestyle='-', color='purple', label='Test RMSE at last epoch', linewidth=1)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('RMSE')
            ax.set_xlim([x_min, x_max]) 
            ax.set_ylim([y_min, y_max]) 
            ax.set_title(model_id)

            #Optional Extra lines
            if (print_best_train_rmse == True) and (self.df_results is not None) and (not self.df_results.empty):
                ax.axhline(y=self.df_results['Train RMSE'].min(), linestyle='--', color='C0', label='Best train RMSE', linewidth=1)
            if (print_best_val_rmse == True) and (self.df_results is not None) and (not self.df_results.empty):
                ax.axhline(y=self.df_results['Val RMSE'].min(), linestyle='--', color='C1', label='Best val RMSE', linewidth=1)
            if (print_best_test_rmse == True) and (self.df_results is not None) and (not self.df_results.empty):
                ax.axhline(y=self.df_results['Test RMSE'].min(), linestyle='--', color='green', label='Best test RMSE', linewidth=1)
                
            ax.legend()
        plt.show()
