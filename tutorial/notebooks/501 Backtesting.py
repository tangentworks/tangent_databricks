# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - Backtesting

# COMMAND ----------

# MAGIC %md
# MAGIC In this tutorial, you will learn to properly backtest using Tangent. Autoforecasting will be used for this.  
# MAGIC Backtesting is instrumental in time series analytics. It allows the user to gain insights in the performance of their forecasting models and validate the value of the forecast. This tutorial will showcase the use of the different backtesting settings and how to apply them in different forecasting situations.
# MAGIC
# MAGIC To show the backtesting capabilities in autoforecasting we will use an example dataset from a energy forecasting use case.  
# MAGIC The goal is to validate a forecasting model that generates day ahead predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Setup

# COMMAND ----------

# MAGIC %md
# MAGIC First, import the tangent_works package and other supporting libraries.

# COMMAND ----------

import tangent_works as tw
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC To visualize the results of this exercise, the following visualization functions can be used.

# COMMAND ----------

# -------------------------------- Supporting Libraries --------------------------------

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as splt

class visualization:

    def predictions(df):
        fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        color_map = {'training':'green','testing':'red','production':'goldenrod'}
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['target'], name='target',line=dict(color='black')), row=1, col=1)
        for forecasting_type in df['type'].unique():
            v_data = df[df['type']==forecasting_type].copy()
            fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['forecast'], name=forecasting_type,line=dict(color=color_map[forecasting_type])), row=1, col=1)
        fig.update_layout(height=500, width=1000, title_text="Results")
        fig.show()

    def data(df,timestamp,target,predictors):
        fig = splt.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=df[timestamp], y=df[target], name=target,connectgaps=True), row=1, col=1)
        for idx, p in enumerate(predictors): fig.add_trace(go.Scatter(x=df[timestamp], y=df[p], name=p,connectgaps=True), row=2, col=1)
        fig.update_layout(height=600, width=1100, title_text="Data visualization")
        fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #1. Data

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset that will be used in this notebook is called belgian_electricity_grid.  
# MAGIC It contains historical electricity consumption and weather data as well as weather forecasts and public holiday information.  
# MAGIC In the cell below, this dataset is preprocessed and made ready for use with Tangent.

# COMMAND ----------

file_path = '/Workspace'+dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 2)[0]+'/data/belgian_electricity_grid.csv'
tangent_dataframe = pd.read_csv(file_path)
group_keys = []
timestamp_column = "Timestamp"
target_column = "Quantity"
predictors = [s for s in list(tangent_dataframe.columns) if s not in group_keys + [timestamp_column, target_column]]
tangent_dataframe = tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors].sort_values(by=group_keys + [timestamp_column]).reset_index(drop=True)
tangent_dataframe[timestamp_column] = pd.to_datetime(pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S"))
tangent_dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC In time series analysis backtesting, when exploring a dataset, it is best practice to visualize the data and learn which patterns might exists in the data that we want Tangent to identify automatically.  
# MAGIC In this graph, the target column "Quantity" is visualized above and the additional explanatory variables or predictors are visualized below.  
# MAGIC Notice that for some predictors, values are available ahead of the last target timestamp throughout the forecast horizon. 

# COMMAND ----------

visualization.data(df=tangent_dataframe,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Tangent

# COMMAND ----------

# MAGIC %md
# MAGIC Below, we combine all the steps in the autoforecasting process to focus this notebook on the process of backtesting. This user defined function then returns the predictions and the properties for later comparison.

# COMMAND ----------

class user_defined:
    def run_auto_forecast(time_series,configuration):
        forecasting = tw.AutoForecasting(time_series=time_series, configuration=configuration)
        forecasting.run()
        model = forecasting.model.to_dict()
        properties = tw.PostProcessing().properties(model=model)
        table = tw.PostProcessing().result_table(forecasting=forecasting)
        return table,properties

# COMMAND ----------

# MAGIC %md
# MAGIC First, we need to create a time series object from the dataset.

# COMMAND ----------

time_series = tw.TimeSeries(tangent_dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, the following situations are outlined:
# MAGIC - Production Forecast:
# MAGIC   - This settings shows how to configure a Tangent autoforecasting job for a production scenario that only focusses on building a model and receiving back the production forecast results.
# MAGIC - Production Forecast & Training:
# MAGIC   - This settings shows the same process as the one above however now, in sample training results are visualized as well.
# MAGIC - Backtest:
# MAGIC   - This example shows how to do a simple backtest with Tangent autoforecasting.
# MAGIC - Long Backtest
# MAGIC   - This example is the same scenario as the one above however now, a different training horizon is selected to show the impact of including more or less historical training data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Forecast

# COMMAND ----------

# MAGIC %md
# MAGIC In order to generate a production forecast and only expect the predicted values in the forecast horizon to be returned, the user simply needs to leave the training_rows and prediction_rows under the preprocessing key blank.  
# MAGIC Autoforecasting is designed to give the Tangent user a way to get to predictions as soon as possible. These row settings are there for specify the rows to use for training and the rows for which to calculate predictions over.

# COMMAND ----------

auto_forecasting_configuration = {
    'preprocessing': {
        # 'training_rows': [{'from': '2021-01-20 00:00:00','to': '2021-10-31 23:00:00'}],
        # 'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
    },
    'engine': {
        'prediction_from': {'base_unit': 'sample','value': 1},
        'prediction_to': {'base_unit': 'sample','value': 24},
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC Run the user defined function to generate a production forecast.

# COMMAND ----------

production_forecast_results_table,production_forecast_properties = user_defined.run_auto_forecast(time_series=time_series,configuration=auto_forecasting_configuration)
production_forecast_properties['id']='production_forecast'

# COMMAND ----------

# MAGIC %md
# MAGIC In the graph & table below, you will see that only the production forecasts are available.

# COMMAND ----------

production_forecast_results_table

# COMMAND ----------

visualization.predictions(production_forecast_results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Forecast & Training Results

# COMMAND ----------

# MAGIC %md
# MAGIC In order to understand if a production forecast makes sense, we can ask Tangent to show insample training results.
# MAGIC In case the training_rows are left blank, all historical data will be use for training, just as is the case in the previous exercise. Only now, the values that are included in the prediction_rows will also be returned. This does not affect model building. 

# COMMAND ----------

auto_forecasting_configuration = {
    'preprocessing': {
        # 'training_rows': [{'from': '2021-01-20 00:00:00','to': '2021-10-31 23:00:00'}],
        'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
    },
    'engine': {
        'prediction_from': {'base_unit': 'sample','value': 1},
        'prediction_to': {'base_unit': 'sample','value': 24},
    }
}

# COMMAND ----------

production_training_results_table,production_training_properties = user_defined.run_auto_forecast(time_series=time_series,configuration=auto_forecasting_configuration)
production_training_properties['id']='production_training'

# COMMAND ----------

# MAGIC %md
# MAGIC The graph and table below show that training results are now also available. The production results should be the same as in the previous exercise. These training results can now be used to get a sense of the quality of the fit of the model and the predictions. As the results are matching quite closely with the historical target values, we can assume the production results will also quite closely match with future values. However to be sure, we need to either wait for new values to become available, or run a backtest.

# COMMAND ----------

production_training_results_table

# COMMAND ----------

visualization.predictions(production_training_results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backtest

# COMMAND ----------

# MAGIC %md
# MAGIC To run a backtest, the user can choose to exclude certain parts of the dataset from the model building process. These rows to exclude can be describe with the training_rows setting. This defines the training - testing split.
# MAGIC
# MAGIC From there, the prediction_rows setting could still cover the entire dataset. The result is that from all predictions that are generated:
# MAGIC - those covered by the training rows are indicated with type: __"training"__.
# MAGIC - those not covered by training and not in the forecasting horizon beyond the last target value are indicated with type __"testing"__.
# MAGIC - those predictions in the forecasting horizon are indicated with type: __"production"__.
# MAGIC
# MAGIC Below the cutoff between training and testing is set at "2022-01-15 23:00:00". This is near the end of the dataset meaning we will only have 1 day of testing results.

# COMMAND ----------

auto_forecasting_configuration = {
    'preprocessing': {
        'training_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-15 23:00:00'}],
        'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
    },
    'engine': {
        'prediction_from': {'base_unit': 'sample','value': 1},
        'prediction_to': {'base_unit': 'sample','value': 24},
    }
}

# COMMAND ----------

backtest_results_table,backtest_properties = user_defined.run_auto_forecast(time_series=time_series,configuration=auto_forecasting_configuration)
backtest_properties['id']='backtest'

# COMMAND ----------

# MAGIC %md
# MAGIC The graph and table below show the different types of forecasted values. Looking at the red values in the graph near the end, we learn that Tangent's testing results also quite closely match the actual target values meaning we can likely trust that the production forecasts will be of high quality.

# COMMAND ----------

backtest_results_table

# COMMAND ----------

visualization.predictions(backtest_results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Long Backtest

# COMMAND ----------

# MAGIC %md
# MAGIC Generally, you would like to backtest on a larger subset of the data to make sure the model performs well on a longer timeframe. The longer the testing horizon however means that less data can be used for training. This trade-off is important to rememeber when backtesting because there could be meaningfull information in recent data points that with a long backtest would not be included in the training process and therefore result in a skewed view on the accuracy. 
# MAGIC It is important that for Tangent, backtesting is an indication of the performance. However to really validate the quality of results by Tangent a simulation needs to be done. This is covered in tutorial 502 - Simulation.
# MAGIC
# MAGIC Below, we will show the impact of a longer testing period. Now the train-test cutoff will be set at "2021-10-31 23:00:00".

# COMMAND ----------

auto_forecasting_configuration = {
    'preprocessing': {
        'training_rows': [{'from': '2021-01-20 00:00:00','to': '2021-10-31 23:00:00'}],
        'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
    },
    'engine': {
        'prediction_from': {'base_unit': 'day','value': 1},
        'prediction_to': {'base_unit': 'day','value': 1},
    }
}

# COMMAND ----------

long_backtest_results_table,long_backtest_properties = user_defined.run_auto_forecast(time_series=time_series,configuration=auto_forecasting_configuration)
long_backtest_properties['id']='long_backtest'

# COMMAND ----------

# MAGIC %md
# MAGIC Now, the table and graph below show many more testing forecasts in red. These still cover the actuals quite well meaning we have a stable forecasting model that could be reused over an extended period of time.

# COMMAND ----------

long_backtest_results_table

# COMMAND ----------

visualization.predictions(long_backtest_results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC #3. Comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predictions

# COMMAND ----------

# MAGIC %md
# MAGIC The comparison of predictions below shows how both the production forecasts from the first two exercises are exactly the same. This shows that the prediction_rows setting doesn't impact model building, only prediction.   
# MAGIC
# MAGIC Also we learn that the production forecast of the short backtest exercise is more closely aligned with the first two exercises. This is logical because almost all the same data was used. Only 1 day of additional data was used in the first two situations, which already led tangent to learn new information and although minor, use different features in the model. 
# MAGIC
# MAGIC The long backtest shows a bigger difference, which highlights the importance of model rebuilding. In that last portion of data, quite some information can still be extracted. That is why Tangent is mostly used in a continuous rebuilding process. 

# COMMAND ----------

fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=tangent_dataframe[timestamp_column], y=tangent_dataframe[target_column], name='target',line=dict(color='black')), row=1, col=1)
v_data = production_forecast_results_table[production_forecast_results_table['type']=='production']
fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['forecast'], name='production_forecast',line=dict(color='green')), row=1, col=1)
v_data = production_training_results_table[production_training_results_table['type']=='production']
fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['forecast'], name='production_training',line=dict(color='goldenrod')), row=1, col=1)
v_data = backtest_results_table[backtest_results_table['type']=='production']
fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['forecast'], name='backtest',line=dict(color='blue')), row=1, col=1)
v_data = long_backtest_results_table[long_backtest_results_table['type']=='production']
fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['forecast'], name='long_backtest',line=dict(color='red')), row=1, col=1)
fig.update_layout(height=500, width=1000, title_text="Results")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Properties

# COMMAND ----------

# MAGIC %md
# MAGIC By comparing the properties from the different exercise we can draw the same conclusions.

# COMMAND ----------

tangent_properties_df = pd.concat([production_forecast_properties,production_training_properties,backtest_properties,long_backtest_properties])
fig = px.bar(tangent_properties_df[tangent_properties_df['importance']>0], x='rel_importance', y="id", color="name", barmode = 'stack',orientation='h')
fig.update_layout(height=500, width=1200, title_text="Properties")
fig.show()
