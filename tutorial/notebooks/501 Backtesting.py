# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - Backtesting

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Setup

# COMMAND ----------

import tangent_works as tw
import pandas as pd
import numpy as np

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

visualization.data(df=tangent_dataframe,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Tangent

# COMMAND ----------

time_series = tw.TimeSeries(tangent_dataframe)

# COMMAND ----------

def run_auto_forecast(time_series,configuration):
    forecasting = tw.AutoForecasting(time_series=time_series, configuration=configuration)
    forecasting.run()
    model = forecasting.model.to_dict()
    properties = tw.PostProcessing().properties(model=model)
    table = tw.PostProcessing().result_table(forecasting=forecasting)
    return table,properties

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Forecast

# COMMAND ----------

auto_forecasting_configuration = {
    'preprocessing': {
        # 'training_rows': [{'from': '2021-01-20 00:00:00','to': '2021-10-31 23:00:00'}],
        # 'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
    },
    'engine': {
        'prediction_from': {'base_unit': 'day','value': 1},
        'prediction_to': {'base_unit': 'day','value': 1},
    }
}

# COMMAND ----------

production_forecast_results_table,production_forecast_properties = run_auto_forecast(time_series=time_series,configuration=auto_forecasting_configuration)
production_forecast_properties['id']='production_forecast'

# COMMAND ----------

visualization.predictions(production_forecast_results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Production Forecast & Training Results

# COMMAND ----------

auto_forecasting_configuration = {
    'preprocessing': {
        # 'training_rows': [{'from': '2021-01-20 00:00:00','to': '2021-10-31 23:00:00'}],
        'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
    },
    'engine': {
        'prediction_from': {'base_unit': 'day','value': 1},
        'prediction_to': {'base_unit': 'day','value': 1},
    }
}

# COMMAND ----------

production_training_results_table,production_training_properties = run_auto_forecast(time_series=time_series,configuration=auto_forecasting_configuration)
production_training_properties['id']='production_training'

# COMMAND ----------

visualization.predictions(production_training_results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backtest

# COMMAND ----------

auto_forecasting_configuration = {
    'preprocessing': {
        'training_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-15 23:00:00'}],
        'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
    },
    'engine': {
        'prediction_from': {'base_unit': 'day','value': 1},
        'prediction_to': {'base_unit': 'day','value': 1},
    }
}

# COMMAND ----------

backtest_results_table,backtest_properties = run_auto_forecast(time_series=time_series,configuration=auto_forecasting_configuration)
backtest_properties['id']='backtest'

# COMMAND ----------

visualization.predictions(backtest_results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Long Backtest

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

long_backtest_results_table,long_backtest_properties = run_auto_forecast(time_series=time_series,configuration=auto_forecasting_configuration)
long_backtest_properties['id']='long_backtest'

# COMMAND ----------

visualization.predictions(long_backtest_results_table)

# COMMAND ----------

# MAGIC %md
# MAGIC #3. Comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predictions

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

tangent_properties_df = pd.concat([production_forecast_properties,production_training_properties,backtest_properties,long_backtest_properties])
fig = px.bar(tangent_properties_df[tangent_properties_df['importance']>0], x='rel_importance', y="id", color="name", barmode = 'stack',orientation='h')
fig.update_layout(height=500, width=1200, title_text="Properties")
fig.show()
