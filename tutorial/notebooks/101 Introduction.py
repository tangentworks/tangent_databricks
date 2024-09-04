# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC This tutorial covers the use of Tangent Databricks. This runtime version of Tangent runs on your Databricks cluster and is accessed with a built-in Python package.  
# MAGIC In this tutorial you will learn how to use the Python package to access Tangent's capabilities in Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC # Tangent

# COMMAND ----------

# MAGIC %md
# MAGIC First, some more information about Tangent.  
# MAGIC Tangent is a lightweight time series model building engine that allows you to build unique timeseries models in a fraction of the time compared to conventional methods.  
# MAGIC It automates the complete process from raw input data to predictions and therefore serves the user by accelerating their work in time series analytics.  
# MAGIC The technology is built around a unique and proprietary model building technique that is designed to find predictive value in time series data through efficient feature engineering.  
# MAGIC
# MAGIC For the user this means they can focus on bringing together useful input data to run through Tangent, and use the resulting predictions and models for solving their use cases and generating insights.  
# MAGIC The Tangent core capabilities have been packaged inside a Docker container. This container can then be installed on your Databricks cluster allowing you to leverage the power of Databricks and Tangent together.
# MAGIC
# MAGIC To learn more about the inner workings of Tangent, you can find more information in the general documentation here: __#TODO__

# COMMAND ----------

# MAGIC %md
# MAGIC # Requirements

# COMMAND ----------

# MAGIC %md
# MAGIC Make sure you have a cluster running in Databricks with the configuration as specified in the installation document shared with this tutorial.

# COMMAND ----------

# MAGIC %md
# MAGIC # Getting started

# COMMAND ----------

# MAGIC %md
# MAGIC Let's test if the Tangent is running by trying out a simple example with synthetic data.  
# MAGIC First import the __tangent_works__ package as tw. Import Pandas as well to manage the the input data.

# COMMAND ----------

import tangent_works as tw
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC Create the synthetic dataset and form a Pandas dataframe to use as example input data.

# COMMAND ----------

tangent_dataframe = pd.DataFrame(
    [
        {'timestamp': '2022-08-01 00:00:00', 'target': 7269, 'predictor': 338.810,'label': 0},
        {'timestamp': '2022-08-01 01:00:00', 'target': 7049, 'predictor': 320.86, 'label': 0},
        {'timestamp': '2022-08-01 02:00:00', 'target': 7013, 'predictor': 329.72, 'label': 0},
        {'timestamp': '2022-08-01 03:00:00', 'target': 7292, 'predictor': 380.00, 'label': 0},
        {'timestamp': '2022-08-01 04:00:00', 'target': 7675, 'predictor': 429.66, 'label': 0},
        {'timestamp': '2022-08-01 05:00:00', 'target': 8299, 'predictor': 467.91, 'label': 0},
        {'timestamp': '2022-08-01 06:00:00', 'target': 8844, 'predictor': 474.90, 'label': 0},
        {'timestamp': '2022-08-01 07:00:00', 'target': 9253, 'predictor': 461.66, 'label': 0},
        {'timestamp': '2022-08-01 08:00:00', 'target': 9546, 'predictor': 446.72, 'label': 0},
        {'timestamp': '2022-08-01 09:00:00', 'target': 9808, 'predictor': 433.25, 'label': 0},
        {'timestamp': '2022-08-01 10:00:00', 'target': 9847, 'predictor': 385.88, 'label': 0},
        {'timestamp': '2022-08-01 11:00:00', 'target': 9719, 'predictor': 344.81, 'label': 0},
        {'timestamp': '2022-08-01 12:00:00', 'target': 9566, 'predictor': 310.97, 'label': 0},
        {'timestamp': '2022-08-01 13:00:00', 'target': 9584, 'predictor': 317.82, 'label': 0},
        {'timestamp': '2022-08-01 14:00:00', 'target': 9412, 'predictor': 344.65, 'label': 0},
        {'timestamp': '2022-08-01 15:00:00', 'target': 9375, 'predictor': 397.27, 'label': 0},
        {'timestamp': '2022-08-01 16:00:00', 'target': 9477, 'predictor': 421.24, 'label': 0},
        {'timestamp': '2022-08-01 17:00:00', 'target': 9279, 'predictor': 434.33, 'label': 0},
        {'timestamp': '2022-08-01 18:00:00', 'target': 8943, 'predictor': 473.33, 'label': 0},
        {'timestamp': '2022-08-01 19:00:00', 'target': 8663, 'predictor': 469.99, 'label': 0},
        {'timestamp': '2022-08-01 20:00:00', 'target': 8725, 'predictor': 475.62, 'label': 0},
        {'timestamp': '2022-08-01 21:00:00', 'target': 8487, 'predictor': 408.11, 'label': 0},
        {'timestamp': '2022-08-01 22:00:00', 'target': 7893, 'predictor': 440.98, 'label': 0},
        {'timestamp': '2022-08-01 23:00:00', 'target': 7540, 'predictor': 390.10, 'label': 1},
     ]
)

group_keys = []
timestamp_column = "timestamp"
target_column = "target"

predictors = [s for s in list(tangent_dataframe.columns) if s not in group_keys + [timestamp_column, target_column]]
tangent_dataframe = tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors].sort_values(by=group_keys + [timestamp_column]).reset_index(drop=True)
tangent_dataframe[timestamp_column] = pd.to_datetime(pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S"))

# COMMAND ----------

# MAGIC %md
# MAGIC The next step is to validate the time series data. To make sure the engine will be able to make calculations, we need a Pandas dataframe in the right format.  
# MAGIC Typically, you would organise your data by placing the timestamp values in the first column, in the second column the values that you want to model and all following columns can be any value that you believe might have predictive value and should be analyzed by Tangent.  
# MAGIC Make sure the timestamp column is in 'datetime' format.

# COMMAND ----------

tw_timeseries = tw.TimeSeries(data=tangent_dataframe,timestamp_column='timestamp')

# COMMAND ----------

# MAGIC %md
# MAGIC Tangent works by combining a dataset and a configuration, telling the engine what to do with the dataset, together to generate results.  
# MAGIC The next step is to bring both together in an object to validate the setup of the experiment with Tangent. Here we will use the Autoforecasting capabilities and use the default configuration settings by leaving the configuration empty.  
# MAGIC By not adding any specific configuration settings, Tangent will decide based on the data, which settings are best applied.  
# MAGIC Tangent is designed to automated as much as possible and by using default settings, the user can let Tangent make data driven decisions to come to the best results.

# COMMAND ----------


tw_autoforecasting = tw.AutoForecasting(
    time_series = tw_timeseries,
    # configuration = {}
)

# COMMAND ----------

# MAGIC %md
# MAGIC With everything set up correctly, the user can send a "run" request to Tangent to start calculations and build the model and predictions.  
# MAGIC Depending on the configuration and the size of the dataset, typical jobs take mere seconds to a couple of minutes at most to complete.

# COMMAND ----------

tw_autoforecasting.run()

# COMMAND ----------

# MAGIC %md
# MAGIC The model is now finished and the predictions are immediatly calculated. We can collect the results from the autoforecasting object with the following code snippets.

# COMMAND ----------

tangent_results_table = tw_autoforecasting.result_table
tangent_auto_forecast_model = tw_autoforecasting.model.to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC The predictions are returned as a Pandas dataframe

# COMMAND ----------

tangent_results_table

# COMMAND ----------

# MAGIC %md
# MAGIC The model is returned as a JSON.

# COMMAND ----------

tangent_auto_forecast_model

# COMMAND ----------

# MAGIC %md
# MAGIC This example shows the basic process for getting to results with Tangent. From here, there are many more capabilities of Tangent that can be leveraged with the Python Package which you can find in the next section.

# COMMAND ----------

# MAGIC %md
# MAGIC # Table of Contents

# COMMAND ----------

# MAGIC %md
# MAGIC In this overview you will find the following notebooks that will guide you through the use of Tangent and all its capabilities.
# MAGIC - __101 Introduction__:&emsp;&emsp;&emsp;&emsp;&emsp;Learn about Tangent and get started with this tutorial.
# MAGIC - __102 Overview__:&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; Learn which classes and functions exist in the Python package and find example configurations.
# MAGIC - __201 Forecasting__:&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; Try out this template to build a Tangent Forecasting model with example data.
# MAGIC - __202 AnomalyDetection__:&emsp;&emsp;&nbsp;Try out this template to build a Tangent AnomalyDetection model with example data.
# MAGIC - __203 AutoForecasting__:&emsp;&emsp;&emsp;&nbsp;Try out this template to build a Tangent AutoForecast with example data.
# MAGIC
