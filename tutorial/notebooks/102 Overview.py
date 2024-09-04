# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - Overview

# COMMAND ----------

# MAGIC %md
# MAGIC This document shows each of the classes and functions that exist in the Tangent Databricks Python package.  
# MAGIC You can also find the configurations and learn how to correctly use each class and function.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Import the tangent_works Python package and other support packages.

# COMMAND ----------

import tangent_works as tw
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC To run this notebook, a synthetic dataset is prepared below.

# COMMAND ----------

dataset = {'columns':['timestamp', 'target', 'predictor', 'label'],
'data':[
    ['2022-08-01 00:00:00', 7269, 338.81, 0],
    ['2022-08-01 01:00:00', 7049, 320.86, 0],
    ['2022-08-01 02:00:00', 7013, 329.72, 0],
    ['2022-08-01 03:00:00', 7292, 380.0, 0],
    ['2022-08-01 04:00:00', 7675, 429.66, 0],
    ['2022-08-01 05:00:00', 8299, 467.91, 0],
    ['2022-08-01 06:00:00', 8844, 474.9, 0],
    ['2022-08-01 07:00:00', 9253, 461.66, 0],
    ['2022-08-01 08:00:00', 9546, 446.72, 0],
    ['2022-08-01 09:00:00', 9808, 433.25, 0],
    ['2022-08-01 10:00:00', 9847, 385.88, 0],
    ['2022-08-01 11:00:00', 9719, 344.81, 0],
    ['2022-08-01 12:00:00', 9566, 310.97, 0],
    ['2022-08-01 13:00:00', 9584, 317.82, 0],
    ['2022-08-01 14:00:00', 9412, 344.65, 0],
    ['2022-08-01 15:00:00', 9375, 397.27, 0],
    ['2022-08-01 16:00:00', 9477, 421.24, 0],
    ['2022-08-01 17:00:00', 9279, 434.33, 0],
    ['2022-08-01 18:00:00', 8943, 473.33, 0],
    ['2022-08-01 19:00:00', 8663, 469.99, 0],
    ['2022-08-01 20:00:00', 8725, 475.62, 0],
    ['2022-08-01 21:00:00', 8487, 408.11, 0],
    ['2022-08-01 22:00:00', 7893, 440.98, 0],
    ['2022-08-01 23:00:00', 7540, 390.1, 1],
    ['2022-08-02 00:00:00', 7265, 382.1, 1],
    ['2022-08-02 01:00:00', 7110, 359.89, 0],
    ['2022-08-02 02:00:00', 7164, 352.69, 0],
    ['2022-08-02 03:00:00', 7358, 393.03, 0],
    ['2022-08-02 04:00:00', 7674, 457.04, 0],
    ['2022-08-02 05:00:00', 8279, 479.9, 1],
    ['2022-08-02 06:00:00', 8851, 478.9, 0],
    ['2022-08-02 07:00:00', 9333, 363.5, 0],
    ['2022-08-02 08:00:00', 9571, 331.7, 0],
    ['2022-08-02 09:00:00', 9658, 250.0, 0],
    ['2022-08-02 10:00:00', 9843, 75.46, 0],
    ['2022-08-02 11:00:00', 9923, 92.29, 0],
    ['2022-08-02 12:00:00', 9890, -37.56, 0],
    ['2022-08-02 13:00:00', 9689, -2.01, 0],
    ['2022-08-02 14:00:00', 9553, 179.0, 0],
    ['2022-08-02 15:00:00', 9757, 179.13, 0],
    ['2022-08-02 16:00:00', 9842, 299.1, 0],
    ['2022-08-02 17:00:00', 9611, 367.31, 0],
    ['2022-08-02 18:00:00', 9340, 329.44, 0],
    ['2022-08-02 19:00:00', 9138, 379.22, 0],
    ['2022-08-02 20:00:00', 9313, 386.8, 0],
    ['2022-08-02 21:00:00', 8871, 387.2, 0],
    ['2022-08-02 22:00:00', 8350, 244.6, 0],
    ['2022-08-02 23:00:00', 7919, 297.93, 0]
]}
tangent_dataframe = pd.DataFrame(dataset['data'],columns=dataset['columns'])

group_keys = []
timestamp_column = "timestamp"
target_column = "target"

predictors = [s for s in list(tangent_dataframe.columns) if s not in group_keys + [timestamp_column, target_column]]
tangent_dataframe = tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors].sort_values(by=group_keys + [timestamp_column]).reset_index(drop=True)
tangent_dataframe[timestamp_column] = pd.to_datetime(pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S"))

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Tangent

# COMMAND ----------

# MAGIC %md
# MAGIC This section will explain all the functionalities of the Python package that communicate with the Tangent Core.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 TimeSeries

# COMMAND ----------

# MAGIC %md
# MAGIC The TimeSeries class creates a time series object that is ready to be analyzed by Tangent.  
# MAGIC The purpose of this step is to validate the format of the data.  
# MAGIC It requires a pandas dataframe with time series data and the user can optionally describe which column is the timestamp column and which columns are the group_keys for panel data.

# COMMAND ----------

tw_timeseries = tw.TimeSeries(
    data = tangent_dataframe,
    # timestamp_column = timestamp_column,
    # group_key_columns = []
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC The first capability of Tangent is to build and use forecasting models.  
# MAGIC The following functions for forecasting are available in the Tangent Python package.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.1 configurations

# COMMAND ----------

# MAGIC %md
# MAGIC Tangent is designed to automate as much as possible in the modeling process.  
# MAGIC There are however configuration settings that you can apply.  
# MAGIC Below you will find the configuration settings for the different functions.  
# MAGIC When specific settings are not included, Tangent will assume default settings and decide automatically which settings to apply in the model building process.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.1.1 build_model

# COMMAND ----------

build_model_configuration = {
    # 'target_column': 'string',
    # 'categorical_columns': [
    #     'string'
    # ],
    # 'holiday_column': 'string',
    # 'prediction_from': {
    #     'base_unit': 'sample',
    #     'value': 1
    # },
    # 'prediction_to': {
    #     'base_unit': 'sample',
    #     'value': 24
    # },
    # 'target_offsets': 'combined',
    # 'predictor_offsets': 'common',
    # 'allow_offsets': True,
    # 'max_offsets_depth': 0,
    # 'normalization': True,
    # 'max_feature_count': 20,
    # 'transformations': [
    #     'exponential_moving_average',
    #     'rest_of_week',
    #     'periodic',
    #     'intercept',
    #     'piecewise_linear',
    #     'time_offsets',
    #     'polynomial',
    #     'identity',
    #     'simple_moving_average',
    #     'month',
    #     'trend',
    #     'day_of_week',
    #     'fourier',
    #     'public_holidays',
    #     'one_hot_encoding'
    # ],
    # 'daily_cycle': True,
    # 'confidence_level': 90,
    # 'data_alignment': [
    #     {
    #         'column_name': 'string',
    #         'timestamp': 'yyyy-mm-dd hh:mm:ssZ'
    #     }
    # ],
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.1.2 predict

# COMMAND ----------

predict_configuration = {
    # 'prediction_from': {
    #     'base_unit': 'sample',
    #     'value': 1
    #     },
    # 'prediction_to': {
    #     'base_unit': 'sample',
    #     'value': 1
    # }, 
    # 'prediction_boundaries': {
    #     'type': 'explicit',
    #     'max_value': 100,
    #     'min_value': 0
    # },
    # 'data_alignment': [
    #     {
    #         'column_name': 'string',
    #         'timestamp': 'yyyy-mm-dd hh:mm:ssZ'
    #     }
    # ],
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2.1.3 rca

# COMMAND ----------

forecasting_rca_configuration = [
    # 1
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.2 class

# COMMAND ----------

# MAGIC %md
# MAGIC The Forecasting class validates the build_model configuration and brings it together with the time series in one object.  
# MAGIC With the resulting object, models, predictions and other analysis can be made.

# COMMAND ----------

tw_forecasting = tw.Forecasting(
    time_series = tw_timeseries,
    configuration = build_model_configuration
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.3 build_model

# COMMAND ----------

# MAGIC %md
# MAGIC The build model function sends a job request to Tangent to build a forecasting model using the time series and configuration shared in the Forecasting object. The model is returned into the object.

# COMMAND ----------

tw_forecasting.build_model()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.4 forecast

# COMMAND ----------

# MAGIC %md
# MAGIC The forecast function uses the same Forecasting object as the build model.  
# MAGIC By default it will reuse the configuration and time series of the build_model.
# MAGIC The user can specify a predict_configuration to use the model and change the output.
# MAGIC The user can also add a new time series to apply a built model to new data.

# COMMAND ----------

tangent_predictions = tw_forecasting.forecast(
  # configuration = predict_configuration,
  # time_series = tw_timeseries,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.5 rca

# COMMAND ----------

# MAGIC %md
# MAGIC With Root Cause Analysis (RCA), the user can go in depth into the features created by Tangent.
# MAGIC The user can access this information using the rca function on the Forecasting object. 
# MAGIC The user can choose to extract the results from specific models in the model zoo if there are multiple.

# COMMAND ----------

tw_forecasting_rca = tw_forecasting.rca(
  # configuration = forecasting_rca_configuration
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.6 outputs

# COMMAND ----------

# MAGIC %md
# MAGIC The following outputs can be extracted from the Forecasting object. 
# MAGIC - result_table: which contains the predictions
# MAGIC - model: which can be analyzed for time series insights
# MAGIC - configuration: which contains the configuration used. This is helpful for tracing back how the model was built.
# MAGIC - time_series: which contains the time series sent to Tangent. This is helpful for tracing back how the model was built.
# MAGIC - rca_table: if RCA was applied, the results can be found here.

# COMMAND ----------

tw_forecasting_result_table = tw_forecasting.result_table
tw_forecasting_model = tw_forecasting.model.to_dict()
tw_forecasting_configuration = tw_forecasting.configuration.to_dict()
tw_forecasting_time_series = tw_forecasting.time_series
tw_forecasting_rca_table = tw_forecasting.rca_table

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 AutoForecasting

# COMMAND ----------

# MAGIC %md
# MAGIC AutoForecasting is an extended capability that builds on Forecasting.  
# MAGIC It combines the steps of preprocessing, model building and prediction into one capability.  
# MAGIC It helps the user with accelerating their timeseries analysis by simplying the process of setting up a forecast.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3.1 configuration

# COMMAND ----------

# MAGIC %md
# MAGIC Many configuration settings of Forecasting can be found again here in AutoForecasting.  
# MAGIC Additional preprocessing functionalities are added to the list of potential configuration settings.

# COMMAND ----------

auto_forecasting_configuration = {
    # 'preprocessing': {
        # 'training_rows': [{'from': '2021-01-20 00:00:00','to': '2021-10-31 23:00:00'}],
        # 'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
        # 'columns': [
            # 'string'
        # ],
        # 'imputation': {
            # 'common': {'type': 'linear','max_gap_length': 0},
            # 'individual': [{'column_name': 'string','value': {'type': 'linear','max_gap_length': 0}}]
        # },
        # 'time_scaling': {
        #     'time_scale': {'base_unit': 'hour','value': 1},
        #     'aggregations': {
        #         'common': 'mean',
        #         'individual': [
        #             {'column_name':'string','value':'mean'}
        #         ]
        #     },
        #     'drop_empty_rows': True
        # }
    # },
    # 'engine': {
        # 'target_column': target_column,
        # 'holiday_column': 'string',
        # 'prediction_from': {'base_unit': 'sample','value': 1},
        # 'prediction_to': {'base_unit': 'sample','value': 1},
        # 'target_offsets': 'combined',
        # 'predictor_offsets': 'common',
        # 'allow_offsets': True,
        # 'max_offsets_depth': 0,
        # 'normalization': True,
        # 'max_feature_count': 20,
        # 'transformations': [
        #     'exponential_moving_average',
        #     'rest_of_week',
        #     'periodic',
        #     'intercept',
        #     'piecewise_linear',
        #     'time_offsets',
        #     'polynomial',
        #     'identity',
        #     'simple_moving_average',
        #     'month',
        #     'trend',
        #     'day_of_week',
        #     'fourier',
        #     'public_holidays',
        #     'one_hot_encoding'
        # ],
        # 'daily_cycle': True,
        # 'confidence_level': 90,
        # 'data_alignment': [
        #     {'column_name': 'string','timestamp': 'yyyy-mm-dd hh:mm:ssZ'}
        # ],
        # 'prediction_boundaries': {
        #     'type': 'explicit',
        #     'max_value': 100,
        #     'min_value': 0
        # }
    # }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3.2 class

# COMMAND ----------

# MAGIC %md
# MAGIC The AutoForecasting class is also activated with a Tangent time series and a configuration.  
# MAGIC This validates the right setup for running a autoforecasting job.

# COMMAND ----------

tw_autoforecasting = tw.AutoForecasting(
    time_series = tw_timeseries,
    configuration = auto_forecasting_configuration
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3.3 run

# COMMAND ----------

# MAGIC %md
# MAGIC To preprocess the data, build a model and generate predictions in one step, the following function can be executed.

# COMMAND ----------

tw_autoforecasting.run()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3.4 rca

# COMMAND ----------

# MAGIC %md
# MAGIC With Root Cause Analysis (RCA), the user can go in depth into the features created by Tangent.
# MAGIC The user can access this information using the rca function on the AutoForecasting object. 
# MAGIC The user can choose to extract the results from specific models in the model zoo if there are multiple.

# COMMAND ----------

tw_autoforecasting_rca = tw_autoforecasting.rca()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3.5 outputs

# COMMAND ----------

# MAGIC %md
# MAGIC The following outputs can be extracted from the AutoForecasting object. 
# MAGIC - result_table: which contains the predictions
# MAGIC - model: which can be analyzed for time series insights
# MAGIC - configuration: which contains the configuration used. This is helpful for tracing back how the model was built.
# MAGIC - time_series: which contains the time series sent to Tangent. This is helpful for tracing back how the model was built.
# MAGIC - rca_table: if RCA was applied, the results can be found here.

# COMMAND ----------

tw_autoforecasting_result_table = tw_autoforecasting.result_table
tw_autoforecasting_model = tw_autoforecasting.model
tw_autoforecasting_configuration = tw_autoforecasting.configuration
tw_autoforecasting_time_series = tw_autoforecasting.time_series
tw_autoforecasting_rca_table = tw_autoforecasting.rca_table

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 AnomalyDetection

# COMMAND ----------

# MAGIC %md
# MAGIC The another capability of Tangent is to build and use anomaly detection models.  
# MAGIC The following functions for detection are available in the Tangent Python package.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4.1 configurations

# COMMAND ----------

# MAGIC %md
# MAGIC The anomaly detection process happens in two stages. First, a normal behavior model is built using Tangent.  
# MAGIC This model is then compared to the original time series and detection layers are calculated from different perspectives that indicate how anomalous certain timestamps are.
# MAGIC For both stages, the configuration settings can be modified with the dictionary below.

# COMMAND ----------

build_anomaly_detection_configuration = {
    'normal_behavior':{
        # 'target_column':'str',
        # 'holiday_column:':'str',
        # 'target_offsets':'combined',
        # 'allow_offsets':True,
        # 'offset_limit':0,
        # 'normalization':True,
        # 'max_feature_count':20,
        # 'transformations': [
        #     'exponential_moving_average',
        #     'rest_of_week',
        #     'periodic',
        #     'intercept',
        #     'piecewise_linear',
        #     'time_offsets',
        #     'polynomial',
        #     'identity',
        #     'simple_moving_average',
        #     'month',
        #     'trend',
        #     'day_of_week',
        #     'fourier',
        #     'public_holidays',
        #     'one_hot_encoding'
        # ],    
        # 'daily_cycle':True,
        # 'confidence_level':90,
        # 'categorical_columns':[
        #     'str'
        # ],
        # 'data_alignment': [
        #     {
        #         'column_name': 'string',
        #         'timestamp': 'yyyy-mm-dd hh:mm:ssZ'
        #     }
        # ],
    },
    # 'detection_layers': [
    #     {
    #         'residuals_transformation':{
    #             'type':'residuals'
    #         },
    #         'sensitivity':0.3
    #     },
    #     {
    #         'residuals_transformation':{
    #             'type':'residuals_change',
    #             'window_length':2
    #         },
    #         'sensitivity':0.3
    #     },
    #     {
    #         'residuals_transformation':{
    #             'type':'moving_average',
    #             'window_length':1
    #         },
    #         'sensitivity':0.3
    #     },
    #     {
    #         'residuals_transformation':{
    #             'type':'moving_average_change',
    #             'window_lengths':[
    #                 2,
    #                 1
    #             ]
    #         },
    #         'sensitivity':0.3
    #     },
    #     {
    #         'residuals_transformation':{
    #             'type':'standard_deviation',
    #             'window_length':1
    #         },
    #         'sensitivity':0.3
    #     },
    #     {
    #         'residuals_transformation':{
    #             'type':'standard_deviation_change',
    #             'window_lengths':[
    #                 2,
    #                 1
    #             ]
    #         },
    #         'sensitivity':0.3
    #     },
    # ]
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4.2 class

# COMMAND ----------

# MAGIC %md
# MAGIC The AnomalyDetection class validates the build_model configuration and brings it together with the time series in one object.  
# MAGIC With the resulting object, models, detections and other analysis can be made.

# COMMAND ----------

tw_anomaly_detection = tw.AnomalyDetection(
    time_series = tw_timeseries,
    configuration = build_anomaly_detection_configuration
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4.3 build_model

# COMMAND ----------

# MAGIC %md
# MAGIC The build model function sends a job request to Tangent to build a anomaly detection model using the time series and configuration shared in the AnomalyDetection object. The model is returned into the object.

# COMMAND ----------

tw_anomaly_detection.build_model()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4.4 detect

# COMMAND ----------

# MAGIC %md
# MAGIC The detect function uses the same AnomalyDetection object as the build model.  
# MAGIC By default it will reuse the configuration and time series of the build_model.  
# MAGIC The user can add a new time series to apply a built model to new data.

# COMMAND ----------

detect_df = tw_anomaly_detection.detect(
    time_series = tw_timeseries
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4.5 rca

# COMMAND ----------

# MAGIC %md
# MAGIC With Root Cause Analysis (RCA), the user can go in depth into the features created by Tangent.
# MAGIC The user can access this information using the rca function on the AnomalyDetection object. 
# MAGIC The user can choose to extract the results from specific models in the model zoo if there are multiple.

# COMMAND ----------

tw_anomaly_detection_rca = tw_anomaly_detection.rca(
  # configuration = [1]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4.6 outputs

# COMMAND ----------

# MAGIC %md
# MAGIC The following outputs can be extracted from the AnomalyDetection object. 
# MAGIC - result_table: which contains the detections
# MAGIC - model: which can be analyzed for time series insights
# MAGIC - configuration: which contains the configuration used. This is helpful for tracing back how the model was built.
# MAGIC - time_series: which contains the time series sent to Tangent. This is helpful for tracing back how the model was built.
# MAGIC - rca_table: if RCA was applied, the results can be found here.

# COMMAND ----------

tw_anomaly_detection_result_table = tw_anomaly_detection.result_table
tw_anomaly_detection_model = tw_anomaly_detection.model
tw_anomaly_detection_configuration = tw_anomaly_detection.configuration
tw_anomaly_detection_time_series = tw_anomaly_detection.time_series
tw_anomaly_detection_rca_table = tw_anomaly_detection.rca_table

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. PostProcessing

# COMMAND ----------

# MAGIC %md
# MAGIC This section will explain the different post processing capabilities to facilitate getting insights from Tangent results.

# COMMAND ----------

tw_post_processing = tw.PostProcessing()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Properties

# COMMAND ----------

# MAGIC %md
# MAGIC The properties function receives a tangent model and transforms the insights into a Pandas dataframe.  
# MAGIC With this, the user can quickly identify which predictors were used by Tangent in the model building process.

# COMMAND ----------

tw_properties = tw_post_processing.properties(response=tw_forecasting_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Features

# COMMAND ----------

# MAGIC %md
# MAGIC The features function receives a tangent model and transforms the insights into a Pandas dataframe.  
# MAGIC With this, the user can quickly identify which features were generated by Tangent in the model building process.

# COMMAND ----------

tw_features = tw_post_processing.features(response=tw_forecasting_model)

# COMMAND ----------


