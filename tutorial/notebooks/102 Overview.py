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

import tangent_works
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
    ['2022-08-01 12:00:00', 9566, 310.97, ],
    ['2022-08-01 13:00:00', 9584, 317.82, ],
    ['2022-08-01 14:00:00', 9412, 344.65, ],
    ['2022-08-01 15:00:00', 9375, 397.27, ],
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
# MAGIC To access all functionalities, active the package such as below.

# COMMAND ----------

tw = tangent_works.TangentWorks()

# COMMAND ----------

# MAGIC %md
# MAGIC The package is divide into 3 subclasses with each their respective methods:
# MAGIC - forecasting 
# MAGIC   - build_model
# MAGIC   - predict
# MAGIC   - rca
# MAGIC   - auto_forecast
# MAGIC - anomaly Detection
# MAGIC   - build_model
# MAGIC   - detect
# MAGIC   - rca
# MAGIC - insights
# MAGIC   - properties
# MAGIC   - features

# COMMAND ----------

# MAGIC %md
# MAGIC Tangent is designed to automate as much as possible in the modeling process.  
# MAGIC There are however configuration settings that you can apply.  
# MAGIC For each method you will find example configuration settings with all possible parameters..  
# MAGIC When specific parameters are not set, Tangent will assume default settings and decide automatically which settings to apply in the process.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC The first capability of Tangent is to build and use forecasting models.  
# MAGIC The following functions for forecasting are available in the Tangent Python package.
# MAGIC - build_model
# MAGIC - predict
# MAGIC - rca
# MAGIC - auto_forecast

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.1 build_model

# COMMAND ----------

# MAGIC %md
# MAGIC The build model function sends a job request to Tangent to build a forecasting model with a prepared time series and configuration. This method returns a Tangent forecasting model.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configuration

# COMMAND ----------

fc_build_model_configuration = {
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
# MAGIC #### Usage

# COMMAND ----------

tw_fc_model = tw.forecasting.build_model(
    configuration = fc_build_model_configuration,
    dataset = tangent_dataframe
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Output

# COMMAND ----------

tw_fc_model

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.2 predict

# COMMAND ----------

# MAGIC %md
# MAGIC The predict function applies data to a Tangent forecasting model to generate predicted values. 
# MAGIC The user can specify a predict_configuration and a dataset. Both should correspond with the configuration used during the model building process.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configuration

# COMMAND ----------

fc_predict_configuration = {
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
# MAGIC #### Usage

# COMMAND ----------

tw_fc_predictions = tw.forecasting.predict(
    configuration = fc_predict_configuration,
    dataset = tangent_dataframe,
    model = tw_fc_model
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Output

# COMMAND ----------

tw_fc_predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.3 rca

# COMMAND ----------

# MAGIC %md
# MAGIC With Root Cause Analysis (RCA), the user can go in depth into the features created by Tangent.
# MAGIC The user can access this information using the rca function on the Forecasting object. 
# MAGIC The user can choose to extract the results from specific models in the model zoo if there are multiple.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC Many configuration settings of Forecasting can be found again here in AutoForecasting.  
# MAGIC Additional preprocessing functionalities are added to the list of potential configuration settings.

# COMMAND ----------

fc_rca_configuration = {
    'model_indexes':[
    ]
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Usage

# COMMAND ----------

tw_forecasting_rca = tw.forecasting.rca(
    dataset=tangent_dataframe,
    model = tw_fc_model,
    configuration = fc_rca_configuration
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Output

# COMMAND ----------

tw_forecasting_rca

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1.4 auto_forecast

# COMMAND ----------

# MAGIC %md
# MAGIC AutoForecasting is an extended capability that builds on forecasting.  
# MAGIC It combines the steps of preprocessing, model building and prediction into one capability.  
# MAGIC It helps the user with accelerating their timeseries analysis by simplifying the process of setting up a forecast.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configuration

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
# MAGIC #### Usage

# COMMAND ----------

tw_fc_auto_forecast = tw.forecasting.auto_forecast(
    configuration = auto_forecasting_configuration,
    dataset = tangent_dataframe
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Output

# COMMAND ----------

tw_fc_auto_forecast_model = tw_fc_auto_forecast.model
tw_fc_auto_forecast_model

# COMMAND ----------

tw_fc_auto_forecast_predictions = tw_fc_auto_forecast.predictions
tw_fc_auto_forecast_predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Anomaly Detection

# COMMAND ----------

# MAGIC %md
# MAGIC The another capability of Tangent is to build and use anomaly detection models.  
# MAGIC The following functions for detection are available in the Tangent Python package.
# MAGIC - build_model
# MAGIC - detect
# MAGIC - rca

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.1 build_model

# COMMAND ----------

# MAGIC %md
# MAGIC The build model function sends a job request to Tangent to build an anomaly detection model with a prepared time series and configuration. This method returns a Tangent anomaly detection model.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC The anomaly detection process happens in two stages. First, a normal behavior model is built using Tangent.  
# MAGIC This model is then compared to the original time series and detection layers are calculated from different perspectives that indicate how anomalous certain timestamps are.
# MAGIC For both stages, the configuration settings can be modified with the dictionary below.

# COMMAND ----------

ad_build_model_config = {
    # 'normal_behavior':{
        # 'target_column':target_column,
        # 'holiday_column:':'str',
        # 'target_offsets':'combined',
        # 'allow_offsets':True,
        # 'max_offsets_depth': 0,
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
    # },
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
# MAGIC #### Usage

# COMMAND ----------

tw_ad_model = tw.anomaly_detection.build_model(
    configuration = ad_build_model_config,
    dataset = tangent_dataframe
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Output

# COMMAND ----------

tw_ad_model

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.2 detect

# COMMAND ----------

# MAGIC %md
# MAGIC The detect function applies data to a Tangent forecasting model to generate predicted values. 
# MAGIC The user can't specify a configuration since the same configuration as the model building has to be applied.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Usage

# COMMAND ----------

tw_ad_detection = tw.anomaly_detection.detect(
    dataset = tangent_dataframe,
    model = tw_ad_model
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Output

# COMMAND ----------

tw_ad_detection

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2.3 rca

# COMMAND ----------

# MAGIC %md
# MAGIC With Root Cause Analysis (RCA), the user can go in depth into the features created by Tangent.
# MAGIC The user can access this information using the rca function. 
# MAGIC The user can choose to extract the results from specific models in the model zoo if there are multiple.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC Many configuration settings of Forecasting can be found again here in AutoForecasting.  
# MAGIC Additional preprocessing functionalities are added to the list of potential configuration settings.

# COMMAND ----------

ad_rca_config = {
    'model_indexes':[
    ]
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Usage

# COMMAND ----------

tw_ad_rca = tw.anomaly_detection.rca(
    configuration = ad_rca_config,
    dataset = tangent_dataframe,
    model = tw_ad_model
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Output

# COMMAND ----------

tw_ad_rca

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3. Insights

# COMMAND ----------

# MAGIC %md
# MAGIC This section will explain the different capabilities to facilitate getting insights from Tangent results.  
# MAGIC The following functions for insights are available in the Tangent Python package.
# MAGIC - properties
# MAGIC - features

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3.1 Properties

# COMMAND ----------

# MAGIC %md
# MAGIC The properties function receives a tangent model and transforms the insights into a Pandas dataframe.  
# MAGIC With this, the user can quickly identify which predictors were used by Tangent in the model building process.

# COMMAND ----------

tw_properties = tw.insights.properties(model=tw_fc_model.to_dict())
tw_properties

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3.2 Features

# COMMAND ----------

# MAGIC %md
# MAGIC The features function receives a tangent model and transforms the insights into a Pandas dataframe.  
# MAGIC With this, the user can quickly identify which features were generated by Tangent in the model building process.

# COMMAND ----------

tw_features = tw.insights.features(model=tw_fc_model.to_dict())
tw_features

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4. SparkParallelProcessing

# COMMAND ----------

# MAGIC %md
# MAGIC This section will explain how to leverage the combined capabilities of Tangent and Spark.  
# MAGIC By using the SparkParallelProcessing class the user can run multiple jobs in parallel to scale up their use of Tangent.
# MAGIC
# MAGIC Every Tangent method can be applied to the Spark parallel processer by describing a correct Spark job with the right parameters.  
# MAGIC Then a list of jobs can be sent to Tangent to be executed simultaneously leveraging Spark RDD's.

# COMMAND ----------

# MAGIC %md
# MAGIC First, activate a Tangent Spark object.

# COMMAND ----------

tw_spark = tangent_works.SparkParallelProcessing(app_name='Example')

# COMMAND ----------

# MAGIC %md
# MAGIC Next, describe an array of Spark jobs by creating a tuple with:
# MAGIC - a unique identifier
# MAGIC - a Tangent function (e.g. tw.forecasting.build_model)
# MAGIC - the inputs for the Tangent function in the shape of a dictionary.

# COMMAND ----------

spark_jobs = []
for job_id in range(2):
    parameters = {
        'dataset':tangent_dataframe,
        'configuration':{}
        }
    spark_job = (
        job_id,
        tw.forecasting.build_model,
        parameters
        )
    spark_jobs.append(spark_job)

# COMMAND ----------

# MAGIC %md
# MAGIC Sent the requests by applying the "run" method on the Tangent Spark object.

# COMMAND ----------


tw_parallel_model_building = tw_spark.run(jobs=spark_jobs)
