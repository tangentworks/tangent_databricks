# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - Mathematical Settings

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Setup

# COMMAND ----------

import tangent_works as tw
import pandas as pd
from copy import deepcopy

# COMMAND ----------

class configuration_tutorial:
    def auto_forecast(
        job_name,
        time_series,
        configuration,
        ):
        tangent_auto_forecast = tw.AutoForecasting(time_series=time_series, configuration=configuration)
        tangent_auto_forecast.run()
        return {
            'job_name':job_name,
            'result':tangent_auto_forecast
            }

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as splt

class visualization:
    def compare_predictions(jobs,default):

        default_result = default['result']
        default_timestamp_column = default_result.time_series.timestamp
        default_df = default_result.time_series.dataframe
        default_target_column = default_result.model.model_zoo.target_name
        default_result_table = default_result.result_table

        fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=default_df[default_timestamp_column], y=default_df[default_target_column], name=default_target_column,line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=default_result_table['timestamp'], y=default_result_table['forecast'], name='default',line=dict(color='blue')), row=1, col=1)

        for job in jobs:
            job_name = job['job_name']
            result = job['result']
            df = result.time_series.dataframe
            timestamp_column = result.time_series.timestamp
            target_column = result.model.model_zoo.target_name
            result_table = result.result_table
            
            fig.add_trace(go.Scatter(x=result_table['timestamp'], y=result_table['forecast'], name=job_name), row=1, col=1)
            if default_target_column!=target_column:
                fig.add_trace(go.Scatter(x=df[timestamp_column], y=df[target_column], name=target_column), row=1, col=1)

        fig.update_layout(height=500, width=1200, title_text="Predictions")
        fig.show()

    def compare_properties(jobs,default):

        default_result = default['result']
        default_model = default_result.model.to_dict()
        default_properties = tw.PostProcessing().properties(model=default_model)
        default_properties['job_name'] = 'default'
        
        all_properties = []
        for job in jobs:
            job_name = job['job_name']
            result = job['result']
            model = result.model.to_dict()
            properties = tw.PostProcessing().properties(model=model)
            properties['job_name'] = job_name
            all_properties.append(properties)

        v_data = pd.concat([default_properties]+all_properties)
        fig = px.bar(v_data[v_data['importance']>0], x='rel_importance', y="job_name", color="name", barmode = 'stack',orientation='h')
        fig.update_layout(height=500, width=1200, title_text="Properties")
        fig.show()

    def compare_features(jobs,default):

        default_result = default['result']
        default_model = default_result.model.to_dict()
        default_features = tw.PostProcessing().features(model=default_model)
        default_features['job_name'] = 'default'
             
        fig = splt.make_subplots(rows=len(jobs)+1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig1 = px.bar(default_features, x='model', y="importance", color="feature", barmode = 'stack',title='default')
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)    
        fig.update_yaxes(title_text="default", row=1, col=1)

        for i,job in enumerate(jobs):
            job_name = job['job_name']
            result = job['result']
            model = result.model.to_dict()
            features = tw.PostProcessing().features(model=model)
            features['job_name'] = job_name

            trace_id = i+2
            fig2 = px.bar(features, x='model', y="importance", color="feature", barmode = 'stack',title=job_name)
            for trace in fig2.data:
                fig.add_trace(trace, row=trace_id, col=1)
            fig.update_yaxes(title_text=job_name, row=trace_id, col=1)
        fig.update_layout(height=max(500,(len(jobs)+1)*200), width=1200, title_text="Features")
        fig.update_layout(barmode='stack')
        fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #1. Data

# COMMAND ----------

file_path = '/Workspace'+dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 2)[0]+'/data/belgian_electricity_grid.csv'
csv_df = pd.read_csv(file_path)[['Timestamp','Quantity','temp','IsPublicHoliday']].tail(24*7*4).head(648).reset_index(drop=True)
for i in range(310,315):
    csv_df.at[i,'Quantity'] = pd.NA
for i in range(400,406):
    csv_df.at[i,'temp'] = pd.NA
for i in range(512,524):
    csv_df.at[i,'IsPublicHoliday'] = pd.NA
tangent_dataframe = pd.concat([csv_df.iloc[:240],csv_df.iloc[246:]])
# ------------------------------------------------------------------------------------------------------------------------
group_keys = []
timestamp_column = "Timestamp"
target_column = "Quantity"
predictors = [s for s in list(tangent_dataframe.columns) if s not in group_keys + [timestamp_column, target_column]]
tangent_dataframe = tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors].sort_values(by=group_keys + [timestamp_column]).reset_index(drop=True)
tangent_dataframe[timestamp_column] = pd.to_datetime(pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S"))
tangent_dataframe

# COMMAND ----------

tw_timeseries = tw.TimeSeries(data = tangent_dataframe)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Tangent

# COMMAND ----------

auto_forecasting_configuration = {
    # 'preprocessing': {
    #     'training_rows': [{'from': '2021-01-20 00:00:00','to': '2021-10-31 23:00:00'}],
    #     'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
    #     'columns': [
    #         'string'
    #     ],
    #     'imputation': {
    #         'common': {'type': 'linear','max_gap_length': 0},
    #         'individual': [{'column_name': 'string','value': {'type': 'linear','max_gap_length': 0}}]
    #     },
    #     'time_scaling': {
    #         'time_scale': {'base_unit': 'hour','value': 1},
    #         'aggregations': {
    #             'common': 'mean',
    #             'individual': [
    #                 {'column_name':'string','value':'mean'}
    #             ]
    #         },
    #         'drop_empty_rows': True
    #     }
    # },
    'engine': {
        # 'target_column': "string",
        # 'holiday_column': 'string',
        'prediction_from': {'base_unit': 'sample','value': 1},
        'prediction_to': {'base_unit': 'sample','value': 24},
    #     'target_offsets': 'combined',
    #     'predictor_offsets': 'common',
    #     'allow_offsets': True,
    #     'max_offsets_depth': 0,
    #     'normalization': True,
    #     'max_feature_count': 20,
    #     'transformations': [
    #         'exponential_moving_average',
    #         'rest_of_week',
    #         'periodic',
    #         'intercept',
    #         'piecewise_linear',
    #         'time_offsets',
    #         'polynomial',
    #         'identity',
    #         'simple_moving_average',
    #         'month',
    #         'trend',
    #         'day_of_week',
    #         'fourier',
    #         'public_holidays',
    #         'one_hot_encoding'
    #     ],
    #     'daily_cycle': True,
    #     'confidence_level': 90,
    #     'data_alignment': [
    #         {'column_name': 'string','timestamp': 'yyyy-mm-dd hh:mm:ssZ'}
    #     ],
    #     'prediction_boundaries': {
    #         'type': 'explicit',
    #         'max_value': 100,
    #         'min_value': 0
    #     }
    }
}

# COMMAND ----------

job_name = 'default_settings'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
# ------------------------------------------------------------------------------------------------------------
default_job_run = configuration_tutorial.auto_forecast(job_name = job_name,configuration=configuration,time_series=tw_timeseries)

# COMMAND ----------

visualization.compare_predictions(jobs=[],default=default_job_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ## target_column

# COMMAND ----------

job_name = 'change_target_column'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['engine']['target_column'] = predictors[0]
# ------------------------------------------------------------------------------------------------------------
job_run = configuration_tutorial.auto_forecast(job_name = job_name,configuration=configuration,time_series=tw_timeseries)

# COMMAND ----------

visualization.compare_predictions(jobs=[job_run],default=default_job_run)

# COMMAND ----------

visualization.compare_properties(jobs=[job_run],default=default_job_run)

# COMMAND ----------

visualization.compare_features(jobs=[job_run],default=default_job_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ## holiday_column

# COMMAND ----------

job_name = 'set_holiday_column'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['engine']['holiday_column'] = predictors[1]
# ------------------------------------------------------------------------------------------------------------
job_run = configuration_tutorial.auto_forecast(job_name = job_name,configuration=configuration,time_series=tw_timeseries)

# COMMAND ----------

visualization.compare_predictions(jobs=[job_run],default=default_job_run)

# COMMAND ----------

visualization.compare_properties(jobs=[job_run],default=default_job_run)

# COMMAND ----------

visualization.compare_features(jobs=[job_run],default=default_job_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ## target_offsets

# COMMAND ----------

# MAGIC %md
# MAGIC Specifies offsets of target used in the model building process. If it is set to *none*, no target offsets and features using target offset will be used in the model.  
# MAGIC If it is set to *common* only common offsets of target for situations within one day will be used. *close* means, that for each situation the closest possible offsets of target will be used.  
# MAGIC If it is set to *combined*, the *close* will be used for situation within the first two days and *common* will be used for further forecasting horizons. Generally, default value is *combined*,  
# MAGIC however if predictor_offsets are set to *close*, then target_offsets will be *close* as well or if allow_offsets is set to false, then *none* will be used.'

# COMMAND ----------

job_runs = []
for setting in ['none', 'common', 'close', 'combined']:
    job_name = 'target_offsets'+'_'+str(setting)
    # ------------------------------------------------------------------------------------------------------------
    configuration = deepcopy(auto_forecasting_configuration)
    configuration['engine']['target_offsets'] = setting
    # ------------------------------------------------------------------------------------------------------------
    job_run = configuration_tutorial.auto_forecast(job_name = job_name,configuration=configuration,time_series=tw_timeseries)
    job_runs.append(job_run)

# COMMAND ----------

visualization.compare_predictions(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_properties(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_features(jobs=job_runs,default=default_job_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  allow_offsets

# COMMAND ----------

# MAGIC %md
# MAGIC Enables using offset transformations of predictors, for example useful for using information of influencers that are not available throughout the entire forecasting horizon

# COMMAND ----------

job_runs = []
for setting in [True,False]:
    job_name = 'allow_offsets'+'_'+str(setting)
    # ------------------------------------------------------------------------------------------------------------
    configuration = deepcopy(auto_forecasting_configuration)
    configuration['engine']['allow_offsets'] = setting
    # ------------------------------------------------------------------------------------------------------------
    job_run = configuration_tutorial.auto_forecast(job_name = job_name,configuration=configuration,time_series=tw_timeseries)
    job_runs.append(job_run)

# COMMAND ----------

visualization.compare_predictions(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_properties(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_features(jobs=job_runs,default=default_job_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  normalization

# COMMAND ----------

# MAGIC %md
# MAGIC Determines whether predictors should be normalized (scaled by their mean and standard deviation); switching this setting off may improve the modeling of data with structural changes

# COMMAND ----------

job_runs = []
for setting in [True,False]:
    job_name = 'normalization'+'_'+str(setting)
    # ------------------------------------------------------------------------------------------------------------
    configuration = deepcopy(auto_forecasting_configuration)
    configuration['engine']['normalization'] = setting
    # ------------------------------------------------------------------------------------------------------------
    job_run = configuration_tutorial.auto_forecast(job_name = job_name,configuration=configuration,time_series=tw_timeseries)
    job_runs.append(job_run)

# COMMAND ----------

visualization.compare_predictions(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_properties(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_features(jobs=job_runs,default=default_job_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  daily_cycle

# COMMAND ----------

# MAGIC %md
# MAGIC Decides whether models should focus on respective times within the day (specific hours, quarter-hours, etc.); if not set, Tangent will determine this property automatically using autocorrelation analysis

# COMMAND ----------

job_runs = []
for setting in [True,False]:
    job_name = 'daily_cycle'+'_'+str(setting)
    # ------------------------------------------------------------------------------------------------------------
    configuration = deepcopy(auto_forecasting_configuration)
    configuration['engine']['daily_cycle'] = setting
    # ------------------------------------------------------------------------------------------------------------
    job_run = configuration_tutorial.auto_forecast(job_name = job_name,configuration=configuration,time_series=tw_timeseries)
    job_runs.append(job_run)

# COMMAND ----------

visualization.compare_predictions(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_properties(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_features(jobs=job_runs,default=default_job_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  max_feature_count

# COMMAND ----------

# MAGIC %md
# MAGIC Determines the maximal possible number of terms/features in each model in the modelZoo; if not set, Tangent will automatically calculate the complexity based on the sampling period of the dataset.

# COMMAND ----------

job_runs = []
for setting in [3,30,300]:
    job_name = 'max_feature_count'+'_'+str(setting)
    # ------------------------------------------------------------------------------------------------------------
    configuration = deepcopy(auto_forecasting_configuration)
    configuration['engine']['max_feature_count'] = setting
    # ------------------------------------------------------------------------------------------------------------
    job_run = configuration_tutorial.auto_forecast(job_name = job_name,configuration=configuration,time_series=tw_timeseries)
    job_runs.append(job_run)

# COMMAND ----------

visualization.compare_predictions(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_properties(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_features(jobs=job_runs,default=default_job_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  offset_limit

# COMMAND ----------

# MAGIC %md
# MAGIC The maximum limit for offsets, defines how far offsets are taken into account in the model building process. If not set Tangent will determine this automatically.

# COMMAND ----------

job_runs = []
for setting in [0,-12,-100]:
    job_name = 'offset_limit'+'_'+str(setting)
    # ------------------------------------------------------------------------------------------------------------
    configuration = deepcopy(auto_forecasting_configuration)
    configuration['engine']['offset_limit'] = setting
    # ------------------------------------------------------------------------------------------------------------
    job_run = configuration_tutorial.auto_forecast(job_name = job_name,configuration=configuration,time_series=tw_timeseries)
    job_runs.append(job_run)

# COMMAND ----------

visualization.compare_predictions(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_properties(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_features(jobs=job_runs,default=default_job_run)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  transformations

# COMMAND ----------

# MAGIC %md
# MAGIC An enumeration of all transformation types Tangent should use during feature engineering. If not provided, the Tangent Engine will determine the optimal transformations automatically

# COMMAND ----------

all_transformation_settings = [
    'exponential_moving_average',
    'rest_of_week',
    'periodic',
    'intercept',
    'piecewise_linear',
    'time_offsets',
    'polynomial',
    'identity',
    'public_holidays',
    'one_hot_encoding',
    'simple_moving_average',
    'month',
    'trend',
    'day_of_week',
    'fourier'
]

default_transformation_settings = [
        'exponential_moving_average',
        'rest_of_week',
        'periodic',
        'intercept',
        'piecewise_linear',
        'time_offsets',
        'polynomial',
        'identity',
        # 'public_holidays',
        # 'one_hot_encoding'
        # 'simple_moving_average',
        # 'month',
        # 'trend',
        # 'day_of_week',
        # 'fourier',
    ]

transformation_settings = []
for key in all_transformation_settings:
    if key in default_transformation_settings:
        transformation_setting = [f for f in default_transformation_settings if f!=key]
        job_name = key+'_'+'off'
    else:
        transformation_setting = default_transformation_settings+[key]
        job_name = key+'_'+'on'
    transformation_settings.append({'job_name':job_name,'setting':transformation_setting})

# COMMAND ----------

job_runs = []
for setting in transformation_settings:
    job_name = setting['job_name']
    # ------------------------------------------------------------------------------------------------------------
    configuration = deepcopy(auto_forecasting_configuration)
    configuration['engine']['transformations'] = setting['setting']
    if job_name=='public_holidays_on':
        configuration['engine']['holiday_column'] = predictors[1]
    if job_name=='one_hot_encoding_on':
        configuration['engine']['categorical_columns'] = [predictors[1]]
    # ------------------------------------------------------------------------------------------------------------
    job_run = configuration_tutorial.auto_forecast(job_name = job_name,configuration=configuration,time_series=tw_timeseries)
    job_runs.append(job_run)

# COMMAND ----------

visualization.compare_predictions(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_properties(jobs=job_runs,default=default_job_run)

# COMMAND ----------

visualization.compare_features(jobs=job_runs,default=default_job_run)
