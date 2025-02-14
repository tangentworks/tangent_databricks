# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - 302 Context Settings

# COMMAND ----------

# MAGIC %md
# MAGIC Tangent is designed to automate as much as possible. By default, Tangent will make decisions based on the data and apply specific settings during the model building and inferencing. However, the user can decide to turn on or off certain settings to optimize Tangent's results for their use case.  
# MAGIC
# MAGIC There are both mathematical and context settings in the Tangent configurations. In this tutorial, we cover the context settings. These settings will have an impact on the output that Tangent generates and are useful for applying Tangent to specific use cases.
# MAGIC
# MAGIC This notebook shows what conext settings exist, how to use them and why or when to use them. The following settings are covered:
# MAGIC
# MAGIC - preprocessing
# MAGIC   - prediction_rows
# MAGIC   - training_rows
# MAGIC   - columns
# MAGIC   - imputation
# MAGIC   - time_scaling
# MAGIC - engine
# MAGIC   - target_column
# MAGIC   - prediction_from
# MAGIC   - prediction_to
# MAGIC   - confidence_level
# MAGIC   - data_alignment
# MAGIC   - prediction_boundaries

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Setup

# COMMAND ----------

# MAGIC %md
# MAGIC First, import the tangent_works package and other supporting libraries.

# COMMAND ----------

import tangent_works
import pandas as pd
from copy import deepcopy
import json

# COMMAND ----------

tw = tangent_works.TangentWorks()

# COMMAND ----------

# MAGIC %md
# MAGIC Here we define a function to combine all steps of autoforecasting. This will simplify the rest of the notebook as we would otherwise be repeating these steps in this notebook.

# COMMAND ----------

class user_defined:
    def auto_forecast(
        job_name,
        dataset,
        configuration,
        ):
        tangent_auto_forecast = tw.forecasting.auto_forecast(configuration=configuration,dataset=dataset)
        model = tangent_auto_forecast.model.to_dict()
        properties = tw.insights.properties(model=model)
        features = tw.insights.features(model=model)
        result_table = tangent_auto_forecast.predictions
        return {
            'job_name':job_name,
            'result_table':result_table,
            'properties':properties,
            'features':features
            }

# COMMAND ----------

# MAGIC %md
# MAGIC To visualize the results of this exercise, the following visualization functions can be used.

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as splt

class visualization:

    def data(df,timestamp,target,predictors):
        fig = splt.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=df[timestamp], y=df[target], name=target,connectgaps=False), row=1, col=1)
        for idx, p in enumerate(predictors): fig.add_trace(go.Scatter(x=df[timestamp], y=df[p], name=p,connectgaps=False), row=2, col=1)
        fig.update_layout(height=600, width=1100, title_text="Data visualization")
        fig.show(renderer='databricks')

    def predictions(df):
        fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        color_map = {'training':'green','testing':'red','production':'goldenrod'}
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['target'], name='target',line=dict(color='black')), row=1, col=1)
        for forecasting_type in df['type'].unique():
            v_data = df[df['type']==forecasting_type].copy()
            fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['forecast'], name=forecasting_type,line=dict(color=color_map[forecasting_type])), row=1, col=1)
            if forecasting_type=='production':
                fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['lower_bound'], name='lower_bound',line=dict(color='grey')), row=1, col=1)
                fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['upper_bound'], name='upper_bound',line=dict(color='grey')), row=1, col=1)
        fig.update_layout(height=500, width=1000, title_text="Results")
        fig.show(renderer='databricks')

    def predictor_importance(df):
        v_data = df[df['importance']>0]
        x_axis = 'name'
        y_axis = 'rel_importance'
        fig1 = go.Figure(go.Bar(x=v_data[x_axis], y=v_data[y_axis],text=round(v_data[y_axis],2),textposition='auto'))
        fig1.update_layout(height=500,width=1000,title_text='Predictor Importances',xaxis_title=x_axis,yaxis_title=y_axis)
        print('Predictors not used:'+str(list(df[~(df['importance']>0)]['name'])))
        fig1.show(renderer='databricks')

    def feature_importance(df):
        fig = px.treemap(df, path=[px.Constant("all"), 'model', 'feature'], values='importance',hover_data=['beta'],color='feature')
        fig.update_traces(root_color="lightgrey")
        fig.update_layout(height=600, width=1000, title_text="Features",margin = dict(t=50, l=25, r=25, b=25))
        fig.show(renderer='databricks')

# COMMAND ----------

# MAGIC %md
# MAGIC #1. Data

# COMMAND ----------

# MAGIC %md
# MAGIC In order to show the impact of all specific settings, one of the example datasets from this tutorial is used and modified. A subset of data is used and gaps are introduced as well.

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

# MAGIC %md
# MAGIC To understand the impact of certain contextual settings, lets visualize the data and scope the context of this exercise.

# COMMAND ----------

visualization.data(df=tangent_dataframe,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Tangent

# COMMAND ----------

# MAGIC %md
# MAGIC ## Default

# COMMAND ----------

# MAGIC %md
# MAGIC First, lets identify the default behavior of Tangent Autoforecasting. To provide a clear example as a reference, we will include a specific forecasting horizon with the default settings. Here we ask Tangent to execute a 24 sample ahead forecast. 

# COMMAND ----------

default_configuration = {
    'preprocessing': {
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
    },
    'engine': {
        # 'target_column': "string",
        'prediction_from': {'base_unit': 'sample','value': 1},
        'prediction_to': {'base_unit': 'sample','value': 24},
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

# MAGIC %md
# MAGIC We then execute the default job using the user defined function by combining the time series and the default configuration.

# COMMAND ----------

job_name = 'default'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(default_configuration)
print("configuration =",json.dumps(configuration, indent=4))
# ------------------------------------------------------------------------------------------------------------
job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC The graph below shows that without any specific parameters set, Tangent will only generate production forecasts by training a model on the entire dataset and only making an inference for the relevant timestamps in the forecasting horizon.  
# MAGIC This enables the user to quickly create a lean forecasting process in production.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC The properties are visualized as a reference for comparing to settings discussed later in the notebook.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC There are two main sections in the autoforecasting configuration. Preprocessing handles all the steps before model building and prediction and contains the following parameters:
# MAGIC
# MAGIC - preprocessing
# MAGIC   - prediction_rows
# MAGIC   - training_rows
# MAGIC   - columns
# MAGIC   - imputation
# MAGIC   - time_scaling

# COMMAND ----------

# MAGIC %md
# MAGIC ### prediction_rows

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC The prediction rows allow the user to generate in-sample training and out-of-sample test values.  
# MAGIC This can help with validating the performance of Tangent's predictions and help the user optimize the training process.  
# MAGIC
# MAGIC The prediction rows are presented to Tangent as an array of dictonaries with timestamps "from" and "to" as content.  
# MAGIC They describe the intervals of timestamps for which predicted values will be generated. This setting only impacts the prediction process and not the model building process.    
# MAGIC
# MAGIC __When to use__:  
# MAGIC In case the user wants to reduce the output generated by Tangent to only a subset of values that is relevant to their use case.  
# MAGIC
# MAGIC __Why to use__:  
# MAGIC This parameter only exists to provide the user with a specified set of predicted values to inspect.
# MAGIC
# MAGIC __How to use__:  
# MAGIC Let's provide two examples. One where all rows in the dataset are mentioned and one where a subset of values is selected.

# COMMAND ----------

# MAGIC %md
# MAGIC #### all rows

# COMMAND ----------

# MAGIC %md
# MAGIC The section below copies the default_configuration and identifies the first and last timestamp in our dataset.  
# MAGIC These values will be the "from" and "to" timestamps in the prediction_rows setting respectively.  
# MAGIC The resulting configuration that Tangent receives is printed below. This configuration will become the basis for other sections of this notebook.

# COMMAND ----------

job_name = 'all_prediction_rows'
# ------------------------------------------------------------------------------------------------------------
min_timestamp = str(tangent_dataframe[timestamp_column].min())
max_timestamp = str(tangent_dataframe[timestamp_column].max())
auto_forecasting_configuration = deepcopy(default_configuration)
auto_forecasting_configuration['preprocessing']['prediction_rows'] = [{'from': min_timestamp,'to': max_timestamp}]
configuration = deepcopy(auto_forecasting_configuration)
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC The job can now be executed and the predictions are collected from the result table.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC In the graph below we can now see that we received in sample training results back from Tangent as compared to the default settings. Sometimes gaps in these predictions can exist whenever the model that was build can't be used during the prediction step. This happens for example at the beginning of the data when the model has to use offsets or when there are gaps in the target or predictors. The latter can be fixed with the imputation setting discussed later.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC You can also compare the properties and see how the are the same as in the default case.  
# MAGIC This shows that the prediction_rows do not impact the model building and only the prediction step.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### subset rows

# COMMAND ----------

# MAGIC %md
# MAGIC In case you only need predictios for a subset of the records in the dataset, you can specify different intervals in the prediction_rows parameter. As an example, below only the first quarter of data is set in a first interval, and the last quarter in a second interval. 

# COMMAND ----------

job_name = 'subset_prediction_rows'
# ------------------------------------------------------------------------------------------------------------
min_timestamp = str(tangent_dataframe[timestamp_column].min())
T1_timestamp = str(tangent_dataframe[timestamp_column].quantile(0.25,interpolation='nearest'))
T2_timestamp = str(tangent_dataframe[timestamp_column].quantile(0.75,interpolation='nearest'))
configuration = deepcopy(auto_forecasting_configuration)
configuration['preprocessing']['prediction_rows'] = [
  {'from': min_timestamp,'to': T1_timestamp},
  {'from': T2_timestamp,'to': max_timestamp},
  ]
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC This subset job can now be sent to Tangent.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC Notice the gap that seems to exist in the history. Tangent will only return records in the result_table for timestamps that were specified in the prediction_rows. In this example only the first and last quarter of the dataset.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Again, the predictor importances remain the same as the model is the same as in the previous exercise.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### training_rows

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC The training rows allow the user to specify to Tangent which records in the dataset need to be considere for training. By default, all records are included but when specified, Tangent will ignore the data outside of the provided intervals.  
# MAGIC
# MAGIC The training rows are presented to Tangent as an array of dictonaries with timestamps "from" and "to" as content.  
# MAGIC This setting only impacts the model building process and not the prediction process.  
# MAGIC
# MAGIC This setting will also work when no prediction_rows are set.
# MAGIC
# MAGIC __When to use__:  
# MAGIC In case the user wants to specify to Tangent to only train the model on a subset of values that is relevant to their use case.  
# MAGIC
# MAGIC __Why to use__:  
# MAGIC This parameter exists to enable the user to excluding portions of the data on which the model shouldn't be trained or include a shorter history to investigate the impact of more recent compared to older data.
# MAGIC
# MAGIC __How to use__:  
# MAGIC Let's provide two examples. One where a simple train test split is achieved and one where a rolling window for dynamic training is selected.

# COMMAND ----------

# MAGIC %md
# MAGIC #### train test split

# COMMAND ----------

# MAGIC %md
# MAGIC As an example, a training period is described by splitting the data in to 2/3 for training & 1/3 for testing. 
# MAGIC This period is found by taking the first timestamp of the dataset and calculating where the last 1/3 of the data starts.
# MAGIC Prediction rows are also provided from the first exercise to be able to show the impact of the training_rows parameter.

# COMMAND ----------

job_name = 'train_test_split'
# ------------------------------------------------------------------------------------------------------------
training_from = str(tangent_dataframe[timestamp_column].min())
training_to = str(tangent_dataframe[timestamp_column].iloc[-len(tangent_dataframe)//3])
configuration = deepcopy(auto_forecasting_configuration)
configuration['preprocessing']['training_rows'] = [{'from': training_from,'to': training_to}]
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job and extract the result table.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC The graph now shows both training and testing results. The training results overlap with the intervals provided with the training_rows parameter and the testing results will indicate all timestamps that were included in the prediction rows but excluded from the training rows.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC When we compare the production forecasts and the models with the results from the default experiment, we will find that they are slightly different as less information was considered in this exercise during model building. 

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### rolling training window

# COMMAND ----------

# MAGIC %md
# MAGIC In the scenario where you want to remain adaptable to changing circumstances and only include the latest information, you can choose to limit the training horizon to a number of values at the end of the dataset. The example below sets a training horizon from the value of two weeks ago (24 hours x 7 days x 2 weeks on hourly sampled data) to the end of the dataset.
# MAGIC
# MAGIC To only show the relevant values, the prediction_rows is also respecified. You will find these two settings are often used together.

# COMMAND ----------

job_name = 'rolling training window'
# ------------------------------------------------------------------------------------------------------------
training_from = str(tangent_dataframe[timestamp_column].tail(24*7*2).min())
training_to = str(tangent_dataframe[timestamp_column].max())
configuration = deepcopy(auto_forecasting_configuration)
configuration['preprocessing']['training_rows'] = [{'from': training_from,'to': training_to}]
configuration['preprocessing']['prediction_rows'] = [{'from': training_from,'to': training_to}]
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job and extract the result table.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']

# COMMAND ----------

# MAGIC %md
# MAGIC In the graph below you can see how only the last part of the dataset was considered for training. This has surely impacted the model and production forecast. The user can make a tradeoff between including sufficient data for capturing the pattern and recent data to capture recent changes in the data.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### columns

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC The "columns" parameter enables the user to include or exclude specific columns from the exercise in the configuration rather than having to modify the time series.
# MAGIC
# MAGIC __When to use__:  
# MAGIC For example, when dynamically testing the impact of specific columns in a predictive model from Tangent.
# MAGIC
# MAGIC __Why to use__:  
# MAGIC When the user wants to focus the analysis on a subset of columns.
# MAGIC
# MAGIC __How to use__:  
# MAGIC The user presents an array of strings with the column names as they appear in the data.

# COMMAND ----------

# MAGIC %md
# MAGIC In this example dataset, the user has "Quantity", "temp" & "IsPublicHoliday" to choose from. By default all columns will be included. 

# COMMAND ----------

print([target_column]+predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's specify to only use the target: "Quantity" and the predictor "temp".

# COMMAND ----------

job_name = 'select columns'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['preprocessing']['columns'] = [
    'Quantity',
    'temp'
]
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job and extract the result table & properties.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC From the predictions visual, we can't learn much about the impact of this setting. The results look similar to the default but are slightly different when we look closer.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC In the properties we see the real impact. Where the IsPublicHoliday column previously appeared in the default model, now the model only is based on features from the included columns.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### imputation

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC The imputation parameter gives the user a dynamic way to fill in missing cell values or interpolate entire rows. Tangent can model over missing values, however when there are too many missing values, the quality of the predictions could decrease. To avoid this the user can impute values either linearly (linear) of by carrying the last observation forward (locf) over a gap of a specific size. Imputing too large gaps however can distort predictive value from the data and impact results.
# MAGIC
# MAGIC __When to use__:  
# MAGIC When there are gaps present in the data.
# MAGIC
# MAGIC __Why to use__:  
# MAGIC To increase the number of cells that can be leveraged during model building.
# MAGIC
# MAGIC __How to use__:  
# MAGIC The user provides a dictionary with a imputation type and max_gap_length. 
# MAGIC The user can either set a common imputation strategy or manage the imputation of specific columns. For both situations an example is provided below. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### common

# COMMAND ----------

# MAGIC %md
# MAGIC In this example, we set a common imputation strategy across all columns to linearly fill in gaps in the data of maximally 24 consecutive empty values.  

# COMMAND ----------

job_name = 'common_imputation'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['preprocessing']['imputation'] = {
    'common': {'type': 'linear','max_gap_length': 24},
}
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job and extract the result table & properties.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC In the graph you will find slight differences in the predicted values compared to the default experiment.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC From the predictor importances we learn that the model has changed. In this example more predictive value was found in the predictors indicating that by imputing the values, Tangent was better able to detect a useful patterns in these columns.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### individual

# COMMAND ----------

# MAGIC %md
# MAGIC When using the individual imputation, the set strategy will only apply to a specific column. This parameter can used for different columns separately. It can also be used in combination with the common strategy to set a general imputation strategy for all other columns except those described individually.  
# MAGIC
# MAGIC The example below uses Last Observation Carried Forward (LOCF) to fill in gaps of maximally 12 consecutive values for column "temp" specifically. All other columns will not be imputed.

# COMMAND ----------

job_name = 'individual_imputation'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['preprocessing']['imputation'] = {
     'individual': [{'column_name': 'temp','value': {'type': 'locf','max_gap_length': 12}}]
}
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job and extract the result table & properties.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC In the graph you will find slight differences in the predicted values compared to the default experiment.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC From the predictor importances we learn that the model has changed and more emphasis is laid on the column "temp" as we would expect.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### time_scaling

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC With time scaling, the user can regroup timeseries to their desired sampling rate. Tangent will match all values to a near and equidistant timestamp. This can also be used to group the timeseries by lower sampling rates e.g. group hourly data to daily data according to different grouping strategies. 
# MAGIC
# MAGIC __When to use__:  
# MAGIC When the dataset contains irregularly sampled time series.
# MAGIC
# MAGIC __Why to use__:  
# MAGIC To allow Tangent to model over irregular time series.
# MAGIC
# MAGIC __How to use__:  
# MAGIC The user provides a dictionary with a timescale units and lengths and specifies the aggregations either commonly across all columns or individually. For both situations an example is provided below. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### common

# COMMAND ----------

# MAGIC %md
# MAGIC In this example, the hourly sampled dataset will be regrouped to daily data by setting the time_scale base unit to "day" and value to "1". All columns will be grouped by day and the aggregation will be the mean (or average).

# COMMAND ----------

job_name = 'common_time_scaling'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['preprocessing']['time_scaling'] =  {
            'time_scale': {'base_unit': 'day','value': 1},
            'aggregations': {'common': 'mean'}
        }
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job and extract the result table & properties.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC In the graph we can now see that the target and predictions are now values with a daily sampling rate. 

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Now the model is also completely different as time scaled values have been used for model building.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### individual

# COMMAND ----------

# MAGIC %md
# MAGIC In this example, the hourly sampled dataset will be regrouped to 4 hourly data by setting the time_scale base unit to "hour" and value to "4". Each column will now be specifically aggregated with a different aggregation type. The options are:
# MAGIC - mean
# MAGIC - sum
# MAGIC - minimum
# MAGIC - maximum
# MAGIC - mode

# COMMAND ----------

job_name = 'individual_time_scaling'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['preprocessing']['time_scaling'] =  {
            'time_scale': {'base_unit': 'hour','value': 4},
            'aggregations': {
                'individual': [
                    {'column_name':'Quantity','value':'mode'},
                    {'column_name':'temp','value':'minimum'},
                    {'column_name':'IsPublicHoliday','value':'maximum'}
                ]
            }
        }
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job and extract the result table & properties.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC The graph shows the new results time scaled per 4 hours. 

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC The properties also show the model has adapted to the new sampling rate. Now the user can select different strategies to optimize for results.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Engine

# COMMAND ----------

# MAGIC %md
# MAGIC The second main section in the autoforecasting configuration contains the engine parameters. This handles all the steps during model building and prediction and contains the following parameters:
# MAGIC
# MAGIC - engine
# MAGIC   - target_column
# MAGIC   - prediction_from
# MAGIC   - prediction_to
# MAGIC   - confidence_level
# MAGIC   - data_alignment
# MAGIC   - prediction_boundaries

# COMMAND ----------

# MAGIC %md
# MAGIC ### target_column

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC This parameter specifies which column should be used as the target. By default the first column after the timestamp is selected when not specified.
# MAGIC
# MAGIC __When to use__:  
# MAGIC When the user wants to dynamically change between different potential target columns during analysis.  
# MAGIC __Why to use__:  
# MAGIC This ensures the user specified column is used as the target in case the dataset is modified.  
# MAGIC __How to use__:  
# MAGIC Specify a column name present in the dataset. This parameter is often used together with the "columns" parameter.

# COMMAND ----------

# MAGIC %md
# MAGIC In this example the column "temp" is specified as the target instead of the column "Quantity", which is the first column after the timestamp.

# COMMAND ----------

job_name = 'target_column'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['engine']['target_column'] = 'temp'
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job and extract the result table & properties.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC The predictions are now completely different from before since a different target column was specified. 

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC In the properties we learn that all the other columns are now considered predictors.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### prediction_from

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC The prediction from sets the starting point of the forecasting horizon.  
# MAGIC __When to use__:  
# MAGIC When the use case requires a specific starting point instead of the first sample after the last target value.  
# MAGIC __Why to use__:  
# MAGIC This enables the user to apply Tangent to different use cases with specific forecasting horizons.  
# MAGIC __How to use__:  
# MAGIC Specify a dictionary with a "base unit" such as "sample", "second", "minute", "hour", "day" and a value in number of such base units.

# COMMAND ----------

# MAGIC %md
# MAGIC In this example we'll set the forecasting horizon to start from the 10th sample after the last target value.

# COMMAND ----------

job_name = 'prediction_from'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['engine']['prediction_from'] = {
    "base_unit": "sample",
    "value": 10
}
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job and extract the result table & properties.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC The predictions now start from the 10th sample. You can see the gap between the target and the production forecast. 

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC In the properties the model will have adapted to only those situations between the 10th aample and the end of the forecasting horizon.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### prediction_to

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC The prediction to sets the end point of the forecasting horizon.  
# MAGIC __When to use__:  
# MAGIC When the use case requires a specific end point.  
# MAGIC __Why to use__:  
# MAGIC This enables the user to apply Tangent to different use cases with specific forecasting horizons.  
# MAGIC __How to use__:  
# MAGIC Specify a dictionary with a "base unit" such as "sample", "second", "minute", "hour", "day" and a value in number of such base units.

# COMMAND ----------

# MAGIC %md
# MAGIC In this example we'll set the forecasting horizon to start end at the 12th sample after the last target value.

# COMMAND ----------

job_name = 'prediction_to'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['engine']['prediction_to'] = {
    "base_unit": "sample",
    "value": 12
}
print("configuration =",json.dumps(configuration, indent=4))
# ------------------------------------------------------------------------------------------------------------
job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC The predictions now end after 12 samples.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC In the properties the model will have adapted to only those situations between the 1st and 12th samples.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### confidence_level

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC The confidence level calculates a band around the predictions to define the certainty level within which a prediction will fall.  
# MAGIC This setting does not affect model building.  
# MAGIC __When to use__:  
# MAGIC When the use case requires the user to express upper or lower bounds of certainty.  
# MAGIC __Why to use__:  
# MAGIC This enables the user to validate the stability of their forecasts.  
# MAGIC __How to use__:  
# MAGIC Specify an integer between 0 and 100 to indicate the confidence percentage level.

# COMMAND ----------

# MAGIC %md
# MAGIC In this example we'll set the confidence_level at 95 instead of the default 90.

# COMMAND ----------

job_name = 'confidence_level'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['engine']['confidence_level'] = 95
print("configuration =",json.dumps(configuration, indent=4))
# ------------------------------------------------------------------------------------------------------------
job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC The lower and upper bounds around the prediction horizon have not shifted.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC The model is not impacted by this setting.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### data_alignment

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC The data alignment describes the data situation that is found in the input dataset.  
# MAGIC __When to use__:  
# MAGIC When the use case requires the user to define a specific data situation which will always occur in a production scenario.  
# MAGIC __Why to use__:  
# MAGIC To ensure proper feature engineering, Tangent needs to be aware of what information with which (leading) lags can be used from all the included columns.  
# MAGIC __How to use__:  
# MAGIC The data aligment is automatically detected from the dataset. Sometimes, it is useful for the user to manually set the data alignment to reflect a data situation that needs to be tested.

# COMMAND ----------

# MAGIC %md
# MAGIC In this example we'll align all the columns with the target.

# COMMAND ----------

last_target_timestamp = str(tangent_dataframe[timestamp_column].iloc[tangent_dataframe[target_column].last_valid_index()])

# COMMAND ----------

job_name = 'data_alignment'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['engine']['data_alignment'] =  [
  {'column_name': 'temp','timestamp': last_target_timestamp}
]
print("configuration =",json.dumps(configuration, indent=4))
# ------------------------------------------------------------------------------------------------------------
job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC The predictions are significantly impacted by the data alignment since Tangent is told it has to use differently lagged information than with the default.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC This setting can heavily impact the model. Make sure to always reflect a realistic data situation.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### prediction_boundaries

# COMMAND ----------

# MAGIC %md
# MAGIC __What__:  
# MAGIC The prediction boundaries keep the predicted value between a minimun and/or a maximum value.  
# MAGIC __When to use__:  
# MAGIC When the use case does not allow for illogical values beyond a certain boundary (e.g. negative values when predicting sales numbers, now the boundary can be set at min_value: 0)  
# MAGIC __Why to use__:  
# MAGIC This enables the user to generate more stable predictions in case there are strong fluctuations in the input data that impact the prediction process.  
# MAGIC __How to use__:  
# MAGIC Specify a dictionary with a min_value or max_value.

# COMMAND ----------

# MAGIC %md
# MAGIC In this example we set both a min_value at 8000 & max value at 11000. The predictions will not go beyond these boundaries.

# COMMAND ----------

job_name = 'prediction_boundaries'
# ------------------------------------------------------------------------------------------------------------
configuration = deepcopy(auto_forecasting_configuration)
configuration['engine']['prediction_boundaries'] = {
    'type': 'explicit',
    'max_value': 11000,
    'min_value': 8000
} 
print("configuration =",json.dumps(configuration, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Run the job and extract the result table & properties.

# COMMAND ----------

job_run = user_defined.auto_forecast(job_name = job_name,configuration=configuration,dataset=tangent_dataframe)
result_table_df = job_run['result_table']
properties_df = job_run['properties']

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see from the graph, the predicted values did not go beyond the set boundaries.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC When comparing the model with the default experiment, we can see that the model is exactly the same. Only the output is modified.

# COMMAND ----------

visualization.predictor_importance(properties_df)
