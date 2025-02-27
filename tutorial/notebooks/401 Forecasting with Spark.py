# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - Forecasting with Spark

# COMMAND ----------

# MAGIC %md
# MAGIC In this tutorial, you will learn to build forecasting models with Tangent, at scale, by leveraging Spark.  
# MAGIC
# MAGIC We will send multiple requests to Tangent in parallel using Spark RDD's. This will allow Databricks to scale the Tangent custom Docker container to handle all the requests. Databricks will then manage all these individual jobs automatically and send results back when all are finished. Dependent on the number of workers are available in the cluster, more jobs can be processed simultaneously. 
# MAGIC
# MAGIC To show these capabilities, we will use an example dataset from a retail sales forecasting use case from Walmart.  
# MAGIC The goal is to forecast sales of several products across different stores 7 days ahead using historical sales data and other explanatory variables.

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Setup

# COMMAND ----------

# MAGIC %md
# MAGIC First, import the tangent_works package and other supporting libraries.

# COMMAND ----------

import tangent_works
import pandas as pd
import uuid
import numpy as np

# COMMAND ----------

tw = tangent_works.TangentWorks()

# COMMAND ----------

# MAGIC %md
# MAGIC To visualize the results of this exercise, the following visualization functions can be used.

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as splt

class visualization:
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

    def predictions(df):
        fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        color_map = {'training':'green','testing':'red','production':'goldenrod'}
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['target'], name='target',line=dict(color='black')), row=1, col=1)
        for forecasting_type in df['type'].unique():
            v_data = df[df['type']==forecasting_type].copy()
            fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['forecast'], name=forecasting_type,line=dict(color=color_map[forecasting_type])), row=1, col=1)
        fig.update_layout(height=500, width=1000, title_text="Results")
        fig.show(renderer='databricks')

    def data(df,timestamp,target,predictors):
        fig = splt.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=df[timestamp], y=df[target], name=target,connectgaps=True), row=1, col=1)
        for idx, p in enumerate(predictors): fig.add_trace(go.Scatter(x=df[timestamp], y=df[p], name=p,connectgaps=True), row=2, col=1)
        fig.update_layout(height=600, width=1100, title_text="Data visualization")
        fig.show(renderer='databricks')

    def all_properties(properties,group_keys):
        fig = px.bar(properties[properties['importance']>0], x='id', y="rel_importance", color="name", barmode = 'stack',hover_data=group_keys)
        fig.update_layout(height=800, width=1200, title_text="All properties")
        fig.show(renderer='databricks')

# COMMAND ----------

# MAGIC %md
# MAGIC #1. Data

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset that will be used in this notebook is called walmart_weather_smart.  
# MAGIC It contains historical daily sales data and several data points of weather information.  
# MAGIC In the cell below, this dataset is preprocessed and made ready for use with Tangent. Important here is to define the group_keys which are used to identify the individual timeseries in the larger dataset. Here store_nbr & item_nbr indicate the individual product sales at each store. There are 43 different stores &  39 products resulting in 122 combinations of store_nbr & item_nbr. For this example we choose to exclude the sales of 2 specific stores.  

# COMMAND ----------

file_path = '/Workspace'+dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 2)[0]+'/data/walmart_weather_smart.csv'
csv_df = pd.read_csv(file_path)

tangent_dataframe = csv_df[~csv_df['store_nbr'].isin([35,39])].drop(columns=['depart','sunrise','sunset','snowfall']).copy()

group_keys = ['store_nbr','item_nbr']
timestamp_column = "date"
target_column = "sales"
predictors = [
    s
    for s in list(tangent_dataframe.columns)
    if s not in group_keys + [timestamp_column, target_column]
]
tangent_dataframe = (
    tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors]
    .sort_values(by=group_keys + [timestamp_column])
    .reset_index(drop=True)
)
tangent_dataframe[timestamp_column] = pd.to_datetime(
    pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S")
)
combinations = tangent_dataframe[group_keys].drop_duplicates().to_dict('records')
print(dict(tangent_dataframe[group_keys].nunique()))
print('combinations:',len(combinations))

tangent_dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC Below we visualize the time series of 1 store/product combination. 

# COMMAND ----------

i = 0
v_data = tangent_dataframe.loc[(tangent_dataframe[list(combinations[i])] == pd.Series(combinations[i])).all(axis=1)]
visualization.data(df=v_data,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Tangent

# COMMAND ----------

# MAGIC %md
# MAGIC In the Tangent Python Databricks package, there exists the SparkParallelProcessing class which allows the user to send jobs to Tangent in parallel. The forecasting process can be parallelized in different ways. In this example we show how to run in parallel each step in the forecasting process seperately and accross all the combinations.  
# MAGIC
# MAGIC Please note that a function can also be created that handles these invididual steps in series and then this function could be ran in parallel. That remains up to the preference of the user. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jobs

# COMMAND ----------

# MAGIC %md
# MAGIC First, to identify each job and link the resulting information back to our use case, tangent_job objects are created containing a unique ID and user specified parameters. These parameters can be anything that the user wants to share to identify the individual jobs. Here the parameters are the combinations of store and item number.

# COMMAND ----------

tangent_jobs = []
for combination in combinations[:]:
    tangent_jobs.append({'id':str(uuid.uuid4()),'parameters':combination})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Building

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's describe the model building configuration that will be applied to each of the individual model builds in this exercise.  
# MAGIC Default settings are used and a forecasting horizon of 7 samples (or days in this case) is set.

# COMMAND ----------

build_model_configuration = {
    # 'target_column': 'sales',
    # 'categorical_columns': [
    #     'string'
    # ],
    # 'holiday_column': 'string',
    'prediction_from': {
        'base_unit': 'sample',
        'value': 1
    },
    'prediction_to': {
        'base_unit': 'sample',
        'value': 7
    },
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
# MAGIC Now, the jobs are ready to be sent to Tangent to build forecasting models. The build_model functions are sent to Spark and Databricks will start building multiple models simultaneously. 

# COMMAND ----------

tw_spark = tangent_works.SparkParallelProcessing(app_name='Example')
spark_jobs = []
for tangent_job in tangent_jobs:
    tangent_job_parameters = tangent_job['parameters']
    dataset = tangent_dataframe.loc[(tangent_dataframe[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
    spark_jobs.append((tangent_job['id'],tw.forecasting.build_model,{'configuration':build_model_configuration,'dataset':dataset}))
tw_forecasting_build_models = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC When all model building jobs are finished, predictions can be made using these models. 
# MAGIC These can also be sent to Tangent in parallel.  
# MAGIC The prediction step also requires a configuration and is typically in line with the model building configuration.

# COMMAND ----------

predict_configuration = {
    'prediction_from': {
        'base_unit': 'sample',
        'value': 1
        },
    'prediction_to': {
        'base_unit': 'sample',
        'value': 7
    }, 
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
# MAGIC Now, the jobs are ready to be sent to Tangent to generate predictions. The predict functions are sent to Spark and Databricks will start calculating forecasts simultaneously. 

# COMMAND ----------

spark_jobs = []
for tw_forecasting_build_model in tw_forecasting_build_models:
    build_model_id = tw_forecasting_build_model['id']
    tangent_job_parameters = [f['parameters'] for f in tangent_jobs if f['id']==build_model_id][0]
    dataset = tangent_dataframe.loc[(tangent_dataframe[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
    model = tw_forecasting_build_model['result']
    spark_jobs.append((tw_forecasting_build_model['id'],tw.forecasting.predict,{'configuration':predict_configuration,'dataset':dataset,'model':model}))
tw_forecasting_predicts = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Results

# COMMAND ----------

# MAGIC %md
# MAGIC Now, Tangent has generated all results and they need to be processed. The following loop extracts the properties, features and predictions from all tangent_jobs. This step could be parallelized as well if required. 

# COMMAND ----------

all_properties,all_features,all_predictions = [],[],[]
for tangent_job in tangent_jobs:
    job_id = tangent_job['id']
    tw_forecasting_build_model_result = [f['result'] for f in tw_forecasting_build_models if f['id']==job_id][0]
    model = tw_forecasting_build_model_result.to_dict()
    
    properties_df = tw.insights.properties(model=model)
    properties_df['id'] = job_id
    all_properties.append(properties_df)

    features_df = tw.insights.features(model=model)
    features_df['id'] = job_id
    all_features.append(features_df)

    train_from = pd.to_datetime(model['model_zoo']['training_periods'][0]['datetime_from'])
    train_to = pd.to_datetime(model['model_zoo']['training_periods'][0]['datetime_to'])
    predictions_df = [f['result'] for f in tw_forecasting_predicts if f['id']==job_id][0]
    predictions_df['type'] = np.where((train_from<=predictions_df['timestamp'])&(predictions_df['timestamp']<=train_to), 'training', 'production')
    predictions_df['id'] = job_id
    all_predictions.append(predictions_df)

tangent_jobs_df = pd.json_normalize(tangent_jobs)
tangent_jobs_df.columns = tangent_jobs_df.columns.str.replace('parameters.','',regex=False)

tangent_predictions_df = pd.concat(all_predictions).merge(tangent_jobs_df,on='id',how='left')
tangent_properties_df = pd.concat(all_properties).merge(tangent_jobs_df,on='id',how='left')
tangent_features_df = pd.concat(all_features).merge(tangent_jobs_df,on='id',how='left')

tangent_predictions_df['error'] = tangent_predictions_df['forecast']-tangent_predictions_df['target']
tangent_predictions_df['MAE'] = abs(tangent_predictions_df['error'])
tangent_predictions_df['MAPE'] = abs(tangent_predictions_df['error']/tangent_predictions_df['target'])

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC Now, the user can inspect the results from the parallel exercise. Below, a single combination of store_nbr and item_nbr is selected from the list and those results are visualized below.

# COMMAND ----------

i = 5
tangent_job = tangent_jobs[i]
tangent_job_parameters = tangent_job['parameters']

# COMMAND ----------

# MAGIC %md
# MAGIC ## predictions

# COMMAND ----------

v_data = tangent_predictions_df.loc[(tangent_predictions_df[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
visualization.predictions(v_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## properties

# COMMAND ----------

v_data = tangent_properties_df.loc[(tangent_properties_df[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
print(tangent_job_parameters)
visualization.predictor_importance(v_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## features

# COMMAND ----------

v_data = tangent_features_df.loc[(tangent_features_df[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
print(tangent_job_parameters)
visualization.feature_importance(v_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## All properties

# COMMAND ----------

# MAGIC %md
# MAGIC The properties of all combinations could also be outlined next to each other and compared. In this example that could provide insights about which combinations of store and item sales are affected by certain predictors.

# COMMAND ----------

visualization.all_properties(properties=tangent_properties_df,group_keys=group_keys)
