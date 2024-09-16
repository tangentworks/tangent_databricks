# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - Forecasting At Scale

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Setup

# COMMAND ----------

import tangent_works as tw
import pandas as pd
import uuid

# COMMAND ----------

# -------------------------------- Supporting Libraries --------------------------------

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as splt

class visualization:

    def data(df,timestamp,target,predictors):
        fig = splt.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=df[timestamp], y=df[target], name=target,connectgaps=True), row=1, col=1)
        for idx, p in enumerate(predictors): fig.add_trace(go.Scatter(x=df[timestamp], y=df[p], name=p,connectgaps=True), row=2, col=1)
        fig.update_layout(height=600, width=1100, title_text="Data visualization")
        fig.show()

    def predictor_importance(df):
        v_data = df[df['importance']>0]
        x_axis = 'name'
        y_axis = 'rel_importance'
        fig1 = go.Figure(go.Bar(x=v_data[x_axis], y=v_data[y_axis],text=round(v_data[y_axis],2),textposition='auto'))
        fig1.update_layout(height=500,width=1000,title_text='Predictor Importances',xaxis_title=x_axis,yaxis_title=y_axis)
        print('Predictors not used:'+str(list(df[~(df['importance']>0)]['name'])))
        fig1.show()

    def feature_importance(df):
        fig = px.treemap(df, path=[px.Constant("all"), 'model', 'feature'], values='importance',hover_data='beta',color='feature')
        fig.update_traces(root_color="lightgrey")
        fig.update_layout(height=600, width=1000, title_text="Features",margin = dict(t=50, l=25, r=25, b=25))
        fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #1. Data

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

# manually choose one of 102 combinations 
i = 0
# display the combination
v_data = tangent_dataframe.loc[(tangent_dataframe[list(combinations[i])] == pd.Series(combinations[i])).all(axis=1)]
visualization.data(df=v_data,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Configuration

# COMMAND ----------

build_model_configuration = {
    # 'target_column': 'string',
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
# MAGIC #3. Tangent

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jobs

# COMMAND ----------

tangent_jobs = []
for combination in combinations[:]:
    tangent_jobs.append({'id':str(uuid.uuid4()),'parameters':combination})

# COMMAND ----------

# MAGIC %md
# MAGIC ## TimeSeries

# COMMAND ----------

tw_spark = tw.SparkParallelProcessing()
spark_jobs = []
for tangent_job in tangent_jobs:
    tangent_job_parameters = tangent_job['parameters']
    data = tangent_dataframe.loc[(tangent_dataframe[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
    spark_jobs.append((tangent_job['id'],tw.TimeSeries,{'data':data}))
tw_parallel_time_series = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecasting

# COMMAND ----------

spark_jobs = []
for tw_time_series in tw_parallel_time_series:
    spark_jobs.append((tw_time_series['id'],tw.Forecasting,{'time_series':tw_time_series['result'],'configuration':build_model_configuration}))
tw_parallel_Forecasting = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Building

# COMMAND ----------

spark_jobs = []
for tw_forecasting in tw_parallel_Forecasting:
    spark_jobs.append((tw_forecasting['id'],tw_forecasting['result'].build_model,{'inplace':False}))
tw_forecasting_build_models = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction

# COMMAND ----------

spark_jobs = []
tw_spark = tw.SparkParallelProcessing()
for tw_forecasting_build_model in tw_forecasting_build_models:
    spark_jobs.append((tw_forecasting_build_model['id'],tw_forecasting_build_model['result'].forecast,{'configuration':predict_configuration}))
tw_forecasting_predicts = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Results

# COMMAND ----------

all_properties,all_features,all_predictions = [],[],[]
for tangent_job in tangent_jobs:
    job_id = tangent_job['id']
    tw_forecasting_build_model_result = [f['result'] for f in tw_forecasting_build_models if f['id']==job_id][0]
    model = tw_forecasting_build_model_result.model.to_dict()
    
    properties_df = tw.PostProcessing().properties(model=model)
    properties_df['id'] = job_id
    all_properties.append(properties_df)

    features_df = tw.PostProcessing().features(model=model)
    features_df['id'] = job_id
    all_features.append(features_df)

    predictions_df = [f['result'] for f in tw_forecasting_predicts if f['id']==job_id][0]
    predictions_df['id'] = job_id
    all_predictions.append(predictions_df)

tangent_jobs_df = pd.json_normalize(tangent_jobs)
tangent_jobs_df.columns = tangent_jobs_df.columns.str.replace('parameters.','')

tangent_predictions_df = pd.concat(all_predictions).merge(tangent_jobs_df,on='id',how='left')
tangent_properties_df = pd.concat(all_properties).merge(tangent_jobs_df,on='id',how='left')
tangent_features_df = pd.concat(all_features).merge(tangent_jobs_df,on='id',how='left')

tangent_predictions_df['error'] = tangent_predictions_df['forecast']-tangent_predictions_df['target']
tangent_predictions_df['MAE'] = abs(tangent_predictions_df['error'])
tangent_predictions_df['MAPE'] = abs(tangent_predictions_df['error']/tangent_predictions_df['target'])

# COMMAND ----------

# MAGIC %md
# MAGIC #5. Visualization

# COMMAND ----------

def visualize_predictions(df):
    fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['target'], name='target',line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['forecast'], name='forecast',line=dict(color='goldenrod')), row=1, col=1)
    fig.update_layout(height=500, width=1000, title_text="Results")
    fig.show()

i = 5
tangent_job = tangent_jobs[i]
tangent_job_parameters = tangent_job['parameters']

v_data = tangent_predictions_df.loc[(tangent_predictions_df[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
visualize_predictions(v_data)

# COMMAND ----------

tangent_job = tangent_jobs[i]
tangent_job_parameters = tangent_job['parameters']
v_data = tangent_properties_df.loc[(tangent_properties_df[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
print(tangent_job_parameters)
visualization.predictor_importance(v_data)

# COMMAND ----------

tangent_job = tangent_jobs[i]
tangent_job_parameters = tangent_job['parameters']
v_data = tangent_features_df.loc[(tangent_features_df[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
print(tangent_job_parameters)
visualization.feature_importance(v_data)

# COMMAND ----------

fig = px.bar(tangent_properties_df[tangent_properties_df['importance']>0], x='id', y="rel_importance", color="name", barmode = 'stack',hover_data=group_keys)
fig.update_layout(height=800, width=1200, title_text="Evolution")
fig.show()
