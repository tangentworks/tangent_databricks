# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - AutoForecasting At Scale

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

    def predictions(df,v_actuals,timestamp_column,target_column):
        fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=v_actuals[timestamp_column], y=v_actuals[target_column], name='target',line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['forecast'], name='forecast',line=dict(color='goldenrod')), row=1, col=1)
        fig.update_layout(height=500, width=1000, title_text="Results")
        fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #1. Data

# COMMAND ----------

file_path = '/Workspace'+dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 2)[0]+'/data/center_meals.csv'
csv_df = pd.read_csv(file_path)
df1 = csv_df[csv_df['center_id'].isin(list(csv_df.groupby('center_id')[['num_orders']].mean().sort_values(by='num_orders').tail().index))]
tangent_dataframe = df1[df1['meal_id'].isin(list(df1.groupby('meal_id')[['num_orders']].mean().sort_values(by='num_orders').tail().index))]

group_keys = ['center_id','meal_id']
timestamp_column = "Timestamp"
target_column = "num_orders"
predictors = [s for s in list(tangent_dataframe.columns) if s not in group_keys + [timestamp_column, target_column]]
tangent_dataframe = tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors].sort_values(by=group_keys + [timestamp_column]).reset_index(drop=True)
tangent_dataframe[timestamp_column] = pd.to_datetime(pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S"))
combinations = tangent_dataframe[group_keys].drop_duplicates().to_dict('records')
print(dict(tangent_dataframe[group_keys].nunique()))
print('combinations:',len(combinations))
tangent_dataframe

# COMMAND ----------

i = 0
v_data = tangent_dataframe.loc[(tangent_dataframe[list(combinations[i])] == pd.Series(combinations[i])).all(axis=1)]
visualization.data(df=v_data,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Configuration

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
    'engine': {
        # 'target_column': target_column,
        # 'holiday_column': 'string',
        'prediction_from': {'base_unit': 'sample','value': 1},
        'prediction_to': {'base_unit': 'sample','value': 4},
        # 'target_offsets': 'combined',
        # 'predictor_offsets': 'common',
        # 'allow_offsets': True,
        # 'max_offsets_depth': 0,
        # 'normalization': True,
        'max_feature_count': 100,
        'transformations': [
            'exponential_moving_average',
            'rest_of_week',
            'periodic',
            'intercept',
            'piecewise_linear',
            'time_offsets',
            'polynomial',
            'identity',
            'simple_moving_average',
            'month',
            'trend',
            'day_of_week',
            # 'fourier',
            # 'public_holidays',
            # 'one_hot_encoding'
        ],
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
    }
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
# MAGIC ## AutoForecast

# COMMAND ----------

class user_defined:
    def run_auto_forecast(data,configuration):
        time_series = tw.TimeSeries(data)
        time_series.validate()
        tangent_auto_forecast = tw.AutoForecasting(time_series=time_series, configuration=configuration)
        tangent_auto_forecast.run()
        result_table = tangent_auto_forecast.result_table
        model = tangent_auto_forecast.model.to_dict()
        return {'result_table':result_table,'model':model}

# COMMAND ----------

tw_spark = tw.SparkParallelProcessing()
spark_jobs = []
for tangent_job in tangent_jobs:
    tangent_job_parameters = tangent_job['parameters']
    data = tangent_dataframe.loc[(tangent_dataframe[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
    spark_jobs.append((tangent_job['id'],user_defined.run_auto_forecast,{'data':data,'configuration':auto_forecasting_configuration}))
tw_parallel_auto_forecasting = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Results

# COMMAND ----------

all_properties,all_features,all_result_tables = [],[],[]
for tangent_job in tangent_jobs:
    job_id = tangent_job['id']
    result = [f['result'] for f in tw_parallel_auto_forecasting if f['id']==job_id][0]
    model = result['model']
    
    properties_df = tw.PostProcessing().properties(response=model)
    properties_df['id'] = job_id
    all_properties.append(properties_df)

    features_df = tw.PostProcessing().features(response=model)
    features_df['id'] = job_id
    all_features.append(features_df)

    result_table = result['result_table']
    result_table['id'] = job_id
    all_result_tables.append(result_table)

tangent_jobs_df = pd.json_normalize(tangent_jobs)
tangent_jobs_df.columns = tangent_jobs_df.columns.str.replace('parameters.','')

tangent_result_tables_df = pd.concat(all_result_tables).merge(tangent_jobs_df,on='id',how='left')
tangent_properties_df = pd.concat(all_properties).merge(tangent_jobs_df,on='id',how='left')
tangent_features_df = pd.concat(all_features).merge(tangent_jobs_df,on='id',how='left')

# COMMAND ----------

# MAGIC %md
# MAGIC #5. Visualization

# COMMAND ----------

i = 2
tangent_job = tangent_jobs[i]
tangent_job_parameters = tangent_job['parameters']

v_data = tangent_result_tables_df.loc[(tangent_result_tables_df[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
v_actuals = tangent_dataframe.loc[(tangent_dataframe[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
visualization.predictions(v_data,v_actuals,target_column=target_column,timestamp_column=timestamp_column)

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
