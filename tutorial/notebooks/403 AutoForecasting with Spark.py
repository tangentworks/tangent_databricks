# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - AutoForecasting with Spark

# COMMAND ----------

# MAGIC %md
# MAGIC In this tutorial, you will learn to build autoforecasting models with Tangent, at scale, by leveraging Spark.  
# MAGIC
# MAGIC We will send multiple requests to Tangent in parallel using Spark RDD's. This will allow Databricks to scale the Tangent custom Docker container to handle all the requests. Databricks will then manage all these individual jobs automatically and send results back when all are finished. Dependent on the number of workers are available in the cluster, more jobs can be processed simultaneously. 
# MAGIC
# MAGIC To show these capabilities, we will use an example dataset from a retail sales forecasting use case.  
# MAGIC The goal is to forecast sales of several products across different locations 4 weeks ahead using historical sales data and other explanatory variables.

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Setup

# COMMAND ----------

# MAGIC %md
# MAGIC First, import the tangent_works package and other supporting libraries.

# COMMAND ----------

import tangent_works as tw
import pandas as pd
import uuid

# COMMAND ----------

# MAGIC %md
# MAGIC To visualize the results of this exercise, the following visualization functions can be used.

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

# MAGIC %md
# MAGIC The dataset that will be used in this notebook is called center_meals.  
# MAGIC It contains historical daily sales data and several data points of price and promotion information.  
# MAGIC In the cell below, this dataset is preprocessed and made ready for use with Tangent. Important here is to define the group_keys which are used to identify the individual timeseries in the larger dataset. Here center_id & meal_id indicate the individual product sales at each location. For this example we choose to analyze the sales of the top 5 products at the top 5 stores resulting in 25 jobs to be executed in Tangent.

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

# MAGIC %md
# MAGIC Below we visualize the time series of 1 location/product combination. 

# COMMAND ----------

i = 0
v_data = tangent_dataframe.loc[(tangent_dataframe[list(combinations[i])] == pd.Series(combinations[i])).all(axis=1)]
visualization.data(df=v_data,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have a dataset to analyze, let's describe the general configuration that will be applied to each of the individual model builds in this exercise. A forecasting horizon of 4 samples (or weeks in this case) is set. As an example, specific mathematical settings are used. For more details see tutorial 301 - Mathematical Settings

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
# MAGIC In the Tangent Python Databricks package, there exists the SparkParallelProcessing class which allows the user to send jobs to Tangent in parallel. The forecasting process can be parallelized in different ways. In this example we show how to create a function that covers all the steps in the forecasting process in series and this specific function is ran simultaneously in databricks. 
# MAGIC
# MAGIC Please note that these invididual steps can also be ran in parallel. That remains up to the preference of the user. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jobs

# COMMAND ----------

# MAGIC %md
# MAGIC First, to identify each job and link the resulting information back to our use case, tangent_job objects are created containing a unique ID and user specified parameters. These parameters can be anything that the user wants to share to identify the individual jobs. Here the parameters are the combinations of location and product id.

# COMMAND ----------

tangent_jobs = []
for combination in combinations[:]:
    tangent_jobs.append({'id':str(uuid.uuid4()),'parameters':combination})

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoForecast

# COMMAND ----------

# MAGIC %md
# MAGIC Below shows an example user defined function that will be process by Spark and Tangent.

# COMMAND ----------

class user_defined:
    def run_auto_forecast(data,configuration):
        time_series = tw.TimeSeries(data)
        time_series.validate()
        tangent_auto_forecast = tw.AutoForecasting(time_series=time_series, configuration=configuration)
        tangent_auto_forecast.run()
        return tangent_auto_forecast

# COMMAND ----------

# MAGIC %md
# MAGIC Here we bring together the general configuration and the individual time series that are filtered from the large dataset using the parameters. Then the user defined functions are sent to Spark and the jobs are executed in parallel.

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

# MAGIC %md
# MAGIC Now, Tangent has generated all results and they need to be processed. The following loop extracts the properties, features and predictions from all tangent_jobs. This step could be parallelized as well if required. 

# COMMAND ----------

all_properties,all_features,all_result_tables = [],[],[]
for tangent_job in tangent_jobs:
    job_id = tangent_job['id']
    result = [f['result'] for f in tw_parallel_auto_forecasting if f['id']==job_id][0]
    model = result.model.to_dict()
    
    properties_df = tw.PostProcessing().properties(model=model)
    properties_df['id'] = job_id
    all_properties.append(properties_df)

    features_df = tw.PostProcessing().features(model=model)
    features_df['id'] = job_id
    all_features.append(features_df)

    result_table = tw.PostProcessing().result_table(result)
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
tangent_job_parameters

# COMMAND ----------

# MAGIC %md
# MAGIC ## predictions

# COMMAND ----------

v_data = tangent_result_tables_df.loc[(tangent_result_tables_df[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
v_actuals = tangent_dataframe.loc[(tangent_dataframe[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=group_keys)
visualization.predictions(v_data,v_actuals,target_column=target_column,timestamp_column=timestamp_column)

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
# MAGIC The properties of all combinations could also be outlined next to each other and compared. In this example that could provide insights about which combinations of location and products sales are affected by certain predictors.

# COMMAND ----------

fig = px.bar(tangent_properties_df[tangent_properties_df['importance']>0], x='id', y="rel_importance", color="name", barmode = 'stack',hover_data=group_keys)
fig.update_layout(height=500, width=1200, title_text="All properties")
fig.show()
