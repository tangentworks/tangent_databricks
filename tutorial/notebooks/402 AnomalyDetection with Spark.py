# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - AnomalyDetection with Spark

# COMMAND ----------

# MAGIC %md
# MAGIC In this tutorial, you will learn to build anomaly detection models with Tangent, at scale, by leveraging Spark.  
# MAGIC
# MAGIC We will send multiple requests to Tangent in parallel using Spark RDD's. This will allow Databricks to scale the Tangent custom Docker container to handle all the requests. Databricks will then manage all these individual jobs automatically and send results back when all are finished. Dependent on the number of workers are available in the cluster, more jobs can be processed simultaneously. 
# MAGIC
# MAGIC To show these capabilities, we will use an example dataset with sensor data from electrical equipement.
# MAGIC The goal is to detect anomalies across each sensor.

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

    def detections(df):
        fig = splt.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['target'], name='target',line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['normal_behavior'], name='normal_behavior',line=dict(color='goldenrod')), row=1, col=1)
        for column in [f for f in df.columns if f not in ['timestamp','target','normal_behavior']]:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[column], name=column), row=2, col=1)
            va = df[df[column]>=1]
            fig.add_trace(go.Scatter(x=va['timestamp'], y=va['target'], name=column.replace('anomaly_indicator_','')+' anomaly',mode='markers', line={'color': 'red'}), row=1, col=1)
        fig.add_hline(y=1,row=2,line=dict(color='red'))
        fig.update_layout(height=800, width=1000, title_text="Results")
        fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #1. Data

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset that will be used in this notebook is called phase_analysis.  
# MAGIC It contains 18 sensor measurements from electrical equipment that has been anonymized.  
# MAGIC
# MAGIC In the cell below, this dataset is preprocessed and made ready for use with Tangent. In this example we will generate an anomaly detection model for each sensor meaning each variable will be set as the target once and all other columns will be used as predictors. Below the dataset is preprocessed in general and made ready for the first sensor. Further in the script all other targets will be set.

# COMMAND ----------

file_path = '/Workspace'+dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 2)[0]+'/data/phase_analysis.csv'
tangent_dataframe = pd.read_csv(file_path).tail(10000)
group_keys = []
timestamp_column = "timestamp"
target_column = "S0_1"
predictors = [s for s in list(tangent_dataframe.columns) if s not in group_keys + [timestamp_column, target_column]]
tangent_dataframe = tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors].sort_values(by=group_keys + [timestamp_column]).reset_index(drop=True)
tangent_dataframe[timestamp_column] = pd.to_datetime(pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S"))
tangent_dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC Below we visualize the time series with S0_1 as the target and all other columns as predictors. 

# COMMAND ----------

visualization.data(df=tangent_dataframe.head(10000),timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC A general model building configuration is created below that will be applied to all targets. For simplicity, in this exercise, only the default residuals perspective is specified in the detection layers.

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
    'detection_layers': [
        {
            'residuals_transformation':{
                'type':'residuals'
            },
    #         'sensitivity':0.3
        },
    #     {
    #         'residuals_transformation':{
    #             'type':'residuals_change',
    # #             'window_length':2
    #         },
    # #         'sensitivity':0.3
    #     },
    #     {
    #         'residuals_transformation':{
    #             'type':'moving_average',
    # #             'window_length':1
    #         },
    # #         'sensitivity':0.3
    #     },
    #     {
    #         'residuals_transformation':{
    #             'type':'moving_average_change',
    # #             'window_lengths':[
    # #                 2,
    # #                 1
    # #             ]
    #         },
    # #         'sensitivity':0.3
    #     },
    #     {
    #         'residuals_transformation':{
    #             'type':'standard_deviation',
    # #             'window_length':1
    #         },
    # #         'sensitivity':0.3
    #     },
    #     {
    #         'residuals_transformation':{
    #             'type':'standard_deviation_change',
    # #             'window_lengths':[
    # #                 2,
    # #                 1
    # #             ]
    #         },
    # #         'sensitivity':0.3
    #     },
    ]
}

# COMMAND ----------

# MAGIC %md
# MAGIC #3. Tangent

# COMMAND ----------

# MAGIC %md
# MAGIC With the data and configuration ready, we can prepare the parallel modeling process. 
# MAGIC
# MAGIC In the Tangent Python Databricks package, there exists the SparkParallelProcessing class which allows the user to send jobs to Tangent in parallel. The forecasting process can be parallelized in different ways. In this example we show how to run in parallel each step in the forecasting process seperately and accross all the combinations.  
# MAGIC
# MAGIC Please note that a function can also be created that handles these invididual steps in series and then this function could be ran in parallel. That remains up to the preference of the user. Also, this process outlines the general steps of running multiple anomaly detection jobs in parallel and this flow could be modified to cover specific cases where different jobs need to run in parallel.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jobs

# COMMAND ----------

# MAGIC %md
# MAGIC First, to identify each job and link the resulting information back to our use case, tangent_job objects are created containing a unique ID and user specified parameters. These parameters can be anything that the user wants to share to identify the individual jobs. Here the parameters are the different column names to be set as a target.

# COMMAND ----------

tangent_jobs = []
for column in [target_column]+predictors:
    tangent_jobs.append({'id':str(uuid.uuid4()),'parameters':{'target_column':column}})

# COMMAND ----------

# MAGIC %md
# MAGIC ## TimeSeries

# COMMAND ----------

# MAGIC %md
# MAGIC In this example, multiple jobs are created using the same dataset. Therefore the timeseries needs to be created only once.

# COMMAND ----------

tw_timeseries = tw.TimeSeries(data=tangent_dataframe)
tw_timeseries.validate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## AnomalyDetection

# COMMAND ----------

# MAGIC %md
# MAGIC The next step in the anomaly detection process is to create AnomalyDetection objects by combining the timeseries' with model building configurations.

# COMMAND ----------

tw_spark = tw.SparkParallelProcessing()
spark_jobs = []
for tangent_job in tangent_jobs:
  configuration = build_anomaly_detection_configuration.copy()
  configuration.update({'normal_behavior': {'target_column':tangent_job['parameters']['target_column']}})
  spark_jobs.append((tangent_job['id'],tw.AnomalyDetection,{'time_series':tw_timeseries,'configuration':configuration}))
tw_parallel_AnomalyDetection = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Building

# COMMAND ----------

# MAGIC %md
# MAGIC Now, the jobs are ready to be sent to Tangent to build anomaly detection models. The build_model functions are sent to Spark and Databricks will start building multiple models simultaneously. 

# COMMAND ----------

spark_jobs = []
for tw_anomaly_detection in tw_parallel_AnomalyDetection:
    spark_jobs.append((tw_anomaly_detection['id'],tw_anomaly_detection['result'].build_model,{'inplace':False}))
tw_anomaly_detection_build_models = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detection

# COMMAND ----------

# MAGIC %md
# MAGIC When all model building jobs are finished, detections can be made using these models. These can also be sent to Tangent in parallel.

# COMMAND ----------

spark_jobs = []
for tw_anomaly_detection_build_model in tw_anomaly_detection_build_models:
    spark_jobs.append((tw_anomaly_detection_build_model['id'],tw_anomaly_detection_build_model['result'].detect,{}))
tw_anomaly_detection_detects = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Results

# COMMAND ----------

# MAGIC %md
# MAGIC Now, Tangent has generated all results and they need to be processed. The following loop extracts the properties, features and detections from all tangent_jobs. This step could be parallelized as well if required. 

# COMMAND ----------

all_properties,all_features,all_result_table_df = [],[],[]
for tangent_job in tangent_jobs:
    job_id = tangent_job['id']
    result = [f['result'] for f in tw_anomaly_detection_build_models if f['id']==job_id][0]
    model = result.model.to_dict()
    
    properties_df = tw.PostProcessing().properties(model=model)
    properties_df['id'] = job_id
    all_properties.append(properties_df)

    features_df = tw.PostProcessing().features(model=model)
    features_df['id'] = job_id
    all_features.append(features_df)

    result_table_df = [f['result'] for f in tw_anomaly_detection_detects if f['id']==job_id][0]
    result_table_df['id'] = job_id
    all_result_table_df.append(result_table_df)

tangent_jobs_df = pd.json_normalize(tangent_jobs)
tangent_jobs_df.columns = tangent_jobs_df.columns.str.replace('parameters.','')

tangent_result_table_df = pd.concat(all_result_table_df).merge(tangent_jobs_df,on='id',how='left')
tangent_properties_df = pd.concat(all_properties).merge(tangent_jobs_df,on='id',how='left')
tangent_features_df = pd.concat(all_features).merge(tangent_jobs_df,on='id',how='left')

# COMMAND ----------

# MAGIC %md
# MAGIC #5. Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC Now, the user can inspect the results from the parallel exercise. Below, a single target is selected from the list and those results are visualized below.

# COMMAND ----------

i = 0
tangent_job = tangent_jobs[i]
tangent_job_parameters = tangent_job['parameters']

# COMMAND ----------

# MAGIC %md
# MAGIC ## detections

# COMMAND ----------

v_data = tangent_result_table_df.loc[(tangent_result_table_df[list(tangent_job_parameters)] == pd.Series(tangent_job_parameters)).all(axis=1)].drop(columns=['id']+list(tangent_job_parameters.keys()))
print(tangent_job_parameters)
visualization.detections(v_data)

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
# MAGIC The properties of all targets could also be outlined next to each other and compared. In this example that could provide insights about which targets are affected by other predictors.

# COMMAND ----------

fig = px.bar(tangent_properties_df[tangent_properties_df['importance']>0], x='id', y="rel_importance", color="name", barmode = 'stack',hover_data=group_keys)
fig.update_layout(height=500, width=1200, title_text="All properties")
fig.show()
