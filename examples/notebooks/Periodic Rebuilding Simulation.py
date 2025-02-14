# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Example - Periodic Rebuilding Simulation

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Setup

# COMMAND ----------

import tangent_works
import pandas as pd
import numpy as np
import uuid
import datetime as dt

# COMMAND ----------

tw = tangent_works.TangentWorks()

# COMMAND ----------

# -------------------------------- Supporting Libraries --------------------------------

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

    def rca(time_series,timestamp_column,target_column,df,rca_tables_df,rca_timestamp,window=48):
        try:
            df = df.rename(columns={'normal_behavior':'forecast'})
        except:
            pass
        rca_index = df[df['timestamp']==rca_timestamp].index[0]
        v_data = time_series.iloc[rca_index-window:rca_index+window]
        v_results = df[(df['timestamp']>=v_data[timestamp_column].min())&(df['timestamp']<=v_data[timestamp_column].max())]
        v_rca = rca_tables_df[(rca_tables_df['type']=='yhat')&(rca_tables_df['timestamp']==rca_timestamp)]
        trace_list_1,trace_list_2,trace_list_3 = [],[],[]
        for i in range(len(v_rca)):
            yhat_df = pd.concat([v_results[v_results['timestamp']!=rca_timestamp][['timestamp','forecast']],pd.DataFrame(v_rca.iloc[i][['timestamp','value']]).transpose().rename(columns={'value':'forecast'})]).sort_values(by='timestamp')
            visibility = True if i==0 else False
            trace_list_1.append(go.Scatter(x=list(v_data[timestamp_column]),y=list(v_data[target_column]), visible=visibility, line={'color': 'black'},name='target'))
            trace_list_2.append(go.Scatter(x=list(v_results['timestamp']),y=list(v_results['forecast']), visible=visibility, line={'color': 'red'},name='forecast'))
            trace_list_3.append(go.Scatter(x=list(yhat_df['timestamp']),y=list(yhat_df['forecast']), visible=visibility, line={'color': 'orange'},name=v_rca.iloc[i]['term']))

        fig = go.Figure(data=trace_list_1+trace_list_2+trace_list_3)
        fig.add_trace(go.Scatter(x=v_data[timestamp_column], y=v_data[target_column], name=target_column, line=dict(color='black')))
        steps = []
        num_steps = len(trace_list_1)
        for i in range(num_steps):
            step = dict(method = 'restyle',args = ['visible', [False] * len(fig.data)])
            step['args'][1][i] = True
            step['args'][1][i+num_steps] = True
            step['args'][1][i+num_steps*2] = True
            steps.append(step)
        sliders = [dict(steps = steps,y= -0.05)]
        fig.layout.sliders = sliders 
        fig.add_vline(x=rca_timestamp, line_dash="dash", line_color="green")
        fig.update_layout(height=600,width=1200,title_text='Model Timestamp Analysis',legend=dict(y=-0.4,x=0.0,orientation='h'))
        fig.show(renderer='databricks')

# COMMAND ----------

  
class simulation:
    def _missing_values_check(df,timestamp,target,predictors,group_keys=[]):
        timestamp_df =  pd.DataFrame(pd.date_range(start=df[timestamp].min(),end=df[timestamp].max(),freq=pd.to_datetime(df[timestamp]).diff().median()),columns=[timestamp])
        if len(group_keys)>0:
            timestamp_df['link'] = 1
            combinations = df[group_keys].drop_duplicates().to_dict('records')
            combinations_df = pd.DataFrame(combinations)
            combinations_df['link'] = 1     
            recombine_df = combinations_df.merge(timestamp_df,on='link',how='left').drop(columns=['link']).merge(df,on=[timestamp]+group_keys,how='left')
        else:
            recombine_df = timestamp_df.merge(df,on=[timestamp]+group_keys,how='left')
        return pd.concat([recombine_df[group_keys+[timestamp]],recombine_df[[target]+predictors].isnull()],axis=1).melt(id_vars=group_keys+[timestamp],value_vars=[target]+predictors,value_name='missing')
        

    def alignment_check(df,timestamp,target,predictors,group_keys=[]):
        missing_values_df = simulation._missing_values_check(
            df = df,
            timestamp = timestamp,
            target = target,
            predictors = predictors
            )
        output = missing_values_df[missing_values_df['missing']==False].groupby(group_keys+['variable'])[[timestamp]].max().sort_values(by=group_keys+[timestamp]).reset_index()
        last_target_timestamp = output[output['variable']==target][timestamp].values[0]
        output['delta'] = output[timestamp] - last_target_timestamp
        sample_rate = df[timestamp].diff().median()
        output['estimated_samples'] = (output['delta']/sample_rate).astype(int)
        return output

    def pov_dataset(pov_datetime,dataframe,timestamp_column,target_column,predictors,alignment_records):
        data_alignment = [{'column_name': target_column,'timestamp': pov_datetime}]
        time_series = tangent_works.utils.time_series.TimeSeries(dataframe)
        sampling_period_value = time_series.sampling_period.value
        for predictor in predictors:
            samples = [f for f in alignment_records if f['variable']==predictor][0]['estimated_samples']
            predictor_timestamp = str(pd.to_datetime(pov_datetime)+dt.timedelta(seconds=sampling_period_value*samples))
            data_alignment.append({'column_name': predictor,'timestamp': predictor_timestamp})

        new = [dataframe[dataframe[timestamp_column]<=max([f['timestamp'] for f in data_alignment])][timestamp_column]]
        for col in data_alignment:
            new.append(tangent_dataframe[tangent_dataframe[timestamp_column]<=col['timestamp']][col['column_name']])
        new_df = pd.concat(new,axis=1)
        return new_df


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Data

# COMMAND ----------

file_path = '/Workspace'+dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 2)[0]+'/data/belgian_grid_load.csv'
tangent_dataframe = pd.read_csv(file_path)
group_keys = []
timestamp_column = "datetime"
target_column = "load"
predictors = [s for s in list(tangent_dataframe.columns) if s not in group_keys + [timestamp_column, target_column]]
tangent_dataframe = tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors].sort_values(by=group_keys + [timestamp_column]).reset_index(drop=True)
tangent_dataframe[timestamp_column] = pd.to_datetime(pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S"))
tangent_dataframe

# COMMAND ----------

visualization.data(df=tangent_dataframe.tail(4*24*30),timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

alignment_df = simulation.alignment_check(tangent_dataframe,timestamp_column,target_column,predictors)
alignment_records = alignment_df[['variable','estimated_samples']].to_dict('records')
pd.DataFrame(alignment_records)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Simulation

# COMMAND ----------

# Prepare Model building Jobs
model_build_pov_datetimes = pd.date_range(start='2022-12-30 23:45:00',end='2022-12-31 23:45:00',freq='D')
build_model_tangent_jobs = []
for pov_datetime in model_build_pov_datetimes:
    build_model_tangent_jobs.append({'id':str(uuid.uuid4()),'parameters':{'pov_datetime':str(pov_datetime)}})

# COMMAND ----------

# Prepare predict Jobs
predict_pov_datetimes = pd.date_range(start='2022-12-30 23:45:00',end='2022-12-31 23:45:00',freq='15min')
predict_tangent_jobs = []
for pov_datetime in predict_pov_datetimes:
    model_id = max([f for f in build_model_tangent_jobs if pd.to_datetime(f['parameters']['pov_datetime'])<=pov_datetime], key=lambda x: x['parameters']['pov_datetime'])['id']
    predict_tangent_jobs.append({'id':str(uuid.uuid4()),'parameters':{'pov_datetime':str(pov_datetime),'model_id':model_id}})

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Tangent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Model Building Simulation

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
        'value': 4*24*2
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

tw_spark = tangent_works.SparkParallelProcessing(app_name='Example')
spark_jobs = []
for tangent_job in build_model_tangent_jobs:
    tangent_job_parameters = tangent_job['parameters']
    pov_datetime = tangent_job_parameters['pov_datetime']
    dataset = simulation.pov_dataset(pov_datetime,tangent_dataframe,timestamp_column,target_column,predictors,alignment_records)
    spark_jobs.append((tangent_job['id'],tw.forecasting.build_model,{'dataset':dataset,'configuration':build_model_configuration}))
tw_parallel_model_building = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Prediction Simulation

# COMMAND ----------

predict_configuration = {
    'prediction_from': {
        'base_unit': 'sample',
        'value': 1
        },
    'prediction_to': {
        'base_unit': 'sample',
        'value': 4*24*2
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

spark_jobs = []
for tangent_job in predict_tangent_jobs:
    tangent_job_parameters = tangent_job['parameters']
    pov_datetime = tangent_job_parameters['pov_datetime']
    model_id = tangent_job_parameters['model_id']
    data = simulation.pov_dataset(pov_datetime,tangent_dataframe,timestamp_column,target_column,predictors,alignment_records)
    model = [m['result'] for m in tw_parallel_model_building if m['id']==model_id][0]
    spark_jobs.append((tangent_job['id'],tw.forecasting.predict,{'dataset':dataset,'configuration':predict_configuration,'model':model}))
tw_parallel_prediction = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Results

# COMMAND ----------

all_result_tables = []
for tangent_job in predict_tangent_jobs:
    job_id = tangent_job['id']
    result_table = [f['result'] for f in tw_parallel_prediction if f['id']==job_id][0]
    result_table['id'] = job_id
    all_result_tables.append(result_table)

tangent_jobs_df = pd.json_normalize(predict_tangent_jobs)
tangent_jobs_df.columns = tangent_jobs_df.columns.str.replace('parameters.','',regex=False)

tangent_result_tables_df = pd.concat(all_result_tables).merge(tangent_jobs_df,on='id',how='left')

# COMMAND ----------

all_properties,all_features = [],[]
for tangent_job in build_model_tangent_jobs:
    job_id = tangent_job['id']
    model = [f['result'] for f in tw_parallel_model_building if f['id']==job_id][0].to_dict()
    
    properties_df = tw.insights.properties(model=model)
    properties_df['id'] = job_id
    all_properties.append(properties_df)

    features_df = tw.insights.features(model=model)
    features_df['id'] = job_id
    all_features.append(features_df)

tangent_jobs_df = pd.json_normalize(build_model_tangent_jobs)
tangent_jobs_df.columns = tangent_jobs_df.columns.str.replace('parameters.','',regex=False)

tangent_properties_df = pd.concat(all_properties).merge(tangent_jobs_df,on='id',how='left')
tangent_features_df = pd.concat(all_features).merge(tangent_jobs_df,on='id',how='left')

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Visualization

# COMMAND ----------

i = 10
v_data = tangent_result_tables_df[tangent_result_tables_df['pov_datetime']==tangent_result_tables_df['pov_datetime'].unique()[i]]
fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=tangent_dataframe[timestamp_column], y=tangent_dataframe[target_column], name='target',line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['forecast'], name='simulation',line=dict(color='goldenrod')), row=1, col=1)
fig.update_layout(height=500, width=1000, title_text="Results")
fig.show(renderer='databricks')

# COMMAND ----------

fig = px.bar(tangent_properties_df[tangent_properties_df['importance']>0], x='pov_datetime', y="rel_importance", color="name", barmode = 'stack',hover_data=group_keys)
fig.update_layout(height=500, width=1200, title_text="Evolution")
fig.show(renderer='databricks')

# COMMAND ----------

i = 0
features_df = tangent_features_df[tangent_features_df['pov_datetime']==tangent_features_df['pov_datetime'].unique()[i]]
visualization.feature_importance(features_df)
