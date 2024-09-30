# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - Simulation

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Setup

# COMMAND ----------

import tangent_works as tw
import pandas as pd
import numpy as np
import datetime as dt
import uuid

# COMMAND ----------

# -------------------------------- Supporting Libraries --------------------------------

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as splt

class visualization:

    def predictions(df):
        fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        color_map = {'training':'green','testing':'red','production':'goldenrod'}
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['target'], name='target',line=dict(color='black')), row=1, col=1)
        for forecasting_type in df['type'].unique():
            v_data = df[df['type']==forecasting_type].copy()
            fig.add_trace(go.Scatter(x=v_data['timestamp'], y=v_data['forecast'], name=forecasting_type,line=dict(color=color_map[forecasting_type])), row=1, col=1)
        fig.update_layout(height=500, width=1000, title_text="Results")
        fig.show()

    def data(df,timestamp,target,predictors):
        fig = splt.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=df[timestamp], y=df[target], name=target,connectgaps=True), row=1, col=1)
        for idx, p in enumerate(predictors): fig.add_trace(go.Scatter(x=df[timestamp], y=df[p], name=p,connectgaps=True), row=2, col=1)
        fig.update_layout(height=600, width=1100, title_text="Data visualization")
        fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #1. Data

# COMMAND ----------

file_path = '/Workspace'+dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 2)[0]+'/data/belgian_electricity_grid.csv'
tangent_dataframe = pd.read_csv(file_path)
group_keys = []
timestamp_column = "Timestamp"
target_column = "Quantity"
predictors = [s for s in list(tangent_dataframe.columns) if s not in group_keys + [timestamp_column, target_column]]
tangent_dataframe = tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors].sort_values(by=group_keys + [timestamp_column]).reset_index(drop=True)
tangent_dataframe[timestamp_column] = pd.to_datetime(pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S"))
tangent_dataframe

# COMMAND ----------

visualization.data(df=tangent_dataframe,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------



# COMMAND ----------

def missing_values_check(df,timestamp,target,predictors,group_keys=[]):
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
    missing_values_df = missing_values_check(
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
    time_series = tw.TimeSeries(dataframe)
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
# MAGIC # 2. Configuration

# COMMAND ----------

alignment_df = alignment_check(tangent_dataframe,timestamp_column,target_column,predictors)
alignment_records = alignment_df[['variable','estimated_samples']].to_dict('records')
alignment_records

# COMMAND ----------

auto_forecasting_configuration = {
    'engine': {
        'prediction_from': {'base_unit': 'day','value': 1},
        'prediction_to': {'base_unit': 'day','value': 1},
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC #3. Tangent

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jobs

# COMMAND ----------

pov_datetimes = pd.date_range(start='2021-10-16 23:00:00',end='2022-01-15 23:00:00',freq='D')
tangent_jobs = []
for pov_datetime in pov_datetimes:
    tangent_jobs.append({'id':str(uuid.uuid4()),'parameters':{'pov_datetime':str(pov_datetime)}})

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
        return tangent_auto_forecast

# COMMAND ----------

tw_spark = tw.SparkParallelProcessing()
spark_jobs = []
for tangent_job in tangent_jobs:
    tangent_job_parameters = tangent_job['parameters']
    pov_datetime = tangent_job_parameters['pov_datetime']
    data = pov_dataset(pov_datetime,tangent_dataframe,timestamp_column,target_column,predictors,alignment_records)
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

simulation_results_df = tangent_result_tables_df[['pov_datetime','timestamp','forecast','type']].dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Visualization

# COMMAND ----------

fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=tangent_dataframe[timestamp_column], y=tangent_dataframe[target_column], name='target',line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=simulation_results_df['timestamp'], y=simulation_results_df['forecast'], name='simulation',line=dict(color='goldenrod')), row=1, col=1)
fig.update_layout(height=500, width=1000, title_text="Results")
fig.show()

# COMMAND ----------

fig = px.bar(tangent_properties_df[tangent_properties_df['importance']>0], x='pov_datetime', y="rel_importance", color="name", barmode = 'stack',hover_data=group_keys)
fig.update_layout(height=500, width=1200, title_text="Evolution")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Comparison

# COMMAND ----------

backtest_configuration = {
    'preprocessing': {
        'training_rows': [{'from': '2021-01-20 00:00:00','to': '2021-10-16 23:00:00'}],
        'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
    },
    'engine': {
        'prediction_from': {'base_unit': 'day','value': 1},
        'prediction_to': {'base_unit': 'day','value': 1},
    }
}
backtest = user_defined.run_auto_forecast(data=tangent_dataframe,configuration=backtest_configuration)
backtest_result_table_df = tw.PostProcessing().result_table(forecasting=backtest)

# COMMAND ----------

visualization.predictions(backtest_result_table_df)

# COMMAND ----------

comparison_df = tangent_dataframe[[timestamp_column,target_column]].rename(columns={timestamp_column:'timestamp',target_column:'target'}).merge(
    simulation_results_df[['timestamp','forecast']].rename(columns={'forecast':'simulation'}),
    on='timestamp',
    how='left'
).merge(
    backtest_result_table_df[backtest_result_table_df['type']=='testing'][['timestamp','forecast']].rename(columns={'forecast':'backtest'}),
    on='timestamp',
    how='left'
)

# COMMAND ----------

accuracy_comparison_df = comparison_df[(comparison_df['target'].notnull())&(comparison_df['simulation'].notnull())&(comparison_df['backtest'].notnull())].copy()
accuracy_comparison_df['simulation_MAE'] = abs(accuracy_comparison_df['simulation'] - accuracy_comparison_df['target'])
accuracy_comparison_df['backtest_MAE'] = abs(accuracy_comparison_df['backtest'] - accuracy_comparison_df['target'])

# COMMAND ----------

fig = splt.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=accuracy_comparison_df['timestamp'], y=accuracy_comparison_df['target'], name='target',line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=accuracy_comparison_df['timestamp'], y=accuracy_comparison_df['simulation'], name='simulation',line=dict(color='goldenrod')), row=1, col=1)
fig.add_trace(go.Scatter(x=accuracy_comparison_df['timestamp'], y=accuracy_comparison_df['backtest'], name='backtest',line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=accuracy_comparison_df['timestamp'], y=accuracy_comparison_df['simulation_MAE'], name='simulation_MAE',line=dict(color='goldenrod')), row=2, col=1)
fig.add_trace(go.Scatter(x=accuracy_comparison_df['timestamp'], y=accuracy_comparison_df['backtest_MAE'], name='backtest_MAE',line=dict(color='red')), row=2, col=1)
fig.add_trace(go.Scatter(x=accuracy_comparison_df['timestamp'], y=accuracy_comparison_df['simulation_MAE'].cumsum(), name='simulation_CMAE',line=dict(color='goldenrod')), row=3, col=1)
fig.add_trace(go.Scatter(x=accuracy_comparison_df['timestamp'], y=accuracy_comparison_df['backtest_MAE'].cumsum(), name='backtest_CMAE',line=dict(color='red')), row=3, col=1)
fig.update_layout(height=800, width=1000, title_text="Comparison")
fig.show()
