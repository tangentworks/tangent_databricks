# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - Simulation

# COMMAND ----------

# MAGIC %md
# MAGIC In this tutorial, you will learn how to perfom a simulation with Tangent.  
# MAGIC Simulations are necessary to get a complete insight about the performance of Tangent in a use case.
# MAGIC
# MAGIC Since generating results with Tangent happens so quickly, Tangent is often set up in a __continuous rebuilding__ forecasting process.  
# MAGIC This means that a simple backtest such as in the previous tutorial is not enough to compare historical actuals to predictions from Tangent.  
# MAGIC Because there, only 1 model was built and 1 inference was made over a testing period. 
# MAGIC
# MAGIC In simulations, we replicate a realistic forecasting scenario in production.  
# MAGIC From the point of view of certain dates in the past, we generate models and simulate the same predictions as we would have received them if we had ran a production forecast on those dates.  
# MAGIC We prepare the dataset and configuration in such a way so that the forecast results generated now will correspond exactly with predictions that would have been created in the past without data leakage.
# MAGIC
# MAGIC The simulation covers a testing period that is defined by the user and production forecast jobs are executed on a rolling basis throughout this period.  
# MAGIC With Tangent these jobs can run simultaneously using Spark to speed up the simulation process and quickly get an insight in how Tangent would have performed over the testing period.

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Setup

# COMMAND ----------

# MAGIC %md
# MAGIC First, import the tangent_works package and other supporting libraries.

# COMMAND ----------

import tangent_works as tw
import pandas as pd
import numpy as np
import datetime as dt
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

    def comparison(df):
        fig = splt.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['target'], name='target',line=dict(color='black')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['simulation'], name='simulation',line=dict(color='goldenrod')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['backtest'], name='backtest',line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['simulation_MAE'], name='simulation_MAE',line=dict(color='goldenrod')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['backtest_MAE'], name='backtest_MAE',line=dict(color='red')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['simulation_MAE'].cumsum(), name='simulation_CMAE',line=dict(color='goldenrod')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['backtest_MAE'].cumsum(), name='backtest_CMAE',line=dict(color='red')), row=3, col=1)
        fig.update_layout(height=800, width=1000, title_text="Comparison")
        fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we can introduce some logic that will help with the simulation process.

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
# MAGIC #1. Data

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset that will be used in this notebook is called belgian_electricity_grid.  
# MAGIC It contains historical electricity consumption and weather data as well as weather forecasts and public holiday information.  
# MAGIC In the cell below, this dataset is preprocessed and made ready for use with Tangent.

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

# MAGIC %md
# MAGIC In time series analysis backtesting, when exploring a dataset, it is best practice to visualize the data and learn which patterns might exists in the data that we want Tangent to identify automatically.  
# MAGIC In this graph, the target column "Quantity" is visualized above and the additional explanatory variables or predictors are visualized below.  
# MAGIC

# COMMAND ----------

visualization.data(df=tangent_dataframe,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that for some predictors, values are available ahead of the last target timestamp throughout the forecast horizon. This is important to remember since Tangent will automatically recognize the data situation from the relative alignment to the target.  
# MAGIC However, in the context of a simulation it is important to make sure the same alignment is used during the rolling model building as it would have in a production scenario.

# COMMAND ----------

alignment_df = simulation.alignment_check(tangent_dataframe,timestamp_column,target_column,predictors)
alignment_records = alignment_df[['variable','estimated_samples']].to_dict('records')
pd.DataFrame(alignment_records)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC Here we define a general configuration that will be used across all jobs in the simulation. It will use the default settings and request a day-ahead forecast to Tangent.

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

# MAGIC %md
# MAGIC Using the AutoForecasting with Spark process we can define a number of Tangent jobs that each represent a forecast executed from a different point of view in the past.  
# MAGIC Here we defined a testing period between '2021-12-17 23:00:00' and '2022-01-15 23:00:00' where jobs will be executed on a daily basis at 23:00 between those two dates.   
# MAGIC This will result in 30 unique models and predictions that will be calculated simultaneously.

# COMMAND ----------

pov_datetimes = pd.date_range(start='2021-12-17 23:00:00',end='2022-01-15 23:00:00',freq='D')
tangent_jobs = []
for pov_datetime in pov_datetimes:
    tangent_jobs.append({'id':str(uuid.uuid4()),'parameters':{'pov_datetime':str(pov_datetime)}})

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoForecast

# COMMAND ----------

# MAGIC %md
# MAGIC Below shows an example user defined function that will be processed by Spark and Tangent.

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
# MAGIC Here the simulation is executed and 92 jobs will be processed simultaneously. The spark_jobs object is created and it combines the general configuration with datasets that are based on the original dataframe, but are processed together with the alignment_records to create a representative dataset from the point of view of a date in the past. Depending on the underlying resources of the Databricks cluster this can take a bit of time, this process can be accelerated if required.

# COMMAND ----------

tw_spark = tw.SparkParallelProcessing()
spark_jobs = []
for tangent_job in tangent_jobs:
    tangent_job_parameters = tangent_job['parameters']
    pov_datetime = tangent_job_parameters['pov_datetime']
    data = simulation.pov_dataset(pov_datetime,tangent_dataframe,timestamp_column,target_column,predictors,alignment_records)
    spark_jobs.append((tangent_job['id'],user_defined.run_auto_forecast,{'data':data,'configuration':auto_forecasting_configuration}))
tw_parallel_auto_forecasting = tw_spark.run(jobs=spark_jobs)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Results

# COMMAND ----------

# MAGIC %md
# MAGIC Once the Spark jobs are finished, all results can be processed and made ready for analysis. We extract the production forecasts, properties and features to gain insights into how the models would have evolved over time.

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

simulation_results_df = tangent_result_tables_df[['pov_datetime','timestamp','forecast','type']].dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC In the graph below, all simulated forecasts are brought together in 1 time series and these can be compared to historical target values.  
# MAGIC You can see how closely the predicted values match with the target meaning we have simulated an accurate production process and we can likely place trust in future forecasts.  
# MAGIC Note that all these forecasts can be easily joined as they each cover 1 day in the testing period. Keep in mind that in other simulations it is certainly possible, forecasts from different point of view dates can overlap if the forecasting horizon is longer than the rolling window of execution.
# MAGIC The user will then need to manage the visualization of results. In this example, the jobs are set up so that a single time series of results can easily be created as show below.

# COMMAND ----------

fig = splt.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=tangent_dataframe[timestamp_column], y=tangent_dataframe[target_column], name='target',line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=simulation_results_df['timestamp'], y=simulation_results_df['forecast'], name='simulation',line=dict(color='goldenrod')), row=1, col=1)
fig.update_layout(height=500, width=1000, title_text="Results")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Since the models are continuously rebuild over the testing period, we can also show the properties over time and see how the models evolve.  
# MAGIC With this we visualize the adaptation of models to changing circumstances. It is a live image of how Tangent would have reacted to changing patterns in the data.  
# MAGIC This can be very useful to find structural changes over time. Luckily Tangent automatically adapts to the changing environment and the risk of model drift is completely managed.

# COMMAND ----------

fig = px.bar(tangent_properties_df[tangent_properties_df['importance']>0], x='pov_datetime', y="rel_importance", color="name", barmode = 'stack',hover_data=group_keys)
fig.update_layout(height=500, width=1200, title_text="Evolution")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Comparison

# COMMAND ----------

# MAGIC %md
# MAGIC The simulation is now complete and the user now should have a good image on the performance of Tangent.  
# MAGIC In this section, we highlight the importance of continuous model rebuilding. Let's create a simple backtest and compare the results of this one off model build and prediction with simulated results from above.

# COMMAND ----------

# MAGIC %md
# MAGIC We make sure the testing period of the backtest corresponds with the simulation period.

# COMMAND ----------

backtest_configuration = {
    'preprocessing': {
        'training_rows': [{'from': '2021-01-20 00:00:00','to': '2021-12-17 23:00:00'}],
        'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
    },
    'engine': {
        'prediction_from': {'base_unit': 'day','value': 1},
        'prediction_to': {'base_unit': 'day','value': 1},
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can run the backtest and inspect the forecast.

# COMMAND ----------

backtest = user_defined.run_auto_forecast(data=tangent_dataframe,configuration=backtest_configuration)
backtest_result_table_df = tw.PostProcessing().result_table(forecasting=backtest)

# COMMAND ----------

# MAGIC %md
# MAGIC At first glance, the difference between the red testing values of the backtest and the gold forecast values from the simulation seem comparable. 

# COMMAND ----------

visualization.predictions(backtest_result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC However, when we bring together both results, place them next to each other and calculate accuracy metrics, a different conclusion emerges.

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
accuracy_comparison_df = comparison_df[(comparison_df['target'].notnull())&(comparison_df['simulation'].notnull())&(comparison_df['backtest'].notnull())].copy()
accuracy_comparison_df['simulation_MAE'] = abs(accuracy_comparison_df['simulation'] - accuracy_comparison_df['target'])
accuracy_comparison_df['backtest_MAE'] = abs(accuracy_comparison_df['backtest'] - accuracy_comparison_df['target'])

# COMMAND ----------

# MAGIC %md
# MAGIC In the graph below, we visualize the target, the predicted values from each approach and accuracy metrics to compare.  
# MAGIC In the top section, the backtest and simulation predicted values seem to be quite similar and it is not yet clear which approach was more accurate.  
# MAGIC
# MAGIC When we visualize the absolute errors (MAE) in the middle section, we start to see that the golden simulations values seem to be consistently lower meaning less error in the forecast on each given timestamp.
# MAGIC
# MAGIC In the bottom section, the MAE is cumulatively added together to show the performance of each approach over time. The lower this line, the better the forecast.  
# MAGIC Here we learn that by the end of the testing period, a structural change occured which significantly impacted the performance of the backtest.  
# MAGIC The simulation seems to be able to quickly adapt to this change and continue creating highly accurate forecasts whereas in the backtest no rebuilding took place so this new information could not be taken into account.
# MAGIC This shows the power and relevance of continuous model rebuilding and how it helps the user with remaining adaptable to structural changes.

# COMMAND ----------

visualization.comparison(df=accuracy_comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Also when we calculate the MAE over this period we can clearly see that the simulated results are lower and outperform the backtest.

# COMMAND ----------

pd.DataFrame(accuracy_comparison_df[['simulation_MAE','backtest_MAE']].mean(),columns=['MAE'])
