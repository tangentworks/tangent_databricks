# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - AutoForecasting

# COMMAND ----------

# MAGIC %md
# MAGIC In this tutorial, you will learn to build a forecasting model and generate predictions with Tangent in 1 step using AutoForecasting as well as use additional capabilities.  
# MAGIC The AutoForecasting module builds on the Forecasting capabilities and combines the steps of model building and prediction into one. 
# MAGIC This helps the user with quickly applying Tangent into a forecasting use case.
# MAGIC
# MAGIC To show the capabilities of the AutoForecasting module, we will use an example dataset from a energy forecasting use case.  
# MAGIC The goal is to forecast the grid load (energy consumption) in Belgium 24 hours ahead using historical consumption data and other explanatory variables such as weather and public holidays.

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

# COMMAND ----------

# MAGIC %md
# MAGIC To visualize the results of this exercise, the following visualization functions can be used.

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
        fig1.show()

    def feature_importance(df):
        fig = px.treemap(df, path=[px.Constant("all"), 'model', 'feature'], values='importance',hover_data='beta',color='feature')
        fig.update_traces(root_color="lightgrey")
        fig.update_layout(height=600, width=1000, title_text="Features",margin = dict(t=50, l=25, r=25, b=25))
        fig.show()

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
        fig.show()

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
# MAGIC In time series analysis, when exploring a dataset, it is best practice to visualize the data and learn which patterns might exists in the data that we want Tangent to identify automatically.  
# MAGIC In this graph, the target column "Quantity" is visualized above and the additional explanatory variables or predictors are visualized below.  
# MAGIC Notice that for some predictors, values are available ahead of the last target timestamp throughout the forecast horizon. 

# COMMAND ----------

visualization.data(df=tangent_dataframe,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC In the autoforecasting process we process the dataset to build the model and generate predictions in 1 step. This means the configuration needs to be able to preprocess the dataset and configure the engine for model building and prediction. Therefore the configuration consists of two parts: "preprocessing" and "engine". The configuration below outlines all the possible settings that ccan be applied in AutoForecasting.  
# MAGIC
# MAGIC Tangent is designed to automate as much as possible. When a parameter is not set, Tangent will assume default settings.  
# MAGIC In that case, Tangent will decided how to apply certain settings for you. You can find the final result in the AutoForecasting object after running the job.
# MAGIC
# MAGIC In this example, default settings will be used. The only configuration will be the forecasting horizon and the training and prediction rows. The forecasting horizon is here set from sample+1 until sample+24. 
# MAGIC
# MAGIC The training rows select the rows in the dataset that will be used for building a model. Here a subset from the first 2/3 of the dataset is selected for training.   
# MAGIC The prediction rows select the rows for which an inference is made. This generates both in sample training, out of sample testing and production results. Here the entire dataset is selected.
# MAGIC
# MAGIC Tangent will automatically recognize the most likely sampling rate, in this case hourly, and build a time series forecasting model to predict values for the next 24 hours. 

# COMMAND ----------

auto_forecasting_configuration = {
    'preprocessing': {
        'training_rows': [{'from': '2021-01-20 00:00:00','to': '2021-10-31 23:00:00'}],
        'prediction_rows': [{'from': '2021-01-20 00:00:00','to': '2022-01-18 23:00:00'}],
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
    },
    'engine': {
        # 'target_column': target_column,
        # 'holiday_column': 'string',
        'prediction_from': {'base_unit': 'sample','value': 1},
        'prediction_to': {'base_unit': 'sample','value': 24},
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
# MAGIC In this section, the following steps take place:
# MAGIC 1. Create and validate a Tangent time series object
# MAGIC 2. Create an AutoForecasting object by combining a time series and auto forecasting configuration.
# MAGIC 3. Send a job request by applying the "run" function. Now the model and predictions are generated.  

# COMMAND ----------

time_series = tw.TimeSeries(tangent_dataframe, timestamp_column)
time_series.validate()

# COMMAND ----------

tangent_auto_forecast = tw.AutoForecasting(time_series=time_series, configuration=auto_forecasting_configuration)

# COMMAND ----------


tangent_auto_forecast.run()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's extract the model from the AutoForecasting object to get insights in the model building process.

# COMMAND ----------

tangent_auto_forecast_model = tangent_auto_forecast.model.to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC #4. Results

# COMMAND ----------

# MAGIC %md
# MAGIC The model and the AutoForecasting object can now post processed into tables that can either be stored, analyzed or visualized by the user.  
# MAGIC Below, the properties and features of the model are extracted. In addition, information about the type of forecasts is added to table with predictions.

# COMMAND ----------

properties_df = tw.PostProcessing().properties(model=tangent_auto_forecast_model)
features_df = tw.PostProcessing().features(model=tangent_auto_forecast_model)
result_table_df = tw.PostProcessing().result_table(forecasting=tangent_auto_forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC #5. Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC All results can be easily visualized using the provided templates.  
# MAGIC The graph below shows:
# MAGIC - The __"target"__ or historical consumption values.
# MAGIC - The __"training"__ forecast which are the in sample results that lie on top of the training data.  
# MAGIC These can be used to learn if Tangent has created a model that fits the original training data well.
# MAGIC They show several 24 hour ahead forecasts throughout the training period.
# MAGIC - The __"testing"__ forecast which are the out of sample results that lie on top of the data that were not seen by Tangent during model building.  
# MAGIC These can be used to learn if Tangent has created a model that also works well outside of the original training data.
# MAGIC They show several 24 hour ahead forecasts throughout the testing period.
# MAGIC - the __"production"__ forecast which contains the out of sample predictions and are the objective of this exercise.  
# MAGIC In this example, we can recognize a 24 hour ahead forecast relative from the last available target value.  
# MAGIC
# MAGIC When exploring the graph, we can recognize that a useful pattern seems to have been identified by Tangent which allows us to accurately forecast 24 hours into the future.

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

# MAGIC %md
# MAGIC In order to understand which patterns Tangent has identified to achieve this forecast, we can visualize several levels of insights.  
# MAGIC
# MAGIC Firstly, the properties, which show the relative importance of each of the columns of the dataset.  
# MAGIC Here, we learn which columns contributed a lot of predictive value to the model and where a lot of useful features have been found.  
# MAGIC If there are predictors from which no features were included in the model building by Tangent, then they will be listed here as well.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Secondly, the actual features from the Tangent model can be visualized as well.  
# MAGIC In this treemap graph, the relative importance of each of the features from each model in the model zoo can be quickly identified.  
# MAGIC These are the predictive patterns that are hidden within the data that Tangent has automatically extracted from the data.  
# MAGIC
# MAGIC Tangent builds multihorizon time series models, meaning for each step in the forecasting horizon, Tangent will by default build a unique model. These models are then combined in a model zoo which is the result that the user receices.  
# MAGIC With this capability, Tangent remains adaptable in modeling different patterns that can be usefull to seperate along the forecasting horizon.  
# MAGIC Also, the further out in the horizon, the more uncertainty exists, and this can be easily managed by creating different models in a model zoo. 

# COMMAND ----------

visualization.feature_importance(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #6. Root Cause Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC An extended capability of Tangent is to apply Root Cause Analysis (RCA) to the predictions.  
# MAGIC With RCA, we can combine the information from both the features in the model as well as the predictions and apply this to a single timestamp in the dataset.  
# MAGIC
# MAGIC A Tangent model is a cumulative addition of identified features that each explain a bit more of the variance in the target signal.  
# MAGIC We can visualize the addition of these features to the model and learn which features contribute to either useful movements or possibly unexpected movements in the prediction.
# MAGIC
# MAGIC The example below first extracts from Tangent how each and every single prediction is built up. This information is processed and from there, the user can select a specific timestamp to analyze and choose a window around that timestamp for additional context.

# COMMAND ----------

tangent_forecast_rca = tangent_auto_forecast.rca()

# COMMAND ----------

rca_tables = []
for model_index in sorted(tangent_forecast_rca.keys()):
    rca_table = tangent_forecast_rca[model_index].melt(id_vars='timestamp')
    rca_table['model_index'] = model_index
    rca_tables.append(rca_table)
rca_tables_df = pd.concat(rca_tables)
rca_tables_df['type'] = np.where(rca_tables_df['variable'].str.contains('term '),'term',np.where(rca_tables_df['variable'].str.contains('yhat '),'yhat','other'))
rca_tables_df['term'] = np.where(rca_tables_df['type'].isin(['term','yhat']),rca_tables_df['variable'].str.replace('term ','').str.replace('yhat ',''),np.nan)

# COMMAND ----------

# MAGIC %md
# MAGIC Move the slider from left to right to add the different features into the model to eventually come to the final predicted value.  
# MAGIC The black line are the original measured values. The orange line shows the RCA values that dynamically moves as features are added.  
# MAGIC The red line shows the eventual prediction that will correspond with the orange line when the slider is moved entirely to the right. 

# COMMAND ----------

rca_timestamp = '2021-08-20 17:00:00'
window = 48
visualization.rca(
    time_series=tangent_dataframe,
    timestamp_column=timestamp_column,
    target_column=target_column,
    df=result_table_df,
    rca_tables_df=rca_tables_df,
    rca_timestamp=rca_timestamp,
    window=window
    )
