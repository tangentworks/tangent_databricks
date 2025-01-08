# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Example - Wind Production Forecasting

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. Setup

# COMMAND ----------

import tangent_works as tw
import pandas as pd
import numpy as np

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
# MAGIC # 1. Data

# COMMAND ----------

file_path = '/Workspace'+dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 2)[0]+'/data/wind_single_asset.csv'
tangent_dataframe = pd.read_csv(file_path)
group_keys = []
timestamp_column = "timestamp"
target_column = "Energy"
predictors = [s for s in list(tangent_dataframe.columns) if s not in group_keys + [timestamp_column, target_column]]
tangent_dataframe = tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors].sort_values(by=group_keys + [timestamp_column]).reset_index(drop=True)
tangent_dataframe[timestamp_column] = pd.to_datetime(pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S"))
tangent_dataframe

# COMMAND ----------

visualization.data(df=tangent_dataframe,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Configuration

# COMMAND ----------

training_from = str(tangent_dataframe[timestamp_column].min())
training_to = str(tangent_dataframe[timestamp_column].quantile(2/3))
prediction_to = str(tangent_dataframe[timestamp_column].max())

# COMMAND ----------

auto_forecasting_configuration = {
    'preprocessing': {
        'training_rows': [{'from': training_from,'to': training_to}],
        'prediction_rows': [{'from': training_from,'to': prediction_to}],
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
        'prediction_to': {'base_unit': 'sample','value': 2},
        # 'target_offsets': 'none',
        # 'predictor_offsets': 'common',
        # 'allow_offsets': True,
        # 'max_offsets_depth': 120,
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
# MAGIC # 3. Tangent

# COMMAND ----------

time_series = tw.TimeSeries(tangent_dataframe, timestamp_column)
time_series.validate()
tangent_auto_forecast = tw.AutoForecasting(time_series=time_series, configuration=auto_forecasting_configuration)
tangent_auto_forecast.run()
tangent_auto_forecast_model = tangent_auto_forecast.model.to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Results

# COMMAND ----------

properties_df = tw.PostProcessing().properties(model=tangent_auto_forecast_model)
features_df = tw.PostProcessing().features(model=tangent_auto_forecast_model)
result_table_df = tw.PostProcessing().result_table(forecasting=tangent_auto_forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Visualization

# COMMAND ----------

visualization.predictions(result_table_df)

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

visualization.feature_importance(features_df)
