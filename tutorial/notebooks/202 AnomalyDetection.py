# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - AnomalyDetection

# COMMAND ----------

# MAGIC %md
# MAGIC #0. Setup

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
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[forecasting_type], name=forecasting_type,line=dict(color=color_map[forecasting_type])), row=1, col=1)
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
# MAGIC #1. Data Processing

# COMMAND ----------

file_path = '/Workspace'+dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit('/', 2)[0]+'/data/gearbox.csv'
tangent_dataframe = pd.read_csv(file_path)
group_keys = []
timestamp_column = "timestamp"
target_column = "GEARBEARINGTEMP"
predictors = [s for s in list(tangent_dataframe.columns) if s not in group_keys + [timestamp_column, target_column]]
tangent_dataframe = tangent_dataframe[group_keys + [timestamp_column, target_column] + predictors].sort_values(by=group_keys + [timestamp_column]).reset_index(drop=True)
tangent_dataframe[timestamp_column] = pd.to_datetime(pd.to_datetime(tangent_dataframe[timestamp_column]).dt.strftime("%Y-%m-%d %H:%M:%S"))
tangent_dataframe

# COMMAND ----------

visualization.data(df=tangent_dataframe,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Configuration

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
        {
            'residuals_transformation':{
                'type':'residuals_change',
    #             'window_length':2
            },
    #         'sensitivity':0.3
        },
        {
            'residuals_transformation':{
                'type':'moving_average',
    #             'window_length':1
            },
    #         'sensitivity':0.3
        },
        {
            'residuals_transformation':{
                'type':'moving_average_change',
    #             'window_lengths':[
    #                 2,
    #                 1
    #             ]
            },
    #         'sensitivity':0.3
        },
        {
            'residuals_transformation':{
                'type':'standard_deviation',
    #             'window_length':1
            },
    #         'sensitivity':0.3
        },
        {
            'residuals_transformation':{
                'type':'standard_deviation_change',
    #             'window_lengths':[
    #                 2,
    #                 1
    #             ]
            },
    #         'sensitivity':0.3
        },
    ]
}

# COMMAND ----------

# MAGIC %md
# MAGIC #3. Tangent

# COMMAND ----------

time_series = tw.TimeSeries(data=tangent_dataframe)
tangent_anomaly_detection = tw.AnomalyDetection(time_series=time_series,configuration=build_anomaly_detection_configuration)
tangent_anomaly_detection.build_model()
tangent_anomaly_detection_model = tangent_anomaly_detection.model.to_dict()

# COMMAND ----------

detect_df = tangent_anomaly_detection.detect()

# COMMAND ----------

# MAGIC %md
# MAGIC #4. Results Processing

# COMMAND ----------

properties_df = tw.PostProcessing().properties(response=tangent_anomaly_detection_model)
features_df = tw.PostProcessing().features(response=tangent_anomaly_detection_model)

# COMMAND ----------

# MAGIC %md
# MAGIC #5. Visualization

# COMMAND ----------

visualization.detections(detect_df)

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

visualization.feature_importance(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #6. Root Cause Analysis

# COMMAND ----------

tangent_rca = tangent_anomaly_detection.rca()

# COMMAND ----------

rca_tables = []
for model_index in sorted(tangent_rca.keys()):
    rca_table = tangent_rca[model_index].melt(id_vars='timestamp')
    rca_table['model_index'] = model_index
    rca_tables.append(rca_table)
rca_tables_df = pd.concat(rca_tables)
rca_tables_df['type'] = np.where(rca_tables_df['variable'].str.contains('term '),'term',np.where(rca_tables_df['variable'].str.contains('yhat '),'yhat','other'))
rca_tables_df['term'] = np.where(rca_tables_df['type'].isin(['term','yhat']),rca_tables_df['variable'].str.replace('term ','').str.replace('yhat ',''),np.nan)

# COMMAND ----------

rca_timestamp = '2020-10-11 00:00:00'
window = 20
visualization.rca(
    time_series=tangent_dataframe,
    timestamp_column=timestamp_column,
    target_column=target_column,
    df=detect_df,
    rca_tables_df=rca_tables_df,
    rca_timestamp=rca_timestamp,
    window=window
    )
