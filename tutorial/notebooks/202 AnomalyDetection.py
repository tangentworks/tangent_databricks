# Databricks notebook source
# MAGIC %md
# MAGIC # Tangent Databricks Tutorial - AnomalyDetection

# COMMAND ----------

# MAGIC %md
# MAGIC In this tutorial, you will learn to build an AnomalyDetection model with Tangent, generate detections using this model and use additional capabilities.  
# MAGIC The AnomalyDetection module exists to have controle over all the steps in the model building & detection process. The main steps are:
# MAGIC 1. building a model using historical training data.
# MAGIC 2. making an inference using this model.
# MAGIC
# MAGIC To show the capabilities of the AnomalyDetection module, we will use an example dataset from an industrial use case.  
# MAGIC The goal is to monitor the state of a gearbox system and prevent damage by detecting anomalies using historical sensor measurements.

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
# MAGIC #1. Data

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset that will be used in this notebook is called "gearbox".  
# MAGIC It contains historical sensor data of a gearbox system such as temperature measurements from different parts of the equipment and power applied to the system.  
# MAGIC In the cell below, this dataset is preprocessed and made ready for use with Tangent.

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

# MAGIC %md
# MAGIC In time series analysis, when exploring a dataset, it is best practice to visualize the data and learn which patterns might exists in the data that we want Tangent to identify automatically.  
# MAGIC In this graph, the target column "GEARBEARINGTEMP" is visualized above and the additional explanatory variables or predictors are visualized below.  
# MAGIC Notice that all measurements are aligned with the target column since they were updated and available at the same time.

# COMMAND ----------

visualization.data(df=tangent_dataframe,timestamp=timestamp_column,target=target_column,predictors=predictors)

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC The first step in the anomaly detection process using the Tangent is model building. To describe to Tangent, how it should build a model using this dataset, we can use the configuration below.  
# MAGIC Many settings can be applied, however Tangent is designed to automate as much as possible. When a parameter is not set, Tangent will assume default settings.  
# MAGIC In that case, Tangent will decided how to apply certain settings for you. You can find the final result in the AnomalyDetection object after model building.  
# MAGIC
# MAGIC In this example, default settings will be used. The only specific configuration will be to showcase the different kind of detection layers. By default, the residual and moving average are applied but there are other types of detection layers available that we will use in the example below.  
# MAGIC Tangent will automatically recognize the most likely sampling rate, in this case quarter hourly, and build a time series anomaly detection model automatically. 

# COMMAND ----------

build_anomaly_detection_configuration = {
    'normal_behavior':{
        # 'target_column':'str',
        # 'holiday_column:':'str',
        # 'target_offsets':'combined',
        # 'allow_offsets':True,
        # 'max_offsets_depth': 0,
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

# MAGIC %md
# MAGIC In this section, the following steps take place:
# MAGIC 1. Create and validate a Tangent time series object
# MAGIC 2. Create an AnomalyDetection object by combining a time series and model building configuration.
# MAGIC 3. Send a model building request by applying the "build_model" function.
# MAGIC 4. Send a detect request by applying the "detect" function.

# COMMAND ----------

time_series = tw.TimeSeries(data=tangent_dataframe)
time_series.validate()

# COMMAND ----------


tangent_anomaly_detection = tw.AnomalyDetection(time_series=time_series,configuration=build_anomaly_detection_configuration)


# COMMAND ----------

tangent_anomaly_detection.build_model()
tangent_anomaly_detection_model = tangent_anomaly_detection.model.to_dict()

# COMMAND ----------

detect_df = tangent_anomaly_detection.detect()

# COMMAND ----------

# MAGIC %md
# MAGIC #4. Results

# COMMAND ----------

# MAGIC %md
# MAGIC The model can now post processed into tables that can either be stored, analyzed or visualized by the user.  
# MAGIC Below, the properties and features of the model are extracted.

# COMMAND ----------

properties_df = tw.PostProcessing().properties(model=tangent_anomaly_detection_model)
features_df = tw.PostProcessing().features(model=tangent_anomaly_detection_model)

# COMMAND ----------

# MAGIC %md
# MAGIC #5. Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC All results can be easily visualized using the provided templates.  
# MAGIC The top graph shows:
# MAGIC - The __"target"__ or historical sensor measurements.
# MAGIC - The __"normal_behavior"__ which are in sample results that lie on top of the training data.  
# MAGIC These values show the expected behavior of the target signal according to Tangent. 
# MAGIC - __"Anomalies"__ from the detection layers indicating points of interest in the target.
# MAGIC
# MAGIC The bottom graph shows several detection layers that are based on the differences between the target signal and the normal behavior. These layers offer different perspectives, and different ways to identify anomalies. 
# MAGIC
# MAGIC When exploring the graph, we can recognize that an effective normal behavior model seems to have been identified by Tangent. The rising detection layers at the end clearly identify increasingly anomalous values near the end of the dataset that, in this case, can be leveraged to detect faults in the gearbox before a breakdown event.

# COMMAND ----------

visualization.detections(detect_df)

# COMMAND ----------

# MAGIC %md
# MAGIC In order to understand which patterns Tangent has identified to achieve this normal behavior, we can visualize several levels of insights.  
# MAGIC
# MAGIC Firstly, the properties, which show the relative importance of each of the columns of the dataset.  
# MAGIC Here, we learn which columns contributed a lot of predictive value to the model and where a lot of useful features have been found.  
# MAGIC If there are predictors from which no features were included in the model building by Tangent, then they will be listed here as well.

# COMMAND ----------

visualization.predictor_importance(properties_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Secondly, the actual features from the Tangent model can be visualized as well.  
# MAGIC In this treemap graph, the relative importance of each of the features from each model in the model zoo can be quickly identified. In this example, only 1 model exists in the model zoo.
# MAGIC These are the predictive patterns that are hidden within the data that Tangent has automatically extracted from the data.  

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

# MAGIC %md
# MAGIC Move the slider from left to right to add the different features into the model to eventually come to the final predicted value.  
# MAGIC The black line are the original measured values. The orange line shows the RCA values that dynamically moves as features are added.  
# MAGIC The red line shows the eventual prediction that will correspond with the orange line when the slider is moved entirely to the right. 

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
