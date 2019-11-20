import boto3
import json
import random
import datetime

from boto3.dynamodb.conditions import Attr, Key

N = 3 # number of video files to be returned in tables of top N

def fetch_ruid(table):
    playable_uid_list = table.scan(ProjectionExpression='id')
    # FUTURE: add filter for Attr approved videos if users will upload
    x = random.choice(list(playable_uid_list['Items']))
    return int(x['id'])

def fetch_file(vuid,table):
    response = table.get_item(Key={'id': str(vuid)}) 
    return(response['Item']['clip_title'])

def fetch_prediction(vuid,table):
    response = table.get_item(Key={'id': str(vuid)})
    return(response['Item']['pred_class'])

def fetch_model_proba(vuid,table):
    response = table.get_item(Key={'id': str(vuid)})
    return(response['Item']['pred_class_proba'])
    
def log_playback(vuid,time,table):
    response = table.get_item(Key={'id': str(vuid)})
    table.update_item(Key={'id': str(vuid)},\
        UpdateExpression='SET recent_play = :val1',\
        ExpressionAttributeValues={':val1': time})

def fetch_recent_N(client,index,table):
    response = client.scan(
        TableName=table.name,
        IndexName=index,
        # ScanIndexForward=True,
        Limit=N)
    result = [clip['clip_title']['S'] for clip in response['Items']]
    return(result)


def lambda_handler(event, context):
    result = None
    now = datetime.datetime.timestamp(datetime.datetime.utcnow())
    dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
    client = boto3.client('dynamodb')
    table = dynamodb.Table('videos')
    index = 'recent_play_index'
    if 'ruid' in event.get('action',''):
        result = fetch_ruid(table)
    elif 'file' in event.get('action',''):
        vuid = event.get('id',0)
        result = fetch_file(vuid,table)
    elif 'prediction' in event.get('action',''):
        vuid = event.get('id',0)
        result = fetch_prediction(vuid,table)
    elif 'probability' in event.get('action',''):
        vuid = event.get('id',0)
        result = fetch_model_proba(vuid,table)
    elif 'playback' in event.get('action',''):
        vuid = event.get('id',0)
        log_playback(vuid,int(now),table)
        result = "logged"
    elif 'recent' in event.get('action',''):
        result = fetch_recent_N(client,index,table)
    return {
        'statusCode': 200,
        'body': result
    }

