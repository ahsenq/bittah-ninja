import boto3
import datetime
from decimal import Decimal
import json
import random

from boto3.dynamodb.conditions import Attr, Key

N = 3 # number of video files to be returned in tables of top N

def fetch_ruid(table):
    # Returns a random video ID
    playable_uid_list = table.scan(ProjectionExpression='id')
    # FUTURE: add filter for Attr approved videos if users will upload
    x = random.choice(list(playable_uid_list['Items']))
    return int(x['id'])

def fetch_file(vuid,table):
    # Returns model's video filename, given its unique ID
    response = table.get_item(Key={'id': str(vuid)}) 
    return(response['Item']['clip_title'])

def fetch_prediction(vuid,table):
    # Returns model's predicted class for this clip
    response = table.get_item(Key={'id': str(vuid)})
    return(response['Item']['pred_class'])

def fetch_model_proba(vuid,table):
    # Returns probability of predicted class from model
    response = table.get_item(Key={'id': str(vuid)})
    return(response['Item']['pred_class_proba'])
    
def log_playback(vuid,time,table):
    # Sets the timestamp of most recent video play
    response = table.get_item(Key={'id': str(vuid)})
    table.update_item(Key={'id': str(vuid)},\
        UpdateExpression='SET recent_play = :val1',\
        ExpressionAttributeValues={':val1': time})

def fetch_recent_N(client,index,table):
    # Returns a list of N clips played recently (most for rewatching purposes)
    response = client.scan(
        TableName=table.name,
        IndexName=index,
        # ScanIndexForward=True,
        Limit=N)
    result = [clip['clip_title']['S'] for clip in response['Items']]
    return(result)

def update_portions(vuid,count_nope,count_punch,table):
    # Used by vote incrementing functions
    new_pct_punch = Decimal(str(count_punch/(count_nope + count_punch)))
    table.update_item(Key={'id': str(vuid)},\
        UpdateExpression='SET portion_humans_saw_punch = :punch_vote',\
        ExpressionAttributeValues={':punch_vote': new_pct_punch})
    new_pct_none = 1 - new_pct_punch
    table.update_item(Key={'id': str(vuid)},\
        UpdateExpression='SET portion_humans_saw_none = :none_vote',\
        ExpressionAttributeValues={':none_vote': new_pct_none})

def increment_punch_vote(vuid,table):
    # Logs when human says clip did contain a punch
    # Updates the tally, the vote percentage, and the human consensus as needed
    record = table.get_item(Key={'id': str(vuid)})
    
    # update humans who voted punch
    old_punch_tally = int(record['Item']['num_humans_saw_punch'])
    new_punch_tally = old_punch_tally + 1
    table.update_item(Key={'id': str(vuid)},\
        UpdateExpression='SET num_humans_saw_punch = :tally',\
        ExpressionAttributeValues={':tally': new_punch_tally})
    
    # update human consensus
    no_punch_tally = int(record['Item']['num_humans_saw_none'])
    new_consensus = 0
    if new_punch_tally > no_punch_tally:
        new_consensus = 1
    elif new_punch_tally == no_punch_tally:
        new_consensus = None
    table.update_item(Key={'id': str(vuid)},\
        UpdateExpression='SET human_consensus = :consensus',\
        ExpressionAttributeValues={':consensus': new_consensus})
        
    # update voting percentages
    update_portions(vuid, no_punch_tally,new_punch_tally,table)
    
    return(table.get_item(Key={'id': str(vuid)}))
    
def increment_none_vote(vuid,table):
    # Logs when human says clip did not contain a punch
    # Updates the tally, the vote percentage, and the human consensus as needed
    record = table.get_item(Key={'id': str(vuid)})
    
    # update humans who voted no-punch
    old_none_tally = int(record['Item']['num_humans_saw_none'])
    new_none_tally = old_none_tally + 1
    table.update_item(Key={'id': str(vuid)},\
        UpdateExpression='SET num_humans_saw_none = :tally',\
        ExpressionAttributeValues={':tally': new_none_tally})
    
    # update human consensus
    punch_tally = int(record['Item']['num_humans_saw_punch'])
    new_consensus = 1
    if new_none_tally > punch_tally:
        new_consensus = 0
    elif new_none_tally == punch_tally:
        new_consensus = None
    table.update_item(Key={'id': str(vuid)},\
        UpdateExpression='SET human_consensus = :consensus',\
        ExpressionAttributeValues={':consensus': new_consensus})
            
    # update voting percentages
    update_portions(vuid, new_none_tally,punch_tally,table)
    
    return(table.get_item(Key={'id': str(vuid)}))
    
def fetch_ambiguous(table):
    # Returns list of clips where between 40-60% of humans saw a punch
    distance = Decimal('0.1')
    fe = Attr('portion_humans_saw_punch').between(Decimal('0.5')-distance,\
        Decimal('0.5')+distance)
    pe = "clip_title,portion_humans_saw_punch"
    response = table.scan(ProjectionExpression=pe,FilterExpression=fe)
    items = response['Items']
    return [{i["clip_title"]:i["portion_humans_saw_punch"]} for i in items]

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
        result = "logged playback"
    elif 'recent' in event.get('action',''):
        result = fetch_recent_N(client,index,table)
    elif 'vote_punch' in event.get('action',''):
        vuid = event.get('id',0)
        increment_punch_vote(vuid,table)
        result = "logged punch vote"
    elif 'vote_none' in event.get('action',''):
        vuid = event.get('id',0)
        increment_none_vote(vuid,table)
        result = "logged none vote"
    elif 'ambiguous' in event.get('action',''):
        result = fetch_ambiguous(table)
    return {
        'statusCode': 200,
        'body': result
    }
