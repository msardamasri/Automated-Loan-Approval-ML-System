import json
import boto3
import urllib.request
from datetime import datetime

#s3 client
s3 = boto3.client('s3')

BUCKET_NAME = '...'#config bucket name

def lambda_handler(event, context):
    print("=== EXTRACTION ===")
    print(f"Timestamp ejecución: {datetime.now().isoformat()}")
    
    try:
        # THIS OPTION IS FOR API DATA EXTRACTION
        url = '...'#fill it up with the api
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            api_data = response.read().decode('utf-8')
            api_json = json.loads(api_data)
        print(f"Status: {response.status}")
        
        #file name
        now = datetime.now()
        date_str = now.strftime('%Y%m%d')
        timestamp_str = now.strftime('%Y%m%d_%H%M%S')
        #file folder with time
        file_key = f"raw/{date_str}/data_{timestamp_str}.json"

        #save in s3
        print(f"Saving in S3: s3://{BUCKET_NAME}/{file_key}")
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=file_key,
            Body=json.dumps(api_json, indent=2, ensure_ascii=False),
            ContentType='application/json',
            Metadata={
                'extraction_date': now.isoformat()
            }
        )
        
        print(f"file stored")
        return {
            'statusCode': 200
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise