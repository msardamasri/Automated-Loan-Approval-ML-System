import json
import boto3
from datetime import datetime
from collections import defaultdict

#s3 client
s3 = boto3.client('s3')

BUCKET_NAME = '...'#config bucket name

def lambda_handler(event, context):
    print("=== PROCESING ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        today = datetime.now().strftime('%Y%m%d')
        prefix = f'raw/{today}/'
        #list files in raw folder
        response = s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix
        )
        #no content
        if 'Contents' not in response:
            print("There are no files to process.")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'No data to process'})
            }
        
        #get latest file
        files = [obj for obj in response['Contents'] if obj['Key'].endswith('.json')]
        files_sorted = sorted(files, key=lambda x: x['LastModified'], reverse=True)
        latest_file = files_sorted[0]

        key = latest_file['Key']
        print(f"Processing ONLY latest file: {key}")

        #read file
        file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        file_content = file_obj['Body'].read().decode('utf-8')
        data = json.loads(file_content)
        
        processed_data = process_loan_applications(data)
        
        output_key = f'processed/{today}/summary_{today}.json'
        print(f"Saving to: s3://{BUCKET_NAME}/{output_key}")
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=output_key,
            Body=json.dumps(processed_data, indent=2, ensure_ascii=False),
            ContentType='application/json'
        )
        
        print("Processing completed")
        
        #lambda function ml-predictor invocation
        lambda_client = boto3.client('lambda')
        invoke_response = lambda_client.invoke(
            FunctionName='lambda-ml-predicator', #change name to your lambda func
            InvocationType='Event',
            Payload=json.dumps({})
        )
        print(f"ML Predictor invoked. StatusCode: {invoke_response['StatusCode']}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing completed',
                'ml_predictor_invoked': True
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def process_loan_applications(applications):
    """
    Validate and clean loan application data
    """
    processed_records = []
    
    def safe_float(value):
        try:
            return float(value) if value not in [None, '', 'null'] else 0.0
        except:
            return 0.0
    
    def safe_int(value):
        try:
            return int(value) if value not in [None, '', 'null'] else 0
        except:
            return 0
    
    for app in applications:
        try:
            #match your notebook feature names
            processed_record = {
                'person_age': safe_int(app.get('person_age', 0)),
                'person_income': safe_float(app.get('person_income', 0.0)),
                'person_emp_exp': safe_int(app.get('person_emp_exp', 0)),
                'loan_amnt': safe_float(app.get('loan_amnt', 0.0)),
                'loan_int_rate': safe_float(app.get('loan_int_rate', 0.0)),
                'loan_percent_income': safe_float(app.get('loan_percent_income', 0.0)),
                'cb_person_cred_hist_length': safe_int(app.get('cb_person_cred_hist_length', 0)),
                'person_gender': app.get('person_gender', 'Unknown'),
                'employment_type': app.get('employment_type', 'Unknown'),
                'person_home_ownership': app.get('person_home_ownership', 'Unknown'),
                'loan_intent': app.get('loan_intent', 'Unknown'),
                'account_type': app.get('account_type', 'Unknown'),
                'person_education': app.get('person_education', 'High School'),
                'previous_loan_defaults_on_file': app.get('previous_loan_defaults_on_file', 'No'),
                #add application metadata
                'application_id': app.get('application_id', ''),
                'timestamp': app.get('timestamp', datetime.now().isoformat())
            }
            processed_records.append(processed_record)
        except Exception as e:
            print(f"Error processing application: {e}")
            continue
    
    return {
        'processed_count': len(processed_records),
        'records': processed_records,
        'timestamp': datetime.now().isoformat()
    }