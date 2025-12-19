import json
import boto3
import joblib
import pandas as pd
from datetime import datetime
from io import BytesIO

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

BUCKET_NAME = 'loan-approval-ml-bucket'
MODEL_KEY = 'models/loan_approval_pipeline.pkl'
DYNAMODB_TABLE = 'LoanApprovalPredictions'

#load model from S3 (cache en memoria)
MODEL_CACHE = {}

def get_model():
    """Load model from S3 with caching"""
    if 'pipeline' not in MODEL_CACHE:
        print("Loading model from S3...")
        model_obj = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
        model_bytes = BytesIO(model_obj['Body'].read())
        MODEL_CACHE['pipeline'] = joblib.load(model_bytes)
        print("✓ Model loaded successfully")
    return MODEL_CACHE['pipeline']


def lambda_handler(event, context):
    """
    Make loan approval predictions using trained Random Forest model
    """
    print("=== LOAN APPROVAL ML PREDICTOR ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    try:
        today = datetime.now().strftime('%Y%m%d')
        processed_key = f'processed/{today}/applications_{today}.json'
        
        #read processed applications
        print(f"Reading: {processed_key}")
        processed_obj = s3.get_object(Bucket=BUCKET_NAME, Key=processed_key)
        processed_data = json.loads(processed_obj['Body'].read().decode('utf-8'))
        records = processed_data.get('records', [])
        
        print(f"Found {len(records)} applications to process")
        
        #load model
        pipeline = get_model()
        
        #convert to DataFrame
        df = pd.DataFrame(records)
        
        #keep application metadata
        metadata_cols = ['application_id', 'timestamp']
        metadata = df[metadata_cols] if all(col in df.columns for col in metadata_cols) else None
        
        #drop metadata for prediction
        feature_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                       'person_gender', 'employment_type', 'person_home_ownership',
                       'loan_intent', 'account_type', 'person_education',
                       'previous_loan_defaults_on_file']
        
        X = df[feature_cols]
        
        #make predictions
        print("Making predictions...")
        predictions = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)
        
        #build results
        results = []
        approved_count = 0
        rejected_count = 0
        
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            decision = 'APPROVED' if pred == 1 else 'REJECTED'
            confidence = float(proba[pred])
            
            if pred == 1:
                approved_count += 1
            else:
                rejected_count += 1
            
            result = {
                'application_id': records[i].get('application_id', f'app_{i}'),
                'decision': decision,
                'confidence': round(confidence, 4),
                'loan_amount': float(records[i].get('loan_amnt', 0)),
                'person_income': float(records[i].get('person_income', 0)),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
        
        stats = {
            'total_applications': len(results),
            'approved': approved_count,
            'rejected': rejected_count,
            'approval_rate': round(approved_count / len(results) * 100, 2) if results else 0
        }
        
        print(f"Predictions complete: {stats}")
        
        #save predictions to S3
        output_key = f'predictions/{today}/predictions_{today}.json'
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=output_key,
            Body=json.dumps({
                'stats': stats,
                'predictions': results,
                'timestamp': datetime.now().isoformat()
            }, indent=2),
            ContentType='application/json'
        )
        print(f"Saved to: s3://{BUCKET_NAME}/{output_key}")
        
        #save to DynamoDB
        table = dynamodb.Table(DYNAMODB_TABLE)
        saved_count = 0
        
        for result in results:
            try:
                table.put_item(Item={
                    'application_id': result['application_id'],
                    'date': today,
                    'decision': result['decision'],
                    'confidence': str(result['confidence']),
                    'loan_amount': str(result['loan_amount']),
                    'person_income': str(result['person_income']),
                    'timestamp': result['timestamp']
                })
                saved_count += 1
            except Exception as e:
                print(f"Error saving to DynamoDB: {e}")
        
        print(f"Saved {saved_count} predictions to DynamoDB")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Predictions completed',
                'stats': stats,
                'dynamodb_saved': saved_count,
                'output_file': output_key
            })
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }