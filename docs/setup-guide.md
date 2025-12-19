# Setup Guide - Loan Approval ML System

Quick deployment guide for AWS serverless ML pipeline.

---

## Prerequisites

- AWS Account with admin access
- AWS CLI configured (`aws configure`)
- Python 3.8+ installed
- Git installed

---

## 1. Clone & Install

```bash
git clone https://github.com/msardamasri/Automated-Loan-Approval-ML-System.git
cd Automated-Loan-Approval-ML-System
cd /requirements
pip install -r requirements.txt
```

---

## 2. AWS Infrastructure Setup

### Create S3 Bucket
```bash
aws s3 mb s3://loan-approval-ml-bucket
aws s3api put-bucket-versioning --bucket loan-approval-ml-bucket --versioning-configuration Status=Enabled
```

### Create DynamoDB Table
```bash
aws dynamodb create-table \
    --table-name LoanApprovalPredictions \
    --attribute-definitions \
        AttributeName=application_id,AttributeType=S \
        AttributeName=date,AttributeType=S \
    --key-schema \
        AttributeName=application_id,KeyType=HASH \
        AttributeName=date,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST
```

### Create IAM Role for Lambda
```bash
aws iam create-role \
    --role-name LoanApprovalLambdaRole \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'

aws iam attach-role-policy \
    --role-name LoanApprovalLambdaRole \
    --policy-arn arn:aws:iam::aws:policy/AWSLambdaExecute

aws iam attach-role-policy \
    --role-name LoanApprovalLambdaRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
```

---

## 3. Train & Upload Model

```bash
#open notebook and run all cells
jupyter notebook notebooks/loan_approval_production.ipynb

#model will be saved locally and uploaded to S3 automatically
#verify upload
aws s3 ls s3://loan-approval-ml-bucket/models/
```

---

## 4. Deploy Lambda Functions

### Package Lambda 1: Extractor
```bash
cd lambdas
zip lambda-loan-extractor.zip lambda-loan-extractor.py

aws lambda create-function \
    --function-name lambda-loan-extractor \
    --runtime python3.9 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/LoanApprovalLambdaRole \
    --handler lambda-loan-extractor.lambda_handler \
    --zip-file fileb://lambda-loan-extractor.zip \
    --timeout 60 \
    --memory-size 512
```

### Package Lambda 2: Processor
```bash
pip install -r requirements-lambda.txt -t package/
cp lambda-loan-processor.py package/
cd package
zip -r ../lambda-loan-processor.zip .
cd ..

aws lambda create-function \
    --function-name lambda-loan-processor \
    --runtime python3.9 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/LoanApprovalLambdaRole \
    --handler lambda-loan-processor.lambda_handler \
    --zip-file fileb://lambda-loan-processor.zip \
    --timeout 120 \
    --memory-size 1024
```

### Package Lambda 3: Predictor
```bash
pip install -r requirements-lambda.txt -t package-predictor/
cp lambda-loan-predictor.py package-predictor/
cd package-predictor
zip -r ../lambda-loan-predictor.zip .
cd ..

aws lambda create-function \
    --function-name lambda-loan-predictor \
    --runtime python3.9 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/LoanApprovalLambdaRole \
    --handler lambda-loan-predictor.lambda_handler \
    --zip-file fileb://lambda-loan-predictor.zip \
    --timeout 300 \
    --memory-size 2048
```

---

## 5. Deploy EC2 API

### Launch EC2 Instance
```bash
#launch Amazon Linux 2 t3.medium instance via console
#security group: allow inbound 22 (SSH) and 8000 (API)
```

### Install Dependencies
```bash
#SSH into EC2
ssh -i your-key.pem ec2-user@YOUR_EC2_IP

#install python and dependencies
sudo yum update -y
sudo yum install python3-pip -y
pip3 install -r requirements-ec2.txt
```

### Deploy API
```bash
#copy API file
scp -i your-key.pem ec2/ml-api-ec2.py ec2-user@YOUR_EC2_IP:~/

#SSH and run
ssh -i your-key.pem ec2-user@YOUR_EC2_IP
python3 ml-api-ec2.py
```

### Run as Background Service (Optional)
```bash
#create systemd service
sudo nano /etc/systemd/system/loan-api.service

#paste:
[Unit]
Description=Loan Approval ML API
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user
ExecStart=/usr/bin/python3 /home/ec2-user/ml-api-ec2.py
Restart=always

[Install]
WantedBy=multi-user.target

#enable and start
sudo systemctl enable loan-api
sudo systemctl start loan-api
sudo systemctl status loan-api
```

---

## 6. Test the Pipeline

### Test Lambda Extractor
```bash
aws lambda invoke \
    --function-name lambda-loan-extractor \
    --payload '{}' \
    response.json
cat response.json
```

### Test Lambda Processor
```bash
aws lambda invoke \
    --function-name lambda-loan-processor \
    --payload '{}' \
    response.json
cat response.json
```

### Test EC2 API
```bash
curl http://YOUR_EC2_IP:8000/health

curl -X POST http://YOUR_EC2_IP:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "person_age": 25,
      "person_income": 50000,
      "person_emp_exp": 3,
      "loan_amnt": 10000,
      "loan_int_rate": 7.5,
      "loan_percent_income": 0.2,
      "cb_person_cred_hist_length": 5,
      "person_gender": "Male",
      "employment_type": "Full-time",
      "person_home_ownership": "RENT",
      "loan_intent": "PERSONAL",
      "account_type": "Checking",
      "person_education": "Bachelor",
      "previous_loan_defaults_on_file": "No"
    }]
  }'
```

### Access Dashboard
```bash
open http://YOUR_EC2_IP:8000/dashboard
```

---

## 7. Verify End-to-End Flow

```bash
#trigger extractor
aws lambda invoke --function-name lambda-loan-extractor --payload '{}' output.json

#wait 30 seconds

#check S3 raw data
aws s3 ls s3://loan-approval-ml-bucket/raw/ --recursive

#check S3 processed data
aws s3 ls s3://loan-approval-ml-bucket/processed/ --recursive

#check S3 predictions
aws s3 ls s3://loan-approval-ml-bucket/predictions/ --recursive

#check DynamoDB
aws dynamodb scan --table-name LoanApprovalPredictions --max-items 5
```

---

## Monitoring

### CloudWatch Logs
```bash
#lambda logs
aws logs tail /aws/lambda/lambda-loan-extractor --follow
aws logs tail /aws/lambda/lambda-loan-processor --follow
aws logs tail /aws/lambda/lambda-loan-predictor --follow

#EC2 logs
ssh ec2-user@YOUR_EC2_IP
tail -f /var/log/messages
```

### Cost Estimation
- **Lambda**: ~$0.20/month (100 invocations/day)
- **S3**: ~$1/month (1GB storage)
- **DynamoDB**: ~$0.50/month (on-demand)
- **EC2**: ~$30/month (t3.medium, 24/7)
- **Total**: ~$32/month

---

## Cleanup

```bash
#delete Lambda functions
aws lambda delete-function --function-name lambda-loan-extractor
aws lambda delete-function --function-name lambda-loan-processor
aws lambda delete-function --function-name lambda-loan-predictor

#delete S3 bucket
aws s3 rm s3://loan-approval-ml-bucket --recursive
aws s3 rb s3://loan-approval-ml-bucket

#delete DynamoDB table
aws dynamodb delete-table --table-name LoanApprovalPredictions

#terminate EC2 instance
aws ec2 terminate-instances --instance-ids YOUR_INSTANCE_ID

#delete IAM role
aws iam detach-role-policy --role-name LoanApprovalLambdaRole --policy-arn arn:aws:iam::aws:policy/AWSLambdaExecute
aws iam detach-role-policy --role-name LoanApprovalLambdaRole --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess
aws iam delete-role --role-name LoanApprovalLambdaRole
```

---

**Setup Complete!** Loan approval ML system is now running on AWS.