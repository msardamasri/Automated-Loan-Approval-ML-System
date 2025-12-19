from flask import Flask, jsonify, request
import boto3
import joblib
import json
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
from botocore.config import Config

app = Flask(__name__)

BUCKET_NAME = '...'#change your bucket name
MODEL_KEY = 'models/loan_approval_pipeline.pkl'#model path in s3

s3 = boto3.client('s3', config=Config(
    signature_version='s3v4',
    connect_timeout=3,
    read_timeout=5,
    retries={'max_attempts': 1}
))

CACHE = {
    'data': None,
    'timestamp': None,
    'ttl': 300
}

model = None


def load_model():
    global model
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
        model = joblib.load(BytesIO(obj['Body'].read()))
        print("Model loaded")
    except Exception as e:
        print(f"Could not load model: {e}")
        model = None


def predict_rule_based(record):
    """rule-based fallback if ML model fails"""
    loan_percent_income = record.get('loan_percent_income', 0)
    person_income = record.get('person_income', 0)
    loan_amnt = record.get('loan_amnt', 0)
    previous_defaults = record.get('previous_loan_defaults_on_file', 'No')
    
    if previous_defaults == 'Yes' or loan_percent_income > 0.4:
        prediction = 'REJECTED'
        confidence = 0.75
    elif loan_percent_income < 0.2 and person_income > 50000:
        prediction = 'APPROVED'
        confidence = 0.80
    elif loan_amnt > 35000 and person_income < 40000:
        prediction = 'REJECTED'
        confidence = 0.70
    else:
        prediction = 'APPROVED'
        confidence = 0.65
    
    return prediction, confidence


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        records = data.get('records', [])
        
        if not records:
            return jsonify({'error': 'No records provided'}), 400
        
        print(f"Received {len(records)} applications for prediction")
        
        predictions = []
        approved_count = 0
        rejected_count = 0
        
        use_model = model is not None
        print(f"Using {'ML Model' if use_model else 'Rule-based System'}")
        
        for record in records:
            try:
                if use_model:
                    try:
                        #prepare dataframe for model
                        df = pd.DataFrame([record])
                        feature_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                                       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                                       'person_gender', 'employment_type', 'person_home_ownership',
                                       'loan_intent', 'account_type', 'person_education',
                                       'previous_loan_defaults_on_file']
                        
                        X = df[feature_cols]
                        
                        pred_class = model.predict(X)[0]
                        pred_proba = model.predict_proba(X)[0]
                        confidence = float(pred_proba[pred_class])
                        
                        prediction = 'APPROVED' if pred_class == 1 else 'REJECTED'
                        
                    except Exception as model_error:
                        print(f"Model prediction failed, using rules: {model_error}")
                        prediction, confidence = predict_rule_based(record)
                else:
                    prediction, confidence = predict_rule_based(record)
                
                if prediction == 'APPROVED':
                    approved_count += 1
                else:
                    rejected_count += 1
                
                predictions.append({
                    'application_id': record.get('application_id', 'N/A'),
                    'person_age': record.get('person_age', 0),
                    'person_income': record.get('person_income', 0),
                    'loan_amnt': record.get('loan_amnt', 0),
                    'loan_percent_income': record.get('loan_percent_income', 0),
                    'decision': prediction,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"Error predicting application {record.get('application_id', 'Unknown')}: {e}")
                continue
        
        stats = {
            'total_applications': len(predictions),
            'approved_count': approved_count,
            'rejected_count': rejected_count,
            'approval_rate': round(approved_count / len(predictions) * 100, 2) if predictions else 0
        }
        
        print(f"Generated {len(predictions)} predictions")
        print(f"Stats: APPROVED={stats['approved_count']}, REJECTED={stats['rejected_count']}, Rate={stats['approval_rate']}%")
        
        return jsonify({
            'predictions': predictions,
            'stats': stats,
            'method': 'ml_model' if use_model else 'rule_based'
        })
        
    except Exception as e:
        print(f"Batch prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def get_cached_data():
    if CACHE['data'] is None or CACHE['timestamp'] is None:
        return None
    
    elapsed = (datetime.now() - CACHE['timestamp']).total_seconds()
    if elapsed < CACHE['ttl']:
        return CACHE['data']
    
    return None


def load_predictions_fast():
    cached = get_cached_data()
    if cached:
        print("Serving from cache")
        return cached
    
    print("Loading from S3...")
    
    for days_ago in range(2):
        try:
            check_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y%m%d')
            key = f'predictions/{check_date}/predictions_{check_date}.json'
            
            print(f"   Trying {check_date}...")
            
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            content = obj['Body'].read(1024 * 1024).decode('utf-8')
            data = json.loads(content)
            
            if 'predictions' in data:
                approved = [p for p in data['predictions'] if p['decision'] == 'APPROVED'][:20]
                rejected = [p for p in data['predictions'] if p['decision'] == 'REJECTED'][:20]
                
                result = {
                    'predictions': approved + rejected,
                    'total_applications': data.get('stats', {}).get('total_applications', 0),
                    'approved_count': data.get('stats', {}).get('approved_count', 0),
                    'rejected_count': data.get('stats', {}).get('rejected_count', 0),
                    'approval_rate': data.get('stats', {}).get('approval_rate', 0),
                    'timestamp': datetime.now().isoformat(),
                    'date': check_date
                }
            else:
                result = data
            
            CACHE['data'] = result
            CACHE['timestamp'] = datetime.now()
            
            print(f"Loaded {len(result.get('predictions', []))} predictions")
            return result
            
        except Exception as e:
            print(f"   {check_date}: {str(e)[:50]}")
            continue
    
    print("No data found, returning empty")
    return {
        'predictions': [],
        'total_applications': 0,
        'approved_count': 0,
        'rejected_count': 0,
        'approval_rate': 0,
        'timestamp': datetime.now().isoformat()
    }


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'cache_valid': get_cached_data() is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predictions/latest', methods=['GET'])
def get_latest():
    try:
        data = load_predictions_fast()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predictions/summary', methods=['GET'])
def get_summary():
    try:
        data = load_predictions_fast()
        return jsonify({
            'total_applications': data.get('total_applications', 0),
            'approved_count': data.get('approved_count', 0),
            'rejected_count': data.get('rejected_count', 0),
            'approval_rate': data.get('approval_rate', 0),
            'timestamp': data.get('timestamp', datetime.now().isoformat())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard', methods=['GET'])
def dashboard():
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Approval Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #e5e5e5;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1500px; margin: 0 auto; }
        
        .header {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            padding: 35px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .header h1 { 
            font-size: 30px; 
            font-weight: 700;
            margin-bottom: 8px;
            color: #ffffff;
        }
        .header p { color: #a3a3a3; font-size: 14px; margin-bottom: 12px; }
        .header .meta { font-size: 12px; color: #737373; }
        
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 25px;
        }
        .kpi-card {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            padding: 24px;
            border-radius: 10px;
            transition: all 0.2s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .kpi-card:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            border-color: #3a3a3a;
        }
        .kpi-title {
            font-size: 11px;
            text-transform: uppercase;
            color: #a3a3a3;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }
        .kpi-value {
            font-size: 34px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .kpi-subtitle { font-size: 13px; color: #737373; }
        .kpi-card.total .kpi-value { color: #ffffff; }
        .kpi-card.approved .kpi-value { color: #22c55e; }
        .kpi-card.rejected .kpi-value { color: #ef4444; }
        .kpi-card.rate .kpi-value { color: #3b82f6; }
        
        .insight {
            background: #262626;
            border: 1px solid #3a3a3a;
            padding: 20px 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .insight-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            color: #d4d4d4;
        }
        .insight-text { font-size: 15px; font-weight: 500; line-height: 1.5; color: #e5e5e5; }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin-bottom: 25px;
        }
        .chart-card {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            padding: 24px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .chart-card.wide {
            grid-column: 1 / -1;
        }
        .chart-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 18px;
            color: #d4d4d4;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        table {
            width: 100%;
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        th, td { padding: 14px 16px; text-align: left; }
        th {
            background: #0a0a0a;
            font-size: 11px;
            text-transform: uppercase;
            color: #a3a3a3;
            font-weight: 600;
            border-bottom: 1px solid #2a2a2a;
        }
        td { color: #e5e5e5; border-bottom: 1px solid #2a2a2a; }
        tbody tr:hover { background: #262626; }
        tbody tr:last-child td { border-bottom: none; }
        
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .badge.approved { 
            background: rgba(34, 197, 94, 0.15); 
            color: #86efac; 
            border: 1px solid rgba(34, 197, 94, 0.3); 
        }
        .badge.rejected { 
            background: rgba(239, 68, 68, 0.15); 
            color: #fca5a5; 
            border: 1px solid rgba(239, 68, 68, 0.3); 
        }
        
        .loading {
            text-align: center;
            padding: 60px 20px;
            color: #a3a3a3;
        }
        .spinner {
            border: 3px solid #262626;
            border-top: 3px solid #a3a3a3;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 1024px) {
            .chart-grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 768px) {
            .kpi-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Loan Approval Analytics Dashboard</h1>
            <p>AI-powered predictive analytics for automated loan decision-making</p>
            <div class="meta">MODEL: Random Forest Classifier | Precision: 90.46% | <span id="update-time">Last update: --</span></div>
        </div>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <div>Loading analytics...</div>
        </div>
        
        <div id="content" style="display:none;">
            <div class="kpi-grid">
                <div class="kpi-card total">
                    <div class="kpi-title">Total Applications</div>
                    <div class="kpi-value" id="total">0</div>
                    <div class="kpi-subtitle">Processed</div>
                </div>
                <div class="kpi-card approved">
                    <div class="kpi-title">Approved</div>
                    <div class="kpi-value" id="approved">0</div>
                    <div class="kpi-subtitle"><span id="approved-pct">0</span>% | Automatic approval</div>
                </div>
                <div class="kpi-card rejected">
                    <div class="kpi-title">Rejected</div>
                    <div class="kpi-value" id="rejected">0</div>
                    <div class="kpi-subtitle"><span id="rejected-pct">0</span>% | Declined</div>
                </div>
                <div class="kpi-card rate">
                    <div class="kpi-title">Approval Rate</div>
                    <div class="kpi-value" id="rate">0%</div>
                    <div class="kpi-subtitle">Overall performance</div>
                </div>
            </div>
            
            <div class="insight">
                <div class="insight-label">Key Insight</div>
                <div class="insight-text" id="insight">Analyzing data...</div>
            </div>
            
            <div class="chart-grid">
                <div class="chart-card">
                    <div class="chart-title">Decision Distribution</div>
                    <canvas id="pie"></canvas>
                </div>
                <div class="chart-card">
                    <div class="chart-title">Applications by Decision Type</div>
                    <canvas id="bar"></canvas>
                </div>
                <div class="chart-card wide">
                    <div class="chart-title">Decision Distribution Across Loan Amounts</div>
                    <canvas id="combined"></canvas>
                </div>
            </div>
            
            <div class="chart-card" style="margin-bottom:0;">
                <div class="chart-title">Recent Loan Decisions (Top 20)</div>
                <table>
                    <thead>
                        <tr>
                            <th>Application ID</th>
                            <th>Decision</th>
                            <th>Confidence</th>
                            <th>Loan Amount</th>
                            <th>Income</th>
                            <th>Loan/Income</th>
                        </tr>
                    </thead>
                    <tbody id="table"></tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        fetch('/predictions/latest')
            .then(r => r.json())
            .then(data => {
                if (data.timestamp) {
                    const dt = new Date(data.timestamp);
                    document.getElementById('update-time').textContent = 
                        'Last update: ' + dt.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
                }
                
                const t = data.total_applications || 0;
                const a = data.approved_count || 0;
                const r = data.rejected_count || 0;
                const rate = data.approval_rate || 0;
                
                document.getElementById('total').textContent = t.toLocaleString();
                document.getElementById('approved').textContent = a.toLocaleString();
                document.getElementById('rejected').textContent = r.toLocaleString();
                document.getElementById('rate').textContent = rate.toFixed(1) + '%';
                
                if (t > 0) {
                    document.getElementById('approved-pct').textContent = ((a/t)*100).toFixed(1);
                    document.getElementById('rejected-pct').textContent = ((r/t)*100).toFixed(1);
                    
                    document.getElementById('insight').textContent = 
                        `${rate.toFixed(0)}% approval rate with ${a} applications automatically approved. Model maintains 90.46% precision on historical data.`;
                }
                
                new Chart(document.getElementById('pie'), {
                    type: 'doughnut',
                    data: {
                        labels: ['Approved', 'Rejected'],
                        datasets: [{
                            data: [a, r],
                            backgroundColor: ['#22c55e', '#ef4444'],
                            borderWidth: 0
                        }]
                    },
                    options: {
                        plugins: { 
                            legend: { 
                                position: 'bottom', 
                                labels: { color: '#a3a3a3', padding: 15, font: { size: 12 } } 
                            } 
                        }
                    }
                });
                
                new Chart(document.getElementById('bar'), {
                    type: 'bar',
                    data: {
                        labels: ['Approved', 'Rejected'],
                        datasets: [{
                            data: [a, r],
                            backgroundColor: ['#22c55e', '#ef4444'],
                            borderRadius: 6
                        }]
                    },
                    options: {
                        plugins: { legend: { display: false } },
                        scales: { 
                            y: { 
                                beginAtZero: true, 
                                ticks: { color: '#a3a3a3' }, 
                                grid: { color: '#262626' },
                                border: { color: '#2a2a2a' }
                            },
                            x: { 
                                ticks: { color: '#a3a3a3' }, 
                                grid: { display: false },
                                border: { color: '#2a2a2a' }
                            }
                        }
                    }
                });
                
                const preds = data.predictions || [];
                
                const ranges = ['<$10k', '$10-20k', '$20-30k', '>$30k'];
                const decisionsByAmount = {
                    approved: [0, 0, 0, 0],
                    rejected: [0, 0, 0, 0]
                };
                
                preds.forEach(p => {
                    if (!p || !p.loan_amnt) return;
                    let idx = 0;
                    if (p.loan_amnt >= 30000) idx = 3;
                    else if (p.loan_amnt >= 20000) idx = 2;
                    else if (p.loan_amnt >= 10000) idx = 1;
                    
                    if (p.decision === 'APPROVED') decisionsByAmount.approved[idx]++;
                    else if (p.decision === 'REJECTED') decisionsByAmount.rejected[idx]++;
                });
                
                new Chart(document.getElementById('combined'), {
                    type: 'bar',
                    data: {
                        labels: ranges,
                        datasets: [
                            {
                                label: 'Approved',
                                data: decisionsByAmount.approved,
                                backgroundColor: '#22c55e',
                                borderRadius: 4
                            },
                            {
                                label: 'Rejected',
                                data: decisionsByAmount.rejected,
                                backgroundColor: '#ef4444',
                                borderRadius: 4
                            }
                        ]
                    },
                    options: {
                        plugins: { 
                            legend: { 
                                display: true,
                                position: 'top',
                                labels: { color: '#a3a3a3', padding: 15, font: { size: 12 } }
                            } 
                        },
                        scales: { 
                            y: { 
                                beginAtZero: true,
                                stacked: true,
                                ticks: { color: '#a3a3a3' }, 
                                grid: { color: '#262626' },
                                border: { color: '#2a2a2a' }
                            },
                            x: { 
                                stacked: true,
                                ticks: { color: '#a3a3a3' }, 
                                grid: { display: false },
                                border: { color: '#2a2a2a' }
                            }
                        }
                    }
                });
                
                const tbody = document.getElementById('table');
                const recent = preds.slice(0, 20);
                
                recent.forEach((app) => {
                    const badgeClass = app.decision === 'APPROVED' ? 'approved' : 'rejected';
                    tbody.innerHTML += `
                        <tr>
                            <td><strong>${app.application_id||'N/A'}</strong></td>
                            <td><span class="badge ${badgeClass}">${app.decision}</span></td>
                            <td><strong>${((app.confidence||0)*100).toFixed(1)}%</strong></td>
                            <td><strong>$${(app.loan_amnt||0).toLocaleString()}</strong></td>
                            <td>$${(app.person_income||0).toLocaleString()}</td>
                            <td>${((app.loan_percent_income||0)*100).toFixed(1)}%</td>
                        </tr>
                    `;
                });
                
                document.getElementById('loading').style.display = 'none';
                document.getElementById('content').style.display = 'block';
            })
            .catch(err => {
                document.getElementById('loading').innerHTML = 
                    '<div style="color:#ef4444;">Error: ' + err.message + 
                    '<br><button onclick="location.reload()" style="margin-top:15px;padding:10px 20px;cursor:pointer;background:#60a5fa;border:none;border-radius:6px;color:white;font-weight:600;">Retry</button></div>';
            });
    </script>
</body>
</html>
    """
    return html


if __name__ == '__main__':
    load_model()
    print("=" * 60)
    print("Loan Approval Flask API Starting")
    print("=" * 60)
    print("Endpoints:")
    print("   - POST /predict-batch")
    print("   - GET  /health")
    print("   - GET  /predictions/summary")
    print("   - GET  /predictions/latest")
    print("   - GET  /dashboard")
    print("=" * 60)
    print("Optimizations:")
    print("   - Cache: 5 min")
    print("   - S3 timeout: 3s connect, 5s read")
    print("   - Predictions: 20 per category (40 total)")
    print("   - Search: Only last 2 days")
    print("   - Fallback: Rule-based if model fails")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8000, debug=False)