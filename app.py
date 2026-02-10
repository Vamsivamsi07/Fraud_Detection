from flask import Flask, render_template, request, jsonify
import pickle, numpy as np
import os
import logging
import json
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

# Setup logging
logging.basicConfig(
    filename='fraud_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Statistics file
STATS_FILE = 'predictions_stats.json'

def load_stats():
    """Load prediction statistics from file"""
    try:
        if Path(STATS_FILE).exists():
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Error loading stats: {e}")
    return {'total': 0, 'fraud': 0, 'legitimate': 0, 'avg_confidence': 0}

def save_stats(stats):
    """Save prediction statistics to file"""
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        logging.error(f"Error saving stats: {e}")

def record_prediction(is_fraud, confidence):
    """Record prediction in statistics"""
    try:
        stats = load_stats()
        stats['total'] += 1
        if is_fraud:
            stats['fraud'] += 1
        else:
            stats['legitimate'] += 1
        # Update average confidence
        if stats['total'] > 0:
            stats['avg_confidence'] = round((stats['avg_confidence'] * (stats['total'] - 1) + confidence) / stats['total'], 2)
        save_stats(stats)
        logging.info(f"Prediction recorded - Fraud: {is_fraud}, Confidence: {confidence}%")
    except Exception as e:
        logging.error(f"Error recording prediction: {e}")

try:
    model = pickle.load(open('svm_model.pkl','rb'))
    encoder = pickle.load(open('encoder.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))
    logging.info('✓ Model files loaded successfully')
    print('✓ Model files loaded successfully')
except FileNotFoundError as e:
    logging.error(f'Model files not found: {e}')
    print(f'❌ Error: Model files not found - {e}')
    print('Please run train_model.py first')

@app.route('/')
def home():
    try:
        stats = load_stats()
        return render_template('home.html', stats=stats)
    except Exception as e:
        logging.error(f"Error loading home page: {e}")
        return render_template('home.html', stats={'total': 0, 'fraud': 0, 'legitimate': 0})

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/stats')
def stats():
    """Get current statistics as JSON"""
    try:
        stats_data = load_stats()
        return jsonify(stats_data)
    except Exception as e:
        logging.error(f"Error fetching stats: {e}")
        return jsonify({'error': str(e)}), 500

def validate_input(step, tx_type, amount, oldOrg, newOrg, oldDest, newDest):
    """Validate transaction input data"""
    errors = []
    
    # Validate step
    if step < 0:
        errors.append("Step must be non-negative")
    if step > 1000000:
        errors.append("Step value seems too large")
    
    # Validate amount
    if amount < 0:
        errors.append("Amount cannot be negative")
    if amount > 10000000:
        errors.append("Amount seems unreasonably large")
    
    # Validate balances
    for balance in [oldOrg, newOrg, oldDest, newDest]:
        if balance < 0:
            errors.append("Balance values cannot be negative")
            break
    
    # Validate balance changes
    balance_change = abs(oldOrg - newOrg)
    if abs(balance_change - amount) > 0.01 and balance_change > 0:
        errors.append("Warning: Amount doesn't match balance change")
    
    # Validate transaction type
    valid_types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT']
    if tx_type not in valid_types:
        errors.append(f"Invalid transaction type. Must be one of: {', '.join(valid_types)}")
    
    return errors

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Extract form data
        step = float(request.form.get('step', 0))
        tx_type = request.form.get('type', '').strip()
        amount = float(request.form.get('amount', 0))
        oldOrg = float(request.form.get('oldbalanceOrg', 0))
        newOrg = float(request.form.get('newbalanceOrig', 0))
        oldDest = float(request.form.get('oldbalanceDest', 0))
        newDest = float(request.form.get('newbalanceDest', 0))

        # Validate input
        validation_errors = validate_input(step, tx_type, amount, oldOrg, newOrg, oldDest, newDest)
        if validation_errors:
            logging.warning(f"Validation errors: {validation_errors}")
            return render_template('submit.html', 
                                 prediction='Error', 
                                 error='Invalid input: ' + '; '.join(validation_errors))

        # Encode and transform
        try:
            tx_type_encoded = encoder.transform([tx_type])[0]
        except ValueError:
            logging.error(f"Invalid transaction type: {tx_type}")
            return render_template('submit.html',
                                 prediction='Error',
                                 error=f"Invalid transaction type: {tx_type}")

        # Prepare features
        features = scaler.transform([[step, tx_type_encoded, amount, oldOrg, newOrg, oldDest, newDest]])
        
        # Make prediction
        pred = model.predict(features)[0]
        prediction_text = 'Is Fraud' if pred == 1 else 'Not Fraud'
        confidence = model.predict_proba(features)[0]
        confidence_score = max(confidence) * 100

        # Record prediction
        record_prediction(pred == 1, round(confidence_score, 2))

        logging.info(f"Prediction: {prediction_text}, Confidence: {confidence_score:.2f}%")
        
        return render_template('submit.html', 
                             prediction=prediction_text, 
                             confidence=f'{confidence_score:.2f}')
    except ValueError as e:
        logging.error(f"ValueError in submit: {e}")
        return render_template('submit.html', 
                             prediction='Error', 
                             error='Invalid input format. Please ensure all values are valid numbers.')
    except Exception as e:
        logging.error(f"Unexpected error in submit: {e}")
        return render_template('submit.html', 
                             prediction='Error', 
                             error=f'An unexpected error occurred: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
