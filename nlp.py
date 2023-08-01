from flask import Flask, render_template, request, send_file, make_response
from sklearn.svm import OneClassSVM
import json
import csv
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import glob
import pandas as pd
import io
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


def jsonify(data):
    return json.dumps(data)


app.jinja_env.filters['jsonify'] = jsonify


# Function to convert int64 values to int
def convert_int64(value):
    if isinstance(value, np.int64):
        return int(value)
    return value


# Function to vectorize text data using TfidfVectorizer
def vectorize_text(texts):
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(texts)
    return vectorized_data


@app.route('/')
def index():
    file_paths = glob.glob('*.json') + glob.glob('*.csv')
    results = []
    i=0
    for file_path in file_paths:
        file_extension = file_path.split('.')[-1].lower()
        print(file_paths)

        if file_extension == 'json':
            with open(file_path, 'r') as file:
                data = json.load(file)
            text = ' '.join(element.get('Text', '') for element in data.get('elements', []))
            tokens = word_tokenize(text)

            sia = SentimentIntensityAnalyzer()

            invoice_number = re.search(r"Invoice number\s*([A-Z0-9]+)", text)
            invoice_number = invoice_number.group(1) if invoice_number else None

            invoice_date = re.search(r"INVOICE DATE\s*([\d-]+)", text)
            invoice_date = invoice_date.group(1) if invoice_date else None

            due_date = re.search(r"Due Date\s*([\d-]+)", text)
            due_date = due_date.group(1) if due_date else None

            customer_id = re.search(r"CUSTOMER ID\s*([A-Z0-9]+)", text)
            customer_id = customer_id.group(1) if customer_id else None

            date = re.search(r"Date\s*([\d-]+)", text)
            date = date.group(1) if date else None

            po = re.search(r"PO\s*([A-Z0-9-]+)", text)
            po = po.group(1) if po else None

            order_number = re.search(r"ORDER #\s*([A-Z0-9]+)", text)
            order_number = order_number.group(1) if order_number else None

            total_due = re.search(r"Total Due\s*([\d.]+)", text)
            total_due = total_due.group(1) if total_due else None

            sentiment_scores = sia.polarity_scores(text)
            sentiment_label = 'positive' if sentiment_scores['compound'] >= 0 else 'negative'

            results.append({
                'pdf_file_name': file_paths[i],
                'text': text,
                'invoice_number': convert_int64(invoice_number),
                'invoice_date': convert_int64(invoice_date),
                'customer_id': convert_int64(customer_id),
                'po': convert_int64(po),
                'order_number': convert_int64(order_number),
                'total_due': convert_int64(total_due),
                'due_date': convert_int64(due_date),
                'sentiment_score': sentiment_scores['compound'],
                'sentiment_label': sentiment_label,
            })
            i += 1

        elif file_extension == 'csv':
            df = pd.read_csv(file_path, header=None)
            data = df.set_index(0)[1].to_dict()
            text = ' '.join(key.strip() + " " + value.strip() for key, value in data.items())

            sia = SentimentIntensityAnalyzer()

            invoice_number = re.search(r"Invoice Number\s*([A-Z0-9-]+)", text)
            invoice_number = invoice_number.group(1) if invoice_number else None

            invoice_date = re.search(r"Invoice Date\s*([\w\s,]+)(?=\s+Due Date|$)", text)
            invoice_date = invoice_date.group(1) if invoice_date else None

            due_date = re.search(r"Due Date\s*([\w\s,]+)(?=\s+Total Due|$)", text)
            due_date = due_date.group(1) if due_date else None

            customer_id = re.search(r"CUSTOMER ID\s*([A-Z0-9]+)", text)
            customer_id = customer_id.group(1) if customer_id else None

            date = re.search(r"Date\s*([\d-]+)", text)
            date = date.group(1) if date else None

            po = re.search(r"PO\s*([A-Z0-9-]+)", text)
            po = po.group(1) if po else None

            order_number = re.search(r"Order Number\s*([0-9]+)", text)
            order_number = order_number.group(1) if order_number else None

            total_due = re.search(r"Total Due\s*([\d.$]+)", text)
            total_due = total_due.group(1) if total_due else None

            sentiment_scores = sia.polarity_scores(text)
            sentiment_label = 'positive' if sentiment_scores['compound'] >= 0 else 'negative'

            results.append({
                'pdf_file_name': file_paths[i],
                'text': text,
                'invoice_number': convert_int64(invoice_number),
                'invoice_date': convert_int64(invoice_date),
                'due_date': convert_int64(due_date),
                'customer_id': convert_int64(customer_id),
                'po': convert_int64(po),
                'order_number': convert_int64(order_number),
                'total_due': convert_int64(total_due),
                'sentiment_score': sentiment_scores['compound'],
                'sentiment_label': sentiment_label,
            })

        else:
            print("Unsupported file format for file:", file_path)
            continue

        if not isinstance(text, str):
            print("Invalid text format for file:", file_path)
            continue

    # Vectorize the text data
    texts = [result['text'] for result in results]
    vectorized_data = vectorize_text(texts)
    print(vectorized_data)

    # Train the machine learning model
    model = OneClassSVM()
    model.fit(vectorized_data)

    # Use the trained model for predictions
    predictions = model.predict(vectorized_data)

    for result, prediction in zip(results, predictions):
        result['prediction'] = convert_int64(prediction)
    print(results)
    return render_template('index.html', results=results)


@app.route('/export', methods=['POST'])
def export_csv():
    try:
        results = request.form.get('results')
        results = json.loads(results)
    except json.JSONDecodeError:
        return "Invalid JSON data"

    # Generate CSV data
    csv_data = io.StringIO()
    csv_writer = csv.writer(csv_data)
    csv_writer.writerow(results[0].keys())  # Write header row
    for result in results:
        csv_writer.writerow(result.values())

    # Prepare response
    response = make_response(csv_data.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=output.csv'
    response.headers['Content-type'] = 'text/csv'

    return response


if __name__ == '__main__':
    app.run(port=5001)
