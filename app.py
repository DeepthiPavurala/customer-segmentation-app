from flask import Flask, request, jsonify, render_template, send_file
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime



app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


with open('KMeans_model.pkl','rb') as file:
    model = pickle.load(file)


def load_and_clean_data(file_path):

    # Load Data
    transactions = pd.read_csv(file_path, header = 0)

    # Cleaning
    transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
    max_date = max(transactions['TransactionDate'])
    transactions['Recency'] = (max_date - transactions['TransactionDate']).dt.days
    rfm_trans_f = transactions.groupby('CustomerID')['TransactionID'].count().reset_index()
    rfm_trans_r = transactions.groupby('CustomerID')['Recency'].min().reset_index()
    rfm_trans_m = transactions.groupby('CustomerID')['TransactionAmount (INR)'].sum().reset_index()
    rfm_trans = pd.merge(rfm_trans_f, rfm_trans_m, on='CustomerID', how='inner')
    rfm_trans = pd.merge(rfm_trans, rfm_trans_r, on='CustomerID', how='inner')
    rfm_trans.columns = ['CustomerID','Frequency', 'Monetary', 'Recency']

    return rfm_trans

def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    numeric_cols = rfm.select_dtypes(include = np.number).columns.to_list()
    rfm_transactions = rfm[numeric_cols]

    scaler = StandardScaler()
    scaled_rfm_transactions = scaler.fit_transform(rfm_transactions)
    scaled_rfm_transactions = pd.DataFrame(scaled_rfm_transactions, columns=numeric_cols)

    return rfm, scaled_rfm_transactions



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    file = request.files['file']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_filename = file.filename.rsplit('.', 1)[0]
    uploaded_filename = f"{original_filename}_{timestamp}.csv"
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    rfm_trans, rfm = preprocess_data(file_path)
    results_df = model.predict(rfm)
    rfm_trans['Cluster'] = results_df

    # saving output file
    output_filename = f"prediction_results_{timestamp}.csv"
    output_path = os.path.join(RESULT_FOLDER, output_filename)
    rfm_trans.to_csv(output_path, index=False)
    print(f'Verify df shape: {rfm_trans.shape}')
    # Generate images and save them
    for col in ['Frequency','Recency', 'Monetary']:
        if col in rfm_trans.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x='Cluster', y= col, data=rfm_trans)
            plt.title(f'{col} by Cluster')
            plt.tight_layout()
            plot_path = os.path.join(STATIC_FOLDER, f'{col}.png')
            plt.savefig(plot_path)
            plt.close()  
            print("Saving plot:", plot_path)
    return render_template('index.html', show_results=True)

@app.route('/download')
def download():
    files = sorted(
        [f for f in os.listdir(RESULT_FOLDER) if f.endswith('.csv')],
        key=lambda x: os.path.getctime(os.path.join(RESULT_FOLDER, x)),
        reverse=True
    )
    latest_file = files[0]
    return send_file(
        os.path.join(RESULT_FOLDER, latest_file),
        download_name=latest_file,
        as_attachment=True
    )

        
if __name__ == '__main__':
    app.run(debug=True)