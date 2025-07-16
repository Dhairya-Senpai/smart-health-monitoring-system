import pandas as pd
import numpy as np
import time

# Real-time data simulation for Smart Health Monitoring System

def generate_vital_signs(num_patients=5):
    data = []
    for pid in range(1, num_patients+1):
        heart_rate = np.random.randint(60, 120)
        bp_systolic = np.random.randint(100, 160)
        bp_diastolic = np.random.randint(60, 100)
        spo2 = np.random.randint(85, 100)
        data.append({
            'patient_id': pid,
            'heart_rate': heart_rate,
            'blood_pressure_systolic': bp_systolic,
            'blood_pressure_diastolic': bp_diastolic,
            'spo2': spo2
        })
    return pd.DataFrame(data)

def simulate_stream(interval=2, iterations=5):
    for _ in range(iterations):
        df = generate_vital_signs()
        print(df)
        time.sleep(interval)

if __name__ == "__main__":
    simulate_stream()
