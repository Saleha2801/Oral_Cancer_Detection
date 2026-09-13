import requests
import json

URL = "http://127.0.0.1:8000/api/predict-multimodal"

def run_test(name, files, data, expected_status=200):
    print(f"\n--- Running Test: {name} ---")
    r = requests.post(URL, files=files, data=data)
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        res = r.json()
        fused = res["fused_result"]
        print(f"Prediction: {fused['fused_prediction']}")
        print(f"Active Modalities ({fused['active_modalities_count']}): {fused['active_modalities']}")
        print(f"Prob: {fused['cancer_probability']} | Uncertainty: {fused['uncertainty_level']}")
        if fused.get("discordance_alert"):
            print(f"Alert: {fused['discordance_alert']}")
    else:
        print(f"Error Response: {r.text}")

# 1. Oral Only
with open("Clinical oral images/cancer/373.jpeg", "rb") as f_oral:
    run_test("1. Oral Image Only", {"clinical_image": f_oral}, {})

# 2. Histopathology Only
with open("NDU-UFES/histopathological images/0000.png", "rb") as f_histo:
    run_test("2. Histopathology Only", {"histopathology_image": f_histo}, {})

# 3. Clinical Data Only
run_test("3. Clinical Data Only", {}, {"clinical_data": json.dumps({"localization": "Tongue", "larger_size": 2.5, "tobacco_use": "Yes", "alcohol_consumption": "Yes", "gender": "M", "age_group": 2})})

# 4. Oral + Histopathology
with open("Clinical oral images/cancer/373.jpeg", "rb") as f_oral, open("NDU-UFES/histopathological images/0000.png", "rb") as f_histo:
    run_test("4. Oral + Histopathology", {"clinical_image": f_oral, "histopathology_image": f_histo}, {})

# 5. Oral + Clinical Data
with open("Clinical oral images/cancer/373.jpeg", "rb") as f_oral:
    run_test("5. Oral + Clinical Data", {"clinical_image": f_oral}, {"clinical_data": json.dumps({"localization": "Lip", "larger_size": 0.5, "tobacco_use": "No", "alcohol_consumption": "No", "gender": "F", "age_group": 0})})

# 6. Histopathology + Clinical Data
with open("NDU-UFES/histopathological images/0008.png", "rb") as f_histo:
    run_test("6. Histopathology + Clinical Data", {"histopathology_image": f_histo}, {"clinical_data": json.dumps({"localization": "Lip", "larger_size": 0.5, "tobacco_use": "No", "alcohol_consumption": "No", "gender": "F", "age_group": 0})})

# 7. All Three Modalities
with open("Clinical oral images/cancer/373.jpeg", "rb") as f_oral, open("NDU-UFES/histopathological images/0000.png", "rb") as f_histo:
    run_test("7. All Three Modalities (Triple)", {"clinical_image": f_oral, "histopathology_image": f_histo}, {"clinical_data": json.dumps({"localization": "Tongue", "larger_size": 2.5, "tobacco_use": "Yes", "alcohol_consumption": "Yes", "gender": "M", "age_group": 2})})

# 8. Empty input (Validation error)
run_test("8. Missing Required Input (Should 400)", {}, {}, expected_status=400)
