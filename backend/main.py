import os
import json
import uuid
import datetime
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import io
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.inference import multimodal_service
from training.experiment_manager import experiment_manager
from training.dataset_manager import dataset_manager

app = FastAPI(
    title="Oral Cancer AI - Multimodal Detection API",
    description="True Multimodal AI System for Oral Cancer Detection integrating Oral Photography, Histopathology Biopsy, and Clinical Data",
    version="2.0.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
ORAL_CANCER_DIR = os.path.join(ROOT_DIR, "Clinical oral images", "cancer")
ORAL_NON_CANCER_DIR = os.path.join(ROOT_DIR, "Clinical oral images", "non-cancer")
HISTO_DIR = os.path.join(ROOT_DIR, "NDU-UFES", "histopathological images")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "app_name": "Oral Cancer AI",
        "subtitle": "True Multimodal Oral Cancer Detection System",
        "version": "2.0.0",
        "system_status": multimodal_service.get_system_status(),
    }


@app.get("/api/model-info")
def get_model_info():
    return {
        "system_name": "Oral Cancer AI Multimodal Suite",
        "version": "2.0.0",
        "system_status": multimodal_service.get_system_status(),
        "models": {
            "clinical_oral_image_model": {
                "name": "Oral Lesion ResNet-18",
                "architecture": "ResNet-18",
                "trained_dataset": "Clinical oral images (1,081 photographs)",
                "classes": ["Cancer", "Non-cancer"],
                "input_resolution": "224x224 RGB",
                "explainability": "Grad-CAM (layer4[-1].conv2)",
                "weight_in_fusion": 0.35,
            },
            "histopathology_biopsy_model": {
                "name": "Biopsy Histopathology ResNet-18",
                "architecture": "ResNet-18 (Transfer Learning)",
                "trained_dataset": "NDU-UFES Microscopic Cohort (Task II: OSCC vs Leukoplakia)",
                "classes": ["OSCC (Malignant)", "Leukoplakia (Non-cancer)"],
                "input_resolution": "224x224 RGB",
                "explainability": "Grad-CAM (layer4[-1].conv2)",
                "weight_in_fusion": 0.45,
            },
            "clinical_data_model": {
                "name": "Clinical Risk Random Forest",
                "architecture": "Random Forest Classifier (100 estimators, max_depth=5)",
                "trained_dataset": "NDU-UFES Clinical Tabular Records (23 one-hot features)",
                "classes": ["Cancer (OSCC)", "Non-cancer (Leukoplakia)"],
                "test_accuracy": "87.18%",
                "test_roc_auc": "0.9875",
                "weight_in_fusion": 0.20,
            },
        },
        "fusion_protocol": {
            "type": "Calibrated Decision-Level Bayesian Late Fusion",
            "base_weights": {"histopathology": 0.45, "oral_image": 0.35, "clinical_data": 0.20},
            "discordance_policy": "Tissue biopsy histology holds clinical precedence over superficial photography if predictions diverge.",
        },
    }


@app.get("/api/sample-images")
def get_sample_images():
    """
    Returns verified sample images for both Oral Photography and Histopathology Biopsy.
    """
    samples = {"oral": [], "histopathology": []}

    # Oral Cancer samples
    if os.path.exists(ORAL_CANCER_DIR):
        for f in sorted(os.listdir(ORAL_CANCER_DIR))[:4]:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples["oral"].append({
                    "filename": f,
                    "category": "oral-cancer",
                    "label": f"Oral Cancer Photo ({f})",
                    "modality": "Clinical Oral Image",
                    "url": f"/api/sample-image/oral-cancer/{f}",
                })

    # Oral Non-Cancer samples
    if os.path.exists(ORAL_NON_CANCER_DIR):
        for f in sorted(os.listdir(ORAL_NON_CANCER_DIR))[:4]:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples["oral"].append({
                    "filename": f,
                    "category": "oral-non-cancer",
                    "label": f"Benign Oral Photo ({f})",
                    "modality": "Clinical Oral Image",
                    "url": f"/api/sample-image/oral-non-cancer/{f}",
                })

    # Histopathology Biopsy samples (NDU-UFES)
    if os.path.exists(HISTO_DIR):
        # 0000.png is OSCC, 0008.png is Leukoplakia/Dysplasia
        histo_picks = ["0000.png", "0001.png", "0008.png", "0016.png"]
        for f in histo_picks:
            if os.path.exists(os.path.join(HISTO_DIR, f)):
                samples["histopathology"].append({
                    "filename": f,
                    "category": "histo",
                    "label": f"Biopsy Slide ({f})",
                    "modality": "Histopathological Image",
                    "url": f"/api/sample-image/histo/{f}",
                })

    return samples


@app.get("/api/sample-image/{category}/{filename}")
def get_sample_file(category: str, filename: str):
    safe_name = os.path.basename(filename)
    if category == "oral-cancer":
        dir_path = ORAL_CANCER_DIR
    elif category == "oral-non-cancer":
        dir_path = ORAL_NON_CANCER_DIR
    elif category == "histo":
        dir_path = HISTO_DIR
    else:
        raise HTTPException(status_code=400, detail="Invalid sample category")

    file_path = os.path.join(dir_path, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Sample file not found")

    ext = os.path.splitext(safe_name)[1].lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    return FileResponse(file_path, media_type=media_type)


@app.post("/api/predict-multimodal")
async def predict_multimodal(
    clinical_image: Optional[UploadFile] = File(None, description="Clinical oral lesion photograph"),
    histopathology_image: Optional[UploadFile] = File(None, description="Histopathological biopsy slide"),
    clinical_data: Optional[str] = Form(None, description="JSON string of patient clinical parameters"),
):
    # Verify at least one modality is provided
    has_oral = clinical_image is not None and clinical_image.filename
    has_histo = histopathology_image is not None and histopathology_image.filename
    has_clinical = clinical_data is not None and clinical_data.strip() != ""

    if not (has_oral or has_histo or has_clinical):
        raise HTTPException(
            status_code=400,
            detail="At least one modality (Clinical Oral Image, Histopathological Biopsy Image, or Clinical Data) must be provided for evaluation.",
        )

    oral_result = None
    histo_result = None
    clinical_result = None
    clinical_input_parsed = None

    # 1. Process Oral Photograph
    if has_oral:
        if clinical_image.content_type and not clinical_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Clinical Oral Image must be a valid image file.")
        try:
            bytes_data = await clinical_image.read()
            pil_img = Image.open(io.BytesIO(bytes_data))
            oral_result = multimodal_service.predict_oral_image(pil_img)
            oral_result["filename"] = clinical_image.filename
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode Clinical Oral Image: {str(e)}")

    # 2. Process Histopathology Biopsy
    if has_histo:
        if histopathology_image.content_type and not histopathology_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Histopathological Image must be a valid image file.")
        try:
            bytes_data = await histopathology_image.read()
            pil_img = Image.open(io.BytesIO(bytes_data))
            histo_result = multimodal_service.predict_histopathology_image(pil_img)
            histo_result["filename"] = histopathology_image.filename
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode Histopathological Image: {str(e)}")

    # 3. Process Clinical Patient Data
    if has_clinical:
        try:
            clinical_input_parsed = json.loads(clinical_data)
            clinical_result = multimodal_service.predict_clinical_data(clinical_input_parsed)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid clinical data JSON payload: {str(e)}")

    # 4. Perform Validated Multimodal Late Fusion
    fused_synthesis = multimodal_service.fuse_multimodal(
        oral_res=oral_result,
        histo_res=histo_result,
        clinical_res=clinical_result,
    )

    # 5. Generate Structured Report
    case_id = f"OC-AI-{uuid.uuid4().hex[:8].upper()}"
    report_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    structured_report = {
        "report_id": case_id,
        "title": "Multimodal Oral Cancer Detection Report",
        "generated_at": report_timestamp,
        "modalities_summary": {
            "total_evaluated": fused_synthesis["active_modalities_count"],
            "total_possible": 3,
            "clinical_oral_image_evaluated": has_oral,
            "histopathological_image_evaluated": has_histo,
            "clinical_data_evaluated": has_clinical,
        },
        "patient_clinical_data": clinical_input_parsed or "No clinical metadata provided",
        "multimodal_synthesis": {
            "fused_prediction": fused_synthesis["fused_prediction"],
            "cancer_probability": fused_synthesis["cancer_probability"],
            "confidence": fused_synthesis["confidence_percentage"],
            "uncertainty_level": fused_synthesis["uncertainty_level"],
            "fusion_method": fused_synthesis["fusion_method"],
            "modality_contributions": fused_synthesis["modality_contributions"],
            "discordance_alert": fused_synthesis["discordance_alert"],
        },
        "individual_modality_findings": {
            "oral_photography": {
                "evaluated": has_oral,
                "output": oral_result["predicted_class"] if oral_result else "Not provided",
                "probability": oral_result["cancer_probability"] if oral_result else None,
                "confidence": oral_result["confidence_percentage"] if oral_result else None,
                "summary": oral_result["findings_summary"] if oral_result else None,
            },
            "histopathology_biopsy": {
                "evaluated": has_histo,
                "output": histo_result["predicted_class"] if histo_result else "Not provided",
                "probability": histo_result["cancer_probability"] if histo_result else None,
                "confidence": histo_result["confidence_percentage"] if histo_result else None,
                "summary": histo_result["findings_summary"] if histo_result else None,
            },
            "clinical_data": {
                "evaluated": has_clinical,
                "output": clinical_result["predicted_class"] if clinical_result else "Not provided",
                "probability": clinical_result["cancer_probability"] if clinical_result else None,
                "identified_risk_factors": clinical_result["identified_risk_factors"] if clinical_result else [],
                "summary": clinical_result["findings_summary"] if clinical_result else None,
            },
        },
        "scientific_limitations": [
            "Research Prototype: This system is designed for academic and educational validation only.",
            "Cohort Independence: Oral lesion photography and biopsy histology were trained on separate research cohorts; multimodal consensus is synthesized via calibrated decision-level Bayesian late fusion.",
            "Non-Diagnostic: Computational predictions must not replace clinical biopsy examination or histopathologist diagnosis.",
        ],
        "clinical_recommendation": "This AI result is not a medical diagnosis. Please consult a qualified healthcare professional or oral maxillofacial pathologist for formal clinical evaluation and confirmatory biopsy.",
    }

    # Return full JSON
    return JSONResponse(content={
        "success": True,
        "case_id": case_id,
        "fused_result": fused_synthesis,
        "modalities": {
            "oral": oral_result,
            "histopathology": histo_result,
            "clinical": clinical_result,
        },
        "structured_report": structured_report,
    })


# ============================================================
# EXPERIMENTATION & RESULTS APIS
# ============================================================

@app.get("/api/experiments")
def get_experiments():
    """
    Returns all authentic recorded training experiments from experiments.csv.
    """
    exps = experiment_manager.get_all_experiments()
    return JSONResponse(content={"experiments": exps, "count": len(exps)})


@app.get("/api/experiments/summary")
def get_experiments_summary():
    """
    Returns overall summary including best-performing model dynamically calculated from test results.
    """
    summary = experiment_manager.get_summary_metrics()
    return JSONResponse(content=summary)


@app.get("/api/experiments/models")
def get_models_info():
    """
    Returns available deep learning architectures and baseline specifications.
    """
    return JSONResponse(content={
        "models": [
            {
                "id": "CustomCNN",
                "name": "Custom CNN Baseline",
                "family": "Convolutional Neural Network",
                "parameters": "468.3K",
                "trainable_parameters": 468322,
                "description": "4-stage baseline CNN with BatchNorm, MaxPool, Dropout, and Global Average Pooling."
            },
            {
                "id": "ResNet18",
                "name": "ResNet-18",
                "family": "Residual Networks (ResNet)",
                "parameters": "11.18M",
                "trainable_parameters": 11177538,
                "description": "Lightweight residual learning with shortcut connections, pretrained on ImageNet."
            },
            {
                "id": "ResNet50",
                "name": "ResNet-50",
                "family": "Residual Networks (ResNet)",
                "parameters": "23.51M",
                "trainable_parameters": 23512130,
                "description": "Deeper 50-layer bottleneck residual architecture for high-capacity representation."
            },
            {
                "id": "EfficientNetB0",
                "name": "EfficientNet-B0",
                "family": "Compound Scaling",
                "parameters": "4.01M",
                "trainable_parameters": 4010110,
                "description": "Uniformly balanced depth, width, and resolution scaling for optimal computational efficiency."
            },
            {
                "id": "DenseNet121",
                "name": "DenseNet-121",
                "family": "Densely Connected Networks",
                "parameters": "6.96M",
                "trainable_parameters": 6955906,
                "description": "Connects all subsequent layers directly, alleviating vanishing gradients and encouraging feature reuse."
            }
        ]
    })


@app.get("/api/experiments/metrics")
def get_experiments_metrics():
    """
    Returns comparison metrics grouped by model, epochs, batch size, and learning rate.
    """
    metrics = experiment_manager.get_comparison_metrics()
    return JSONResponse(content=metrics)


@app.get("/api/experiments/history/{experiment_id}")
def get_experiment_history(experiment_id: str):
    """
    Returns epoch-by-epoch loss and accuracy history for a specific experiment.
    """
    history_data = experiment_manager.get_history(experiment_id)
    if not history_data:
        raise HTTPException(status_code=404, detail=f"History for experiment {experiment_id} not found")
    return JSONResponse(content=history_data)


@app.get("/api/experiments/confusion-matrix/{experiment_id}")
def get_experiment_confusion_matrix(experiment_id: str):
    """
    Returns confusion matrix data for a specific experiment.
    """
    cm_data = experiment_manager.get_confusion_matrix(experiment_id)
    if not cm_data:
        raise HTTPException(status_code=404, detail=f"Confusion matrix for experiment {experiment_id} not found")
    return JSONResponse(content=cm_data)


@app.get("/api/experiments/datasets")
def get_datasets_statistics():
    """
    Returns real statistics for both Clinical Oral Photography and NDU-UFES Histopathology datasets.
    """
    try:
        stats_oral = dataset_manager.get_dataset_statistics("clinical_oral")
    except Exception as e:
        stats_oral = {"error": str(e)}

    try:
        stats_histo = dataset_manager.get_dataset_statistics("ndu_ufes_histo")
    except Exception as e:
        stats_histo = {"error": str(e)}

    return JSONResponse(content={
        "clinical_oral": stats_oral,
        "ndu_ufes_histo": stats_histo,
        "additional_datasets_recommended": [
            {
                "name": "Histopathological Imaging Database (Rahman et al.)",
                "role": "External Validation / Pre-training",
                "samples": 1224,
                "modality": "Histopathology H&E",
                "status": "Ready for drop-in placement at datasets/external/histopathology_external"
            },
            {
                "name": "Annotated Oral Cavity Images (AIIMS / Lin et al.)",
                "role": "Clinical Photography External Validation",
                "samples": 3000,
                "modality": "Intraoral Photography",
                "status": "Ready for drop-in placement at datasets/external/clinical_external"
            }
        ]
    })

