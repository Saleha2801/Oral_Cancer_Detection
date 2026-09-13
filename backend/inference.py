import io
import os
import base64
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.cm as cm

# Constants matching training pipelines
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class GradCAMExplainer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam).detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam

    def cleanup(self):
        try:
            self.fwd_hook.remove()
            self.bwd_hook.remove()
        except Exception:
            pass


class MultimodalInferenceService:
    def __init__(self):
        self.device = device
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(self.base_dir, ".."))
        self.models_dir = os.path.join(self.root_dir, "models")

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Model containers
        self.oral_model = None
        self.histo_model = None
        self.clinical_artifact = None

        self._load_all_models()

    def _load_all_models(self):
        # 1. Load Oral ResNet-18 Model
        oral_candidates = [
            os.path.join(self.models_dir, "best_oral_cancer_resnet18.pth"),
            os.path.join(self.root_dir, "best_oral_cancer_resnet18.pth"),
        ]
        oral_path = next((p for p in oral_candidates if os.path.exists(p)), None)
        if oral_path:
            try:
                print(f"[MultimodalService] Loading Oral Model from {oral_path} on {self.device}")
                m = models.resnet18(weights=None)
                m.fc = nn.Linear(m.fc.in_features, 2)
                state = torch.load(oral_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                m.load_state_dict(state)
                m = m.to(self.device)
                m.eval()
                self.oral_model = m
                print("[MultimodalService] Oral Model loaded successfully.")
            except Exception as e:
                print(f"[MultimodalService] Error loading oral model: {e}")

        # 2. Load Histopathology ResNet-18 Model
        histo_candidates = [
            os.path.join(self.models_dir, "best_histopathology_model.pth"),
            os.path.join(self.root_dir, "best_histopathology_model.pth"),
        ]
        histo_path = next((p for p in histo_candidates if os.path.exists(p)), None)
        if histo_path:
            try:
                print(f"[MultimodalService] Loading Histopathology Model from {histo_path} on {self.device}")
                m = models.resnet18(weights=None)
                m.fc = nn.Linear(m.fc.in_features, 2)
                state = torch.load(histo_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                m.load_state_dict(state)
                m = m.to(self.device)
                m.eval()
                self.histo_model = m
                print("[MultimodalService] Histopathology Model loaded successfully.")
            except Exception as e:
                print(f"[MultimodalService] Error loading histopathology model: {e}")

        # 3. Load Clinical Random Forest Model
        clinical_path = os.path.join(self.models_dir, "clinical_model.joblib")
        if os.path.exists(clinical_path):
            try:
                print(f"[MultimodalService] Loading Clinical Model from {clinical_path}")
                self.clinical_artifact = joblib.load(clinical_path)
                print(f"[MultimodalService] Clinical Model loaded. Test Acc: {self.clinical_artifact.get('test_accuracy', 0):.2%}")
            except Exception as e:
                print(f"[MultimodalService] Error loading clinical model: {e}")

    def get_system_status(self):
        return {
            "oral_model_loaded": self.oral_model is not None,
            "histo_model_loaded": self.histo_model is not None,
            "clinical_model_loaded": self.clinical_artifact is not None,
            "device": str(self.device),
            "modalities_available": [
                m for m, loaded in [
                    ("clinical_oral_image", self.oral_model is not None),
                    ("histopathological_image", self.histo_model is not None),
                    ("clinical_patient_data", self.clinical_artifact is not None),
                ] if loaded
            ],
            "fusion_method": "Calibrated Decision-Level Evidence-Weighted Late Fusion",
            "pairing_disclosure": "Oral photography and NDU-UFES biopsy data come from separate patient cohorts. Decision-level fusion synthesizes independent modality evidence.",
        }

    def _render_cam_b64(self, original_img: Image.Image, cam: np.ndarray):
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
        cam_norm = np.array(cam_img).astype(np.float32) / 255.0

        colormap = matplotlib.colormaps["jet"]
        heatmap_rgba = colormap(cam_norm)
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

        orig_resized = np.array(original_img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)).astype(np.float32)
        alpha = 0.45
        overlay = (orig_resized * (1.0 - alpha) + heatmap_rgb.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

        buf_hm = io.BytesIO()
        Image.fromarray(heatmap_rgb).save(buf_hm, format="PNG")
        b64_hm = base64.b64encode(buf_hm.getvalue()).decode("utf-8")

        buf_ol = io.BytesIO()
        Image.fromarray(overlay).save(buf_ol, format="PNG")
        b64_ol = base64.b64encode(buf_ol.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{b64_hm}", f"data:image/png;base64,{b64_ol}"

    def predict_oral_image(self, pil_image: Image.Image):
        if self.oral_model is None:
            raise RuntimeError("Oral ResNet-18 model weights not loaded.")

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.oral_model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        pred_idx = int(torch.argmax(probs).item())
        # 0: Cancer, 1: Non-cancer
        cancer_prob = float(probs[0].item())
        non_cancer_prob = float(probs[1].item())
        confidence = float(probs[pred_idx].item())
        pred_label = "Cancer" if pred_idx == 0 else "Non-cancer"

        # Grad-CAM
        heatmap_b64, overlay_b64 = None, None
        try:
            target_layer = self.oral_model.layer4[-1].conv2
            explainer = GradCAMExplainer(self.oral_model, target_layer)
            cam = explainer.generate(input_tensor, class_idx=pred_idx)
            explainer.cleanup()
            heatmap_b64, overlay_b64 = self._render_cam_b64(pil_image, cam)
        except Exception as e:
            print(f"[MultimodalService] Oral GradCAM error: {e}")

        return {
            "modality": "Clinical Oral Image",
            "model_architecture": "ResNet-18",
            "predicted_class": pred_label,
            "is_malignant": pred_idx == 0,
            "cancer_probability": round(cancer_prob, 4),
            "non_cancer_probability": round(non_cancer_prob, 4),
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "gradcam_heatmap": heatmap_b64,
            "gradcam_overlay": overlay_b64,
            "findings_summary": f"Oral lesion visual features demonstrate a {cancer_prob * 100:.1f}% likelihood of malignancy (predicted: {pred_label}).",
        }

    def predict_histopathology_image(self, pil_image: Image.Image):
        if self.histo_model is None:
            raise RuntimeError("Histopathology ResNet-18 model weights not loaded.")

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.histo_model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        pred_idx = int(torch.argmax(probs).item())
        # 0: OSCC (Malignant), 1: Leukoplakia (Non-cancer)
        oscc_prob = float(probs[0].item())
        leuko_prob = float(probs[1].item())
        confidence = float(probs[pred_idx].item())
        pred_label = "OSCC (Malignant)" if pred_idx == 0 else "Leukoplakia (Non-cancer)"

        # Grad-CAM
        heatmap_b64, overlay_b64 = None, None
        try:
            target_layer = self.histo_model.layer4[-1].conv2
            explainer = GradCAMExplainer(self.histo_model, target_layer)
            cam = explainer.generate(input_tensor, class_idx=pred_idx)
            explainer.cleanup()
            heatmap_b64, overlay_b64 = self._render_cam_b64(pil_image, cam)
        except Exception as e:
            print(f"[MultimodalService] Histo GradCAM error: {e}")

        return {
            "modality": "Histopathological Image",
            "model_architecture": "ResNet-18 (NDU-UFES Trained)",
            "predicted_class": pred_label,
            "is_malignant": pred_idx == 0,
            "cancer_probability": round(oscc_prob, 4),
            "non_cancer_probability": round(leuko_prob, 4),
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "gradcam_heatmap": heatmap_b64,
            "gradcam_overlay": overlay_b64,
            "findings_summary": f"Microscopic tissue architecture reveals cellular patterns consistent with {pred_label} (Malignancy probability: {oscc_prob * 100:.1f}%).",
        }

    def predict_clinical_data(self, clinical_input: dict):
        if self.clinical_artifact is None:
            raise RuntimeError("Clinical model artifact not loaded.")

        rf = self.clinical_artifact["model"]
        feature_cols = self.clinical_artifact["feature_cols"]

        # Construct one-hot feature vector matching ndbufes_TaskII
        row = {col: 0 for col in feature_cols}

        # 1. Localization
        loc = clinical_input.get("localization", "Tongue")
        loc_key = f"localization_{loc}"
        if loc_key in row:
            row[loc_key] = 1

        # 2. Larger size
        try:
            size_val = float(clinical_input.get("larger_size", 1.5))
        except (ValueError, TypeError):
            size_val = 1.5
        row["larger_size"] = size_val

        # 3. Tobacco use
        tob = clinical_input.get("tobacco_use", "Not informed")
        tob_key = f"tobacco_use_{tob}"
        if tob_key in row:
            row[tob_key] = 1

        # 4. Alcohol consumption
        alc = clinical_input.get("alcohol_consumption", "Not informed")
        alc_key = f"alcohol_consumption_{alc}"
        if alc_key in row:
            row[alc_key] = 1

        # 5. Sun exposure
        sun = clinical_input.get("sun_exposure", "Not informed")
        sun_key = f"sun_exposure_{sun}"
        if sun_key in row:
            row[sun_key] = 1

        # 6. Gender
        gen = clinical_input.get("gender", "M")
        gen_key = f"gender_{gen}"
        if gen_key in row:
            row[gen_key] = 1

        # 7. Age group (0: <40, 1: 40-60, 2: >60)
        age = str(clinical_input.get("age_group", "1"))
        age_key = f"age_group_{age}"
        if age_key in row:
            row[age_key] = 1

        # Convert to DataFrame
        X = pd.DataFrame([row])[feature_cols]
        probs = rf.predict_proba(X)[0]

        # In our training: 0 = Cancer (OSCC), 1 = Leukoplakia
        cancer_prob = float(probs[0])
        non_cancer_prob = float(probs[1])
        pred_idx = 0 if cancer_prob >= 0.5 else 1
        confidence = cancer_prob if pred_idx == 0 else non_cancer_prob
        pred_label = "High Clinical Risk (OSCC Profile)" if pred_idx == 0 else "Low/Moderate Clinical Risk"

        # Risk factor identification
        risk_factors = []
        if tob == "Yes":
            risk_factors.append("Active tobacco consumption")
        if alc == "Yes":
            risk_factors.append("Active alcohol consumption (synergistic carcinogen)")
        if loc in ["Tongue", "Floor of mouth"]:
            risk_factors.append(f"High-risk anatomical site ({loc})")
        if size_val >= 2.0:
            risk_factors.append(f"Lesion dimensions >= 2.0 cm ({size_val} cm)")
        if age == "2":
            risk_factors.append("Demographic age group > 60 years")

        return {
            "modality": "Clinical Patient Information",
            "model_architecture": "Random Forest Classifier (NDU-UFES Clinical)",
            "predicted_class": pred_label,
            "is_malignant": pred_idx == 0,
            "cancer_probability": round(cancer_prob, 4),
            "non_cancer_probability": round(non_cancer_prob, 4),
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "identified_risk_factors": risk_factors,
            "input_features": clinical_input,
            "findings_summary": f"Epidemiological profile evaluated at {cancer_prob * 100:.1f}% risk for oral malignancy.",
        }

    def fuse_multimodal(self, oral_res=None, histo_res=None, clinical_res=None):
        """
        Decision-level evidence-weighted Bayesian late fusion.
        Base weights:
          - Histopathology: 0.45 (biopsy gold standard)
          - Oral Photography: 0.35 (macroscopic lesion photo)
          - Clinical Patient Data: 0.20 (epidemiological profile)
        Dynamically renormalizes when modalities are absent.
        """
        active_modalities = []
        raw_weights = {}

        if histo_res is not None:
            active_modalities.append("Histopathological Image")
            raw_weights["histo"] = 0.45

        if oral_res is not None:
            active_modalities.append("Clinical Oral Image")
            raw_weights["oral"] = 0.35

        if clinical_res is not None:
            active_modalities.append("Clinical Patient Data")
            raw_weights["clinical"] = 0.20

        if not active_modalities:
            raise ValueError("Cannot perform fusion: No modality results provided.")

        # Re-normalize weights
        total_w = sum(raw_weights.values())
        norm_weights = {k: v / total_w for k, v in raw_weights.items()}

        # Compute weighted cancer probability
        fused_cancer_prob = 0.0
        modality_contributions = {}

        if histo_res is not None:
            w = norm_weights["histo"]
            p = histo_res["cancer_probability"]
            fused_cancer_prob += w * p
            modality_contributions["Histopathological Image"] = {
                "weight": round(w, 3),
                "probability": p,
                "weighted_contribution": round(w * p, 4),
            }

        if oral_res is not None:
            w = norm_weights["oral"]
            p = oral_res["cancer_probability"]
            fused_cancer_prob += w * p
            modality_contributions["Clinical Oral Image"] = {
                "weight": round(w, 3),
                "probability": p,
                "weighted_contribution": round(w * p, 4),
            }

        if clinical_res is not None:
            w = norm_weights["clinical"]
            p = clinical_res["cancer_probability"]
            fused_cancer_prob += w * p
            modality_contributions["Clinical Patient Data"] = {
                "weight": round(w, 3),
                "probability": p,
                "weighted_contribution": round(w * p, 4),
            }

        fused_non_cancer_prob = 1.0 - fused_cancer_prob
        is_malignant = fused_cancer_prob >= 0.5
        predicted_class = "Malignant (Cancer / OSCC)" if is_malignant else "Benign / Low-Risk (Non-cancer)"
        confidence = fused_cancer_prob if is_malignant else fused_non_cancer_prob

        # Uncertainty level
        uncertainty_gap = abs(fused_cancer_prob - 0.5)
        if uncertainty_gap < 0.15:
            uncertainty_level = "High Uncertainty (Borderline Consensus)"
        elif uncertainty_gap < 0.30:
            uncertainty_level = "Moderate Uncertainty"
        else:
            uncertainty_level = "Low Uncertainty (High Modality Agreement)"

        # Discordance check (e.g. Oral vs Histo disagree)
        discordance_alert = None
        if oral_res is not None and histo_res is not None:
            if oral_res["is_malignant"] != histo_res["is_malignant"]:
                discordance_alert = (
                    f"Modality Discordance Alert: Clinical oral photo predicted '{oral_res['predicted_class']}' "
                    f"whereas biopsy histopathology predicted '{histo_res['predicted_class']}'. "
                    "Under clinical oncological protocols, tissue biopsy histology holds clinical precedence."
                )

        return {
            "fused_prediction": predicted_class,
            "is_malignant": is_malignant,
            "cancer_probability": round(fused_cancer_prob, 4),
            "non_cancer_probability": round(fused_non_cancer_prob, 4),
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.1f}%",
            "active_modalities_count": len(active_modalities),
            "total_modalities_possible": 3,
            "active_modalities": active_modalities,
            "modality_contributions": modality_contributions,
            "uncertainty_level": uncertainty_level,
            "discordance_alert": discordance_alert,
            "fusion_method": "Calibrated Decision-Level Bayesian Late Fusion",
        }

# Global singleton service
multimodal_service = MultimodalInferenceService()
