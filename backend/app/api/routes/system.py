import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings

router = APIRouter(prefix="/system", tags=["system-information"])


def query_gpu_info() -> Dict[str, Any]:
    """Queries GPU telemetry via nvidia-smi with graceful fallback if unavailable."""
    if not shutil.which("nvidia-smi"):
        return {
            "gpu_detected": False,
            "error": "nvidia-smi binary not in system PATH",
        }

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 4:
                return {
                    "gpu_detected": True,
                    "model_name": parts[0],
                    "total_memory_mb": int(parts[1]),
                    "free_memory_mb": int(parts[2]),
                    "driver_version": parts[3],
                }
    except Exception as exc:
        return {
            "gpu_detected": False,
            "error": str(exc),
        }

    return {
        "gpu_detected": False,
        "error": "No GPU devices reported by nvidia-smi",
    }


@router.get(
    "/info",
    summary="Get real system environment, hardware telemetry, and augmentation capabilities",
)
def get_system_info(settings: Settings = Depends(get_settings)):
    """Returns genuine hardware status, AI worker status, and capability descriptors."""
    gpu_telemetry = query_gpu_info()
    ai_environment_ready = settings.AI_PYTHON_PATH.exists()

    return {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "hardware": {
            "gpu": gpu_telemetry,
        },
        "ai_worker": {
            "environment_ready": ai_environment_ready,
            "python_executable": str(settings.AI_PYTHON_PATH),
        },
        "capabilities": {
            "augmentation_methods": ["traditional", "generative"],
            "traditional_transforms": [
                "brightness",
                "contrast",
                "gamma",
                "uneven_illumination",
                "soft_shadow",
                "gaussian_blur",
                "gaussian_noise",
                "jpeg_compression",
                "downsample_restore",
            ],
            "generative_conditions": [
                "low_light_inspection",
                "directional_industrial_lighting",
                "bright_daylight",
                "harsh_shadows",
                "wet_surface",
                "dusty_surface",
            ],
            "frozen_controlnet_config": {
                "base_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "controlnet_model": "lllyasviel/control_v11p_sd15_canny",
                "control_scale": 0.80,
                "strength": 0.35,
                "guidance_scale": 5.0,
                "num_inference_steps": 30,
                "canny_thresholds": [100, 200],
            },
        },
        "upload_limits": {
            "max_upload_bytes": settings.MAX_UPLOAD_BYTES,
            "max_upload_mb": settings.MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_image_pixels": settings.MAX_IMAGE_PIXELS,
            "supported_formats": ["JPEG", "PNG", "WEBP"],
        },
    }
