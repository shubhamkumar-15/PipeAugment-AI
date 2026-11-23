import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.core.config import Settings, get_settings

router = APIRouter(prefix="/evaluation", tags=["model-evaluation"])

ALLOWLISTED_CHARTS = {
    "abc_clean_vs_stress_f1.png": "charts/abc_clean_vs_stress_f1.png",
    "abc_corrosion_recall_robustness.png": "charts/abc_corrosion_recall_robustness.png",
    "abc_confusion_matrices_clean.png": "charts/abc_confusion_matrices_clean.png",
    "stress_suite_preview.png": "stress_suite_preview.png",
}


@router.get(
    "/benchmark",
    summary="Get multi-seed final benchmark evaluation results and robustness stress metrics",
)
def get_benchmark_results(settings: Settings = Depends(get_settings)):
    """
    Returns the real, multi-seed final evaluation benchmark results.
    Includes clean held-out test set metrics (Mean ± Std) and controlled
    synthetic stress perturbation metrics across Baseline A, Experiment B, and Experiment C.
    """
    json_path = settings.FINAL_EVAL_DIR / "final_benchmark_results.json"
    if not json_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark evaluation results not found. Has Phase 7 benchmark been executed?",
        )

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read benchmark results: {str(exc)}",
        ) from exc


@router.get(
    "/charts/{chart_name}",
    summary="Retrieve publication-grade benchmark visualization charts",
)
def get_evaluation_chart(
    chart_name: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serves allowlisted evaluation chart images directly from disk with path traversal protection."""
    if chart_name not in ALLOWLISTED_CHARTS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chart '{chart_name}' not found. Allowlisted charts: {list(ALLOWLISTED_CHARTS.keys())}",
        )

    rel_subpath = ALLOWLISTED_CHARTS[chart_name]
    chart_path = (settings.FINAL_EVAL_DIR / rel_subpath).resolve()
    base_dir = settings.FINAL_EVAL_DIR.resolve()

    try:
        if not chart_path.is_relative_to(base_dir):
            raise HTTPException(status_code=400, detail="Invalid chart path.")
    except AttributeError:
        if base_dir not in chart_path.parents:
            raise HTTPException(status_code=400, detail="Invalid chart path.")

    if not chart_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chart file '{chart_name}' missing on disk.",
        )

    return FileResponse(
        path=str(chart_path),
        media_type="image/png",
        filename=chart_name,
    )
