"""
Prompt Templates for PipeAugment AI Pretrained Diffusion Pilot.
Focuses strictly on environmental lighting variations while preserving pipeline surface
and structural features across both classes neutrally.

NOTE: Prompts and negative prompts are requests to the diffusion model,
not physical or mathematical guarantees of geometry or corrosion preservation.
"""

from typing import Dict, Any


BASE_INTENT = (
    "Photorealistic photograph of the same industrial pipeline, "
    "same viewpoint, same pipe shape, same joints and background, "
    "retaining existing surface texture, rust locations and surface defects"
)

CONDITIONS: Dict[str, Dict[str, str]] = {
    "low_light_inspection": {
        "display_name": "Low-Light Inspection",
        "description": "Dim, underexposed ambient inspection lighting with subtle shadows.",
        "prompt_extension": "dark subterranean tunnel ambiance, dim cool work-lamp falloff, deep ambient shadows, crisp metallic surface detail",
    },
    "directional_industrial_lighting": {
        "display_name": "Directional Industrial Lighting",
        "description": "Harsh, focused industrial inspection work-light casting directional shadows.",
        "prompt_extension": "strong oblique industrial spotlight from one side, bright specular steel rim highlights, crisp cast shadows",
    },
    "bright_daylight": {
        "display_name": "Bright Daylight",
        "description": "Even, bright natural daylight illumination across the pipeline structure.",
        "prompt_extension": "harsh direct midday sunlight, crisp bright metallic reflections, high clarity outdoor sky illumination",
    },
    "harsh_shadows": {
        "display_name": "Harsh Directional Shadows",
        "description": "High contrast illumination casting deep structural shadows across joints and seams.",
        "prompt_extension": "dramatic high-contrast diagonal beam shadows cast across the pipe body from overhead structural grating",
    },
    "wet_surface": {
        "display_name": "Wet Industrial Surface",
        "description": "Damp pipeline surface with subtle water sheen and specular surface reflections.",
        "prompt_extension": "glistening wet steel surface, clear water sheen, damp moisture streaks and sharp specular water droplets reflecting light",
    },
    "dusty_surface": {
        "display_name": "Dusty Pipeline Environment",
        "description": "Dry industrial dust haze with fine airborne particulate and subtle surface dust.",
        "prompt_extension": "matte industrial dust haze, fine dry mineral powder and grime coating over the pipe surface, muted specular glare",
    },
}

NEGATIVE_PROMPT = (
    "changed pipe geometry, added or missing pipes, changed joints, altered structural features, "
    "painted-over surfaces, invented surface defects, removed corrosion, added corrosion, "
    "text, watermarks, signature, logo, illustration, rendering, 3d render, cartoon, blurry, "
    "heavy blur, out of focus, distorted, artifact"
)


def get_prompt_for_condition(condition_name: str) -> str:
    """
    Constructs the full positive prompt by appending the environmental condition
    to the neutral base preservation intent.
    """
    if condition_name not in CONDITIONS:
        raise ValueError(f"Unknown condition '{condition_name}'. Supported: {list(CONDITIONS.keys())}")

    condition_ext = CONDITIONS[condition_name]["prompt_extension"]
    return f"{BASE_INTENT}, {condition_ext}."


def get_negative_prompt() -> str:
    """Returns the standardized negative prompt."""
    return NEGATIVE_PROMPT
