"""Pydantic request/response models for the pipeline API."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    max_iterations: int = Field(default=3, ge=1, le=5)
    max_retries_per_phase: int = Field(default=2, ge=0, le=3)


class RunResponse(BaseModel):
    thread_id: str


class TokenSummary(BaseModel):
    by_agent: dict[str, dict[str, int]] = {}
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class PipelineResult(BaseModel):
    thread_id: str
    status: str  # "complete" | "rejected" | "error" | "pending"
    draft_report: Optional[str] = None
    citations: list[str] = []
    quality_score: float = 0.0
    is_approved: bool = False
    out_of_scope: bool = False
    scope_rejection_reason: str = ""
    clarification_needed: bool = False
    clarification_question: str = ""
    surface_error: bool = False
    pipeline_error_message: str = ""
    errors: list[str] = []
    token_summary: Optional[TokenSummary] = None
    key_findings: list[str] = []
    evidence_quality: str = ""
    evidence_grade: str = ""
