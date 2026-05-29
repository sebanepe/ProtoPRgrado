from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FlexibleItem(BaseModel):
    class Config:
        extra = "allow"


class RuleAnalyzeRequest(BaseModel):
    preprocessed_run_id: str = Field(..., description="Example: preprocessed_run_26")
    force: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)


class RuleAnalyzeResponse(BaseModel):
    status: str
    source_run: str
    total_transactions: int
    alerts_file: str
    summary_file: str
    report_file: str
    total_alerts: int
    total_summary_alerts: int
    message: str


class RunListItem(BaseModel):
    run_id: str
    filename: str
    path: str
    created_at: str
    size_bytes: int
    has_alerts: bool
    has_summary: bool
    has_report: bool
    alerts_file: Optional[str] = None
    summary_file: Optional[str] = None
    report_file: Optional[str] = None


class AlertItem(FlexibleItem):
    pass


class AlertSummaryItem(FlexibleItem):
    pass


class PaginatedAlertsResponse(BaseModel):
    run_id: str
    page: int
    page_size: int
    total_items: int
    total_pages: int
    items: List[AlertItem]


class PaginatedAlertSummaryResponse(BaseModel):
    run_id: str
    page: int
    page_size: int
    total_items: int
    total_pages: int
    items: List[AlertSummaryItem]


class RuleMetricsResponse(BaseModel):
    run_id: str
    total_alerts: int
    total_summary_alerts: int
    alerts_by_rule: Dict[str, int]
    alerts_by_risk_level: Dict[str, int]
    alerts_by_mcc: Dict[str, int]
    alerts_by_country: Dict[str, int]
    top_customers: List[Dict[str, Any]]