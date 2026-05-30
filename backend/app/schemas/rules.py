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


class CustomerCardLookupResponse(BaseModel):
    customer_hash: str
    masked_card: Optional[str] = None
    last4: Optional[str] = None
    available: bool = False


class SummaryTransactionItem(BaseModel):
    transaction_id: Optional[str] = None
    transaction_datetime: Optional[str] = None
    amount: Optional[float] = None
    country_code: Optional[str] = None
    pos_entry_mode: Optional[str] = None
    merchant_rubro_proxy: Optional[str] = None
    merchant_name: Optional[str] = None
    has_pinblock: Optional[int] = None
    risk_score: Optional[float] = None
    customer_hash: Optional[str] = None
    masked_card: Optional[str] = None


class SummaryTransactionsResponse(BaseModel):
    run_id: str
    alert_id: str
    total_transactions: int
    items: List[SummaryTransactionItem]
    warning: Optional[str] = None


# ============================================================
# Alert Review Schemas (PHASE B.3)
# ============================================================


class AlertStatusUpdateRequest(BaseModel):
    """Request to update alert status."""
    run_id: str = Field(..., description="Source run ID")
    new_status: str = Field(
        ...,
        description="New status: NEW, IN_REVIEW, DISMISSED, FALSE_POSITIVE, CONFIRMED_FRAUD"
    )
    analyst_notes: Optional[str] = Field(None, description="Optional notes from analyst")
    reviewed_by: Optional[str] = Field(None, description="Analyst email/identifier")


class AlertStatusUpdateResponse(BaseModel):
    """Response after updating alert status."""
    status: str = Field("OK", description="Operation status")
    alert_id: Optional[str] = None
    summary_alert_id: Optional[str] = None
    run_id: str
    new_status: str
    reviewed_at: str
    message: Optional[str] = None


class AlertReviewHistoryItem(BaseModel):
    """Single review history entry."""
    id: int
    source_run: str
    alert_id: Optional[str] = None
    summary_alert_id: Optional[str] = None
    rule_code: str
    previous_status: Optional[str] = None
    new_status: str
    analyst_notes: Optional[str] = None
    reviewed_by_id: Optional[int] = None
    reviewed_at: str


class AlertReviewHistoryResponse(BaseModel):
    """Response with review history for an alert."""
    alert_id: Optional[str] = None
    summary_alert_id: Optional[str] = None
    run_id: str
    history: List[AlertReviewHistoryItem]


class PaginatedReviewsResponse(BaseModel):
    """Paginated list of reviews."""
    run_id: str
    page: int
    page_size: int
    total_items: int
    total_pages: int
    items: List[AlertReviewHistoryItem]