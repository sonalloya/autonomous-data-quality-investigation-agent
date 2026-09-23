"""
tests/test_phase11_frontend.py
-------------------------------
Phase 11 — Web Frontend & Static Asset Serving Test Suite.

Verifies:
  1. Root endpoint GET / serves the HTML5 dashboard.
  2. Static CSS assets are served correctly with proper MIME types.
  3. Modular JavaScript assets are served correctly.
  4. Swagger documentation GET /docs remains accessible.
  5. Core API endpoints GET /health and POST /investigate continue functioning.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_serves_frontend_html():
    """Root GET / must return HTTP 200 and HTML5 dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "DQ INSIGHT" in response.text
    assert "Autonomous Data Quality Investigation" in response.text


def test_static_css_assets_served():
    """Static CSS files must return HTTP 200."""
    css_files = ["styles.css", "components.css", "responsive.css"]
    for css in css_files:
        response = client.get(f"/static/css/{css}")
        assert response.status_code == 200, f"Failed to serve /static/css/{css}"
        assert "text/css" in response.headers.get("content-type", "") or "text/plain" in response.headers.get("content-type", "")


def test_static_js_assets_served():
    """Static JavaScript files must return HTTP 200."""
    js_files = ["app.js", "api.js", "state.js", "ui.js", "charts.js"]
    for js in js_files:
        response = client.get(f"/static/js/{js}")
        assert response.status_code == 200, f"Failed to serve /static/js/{js}"


def test_swagger_docs_preserved():
    """OpenAPI Swagger docs at /docs must remain active and accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_health_endpoint_remains_active():
    """Backend health endpoint /health must continue to function alongside frontend."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Autonomous Data Quality Investigation Agent" in data["service"]


def test_investigate_api_remains_active():
    """Backend investigate endpoint /investigate must continue to process workflows."""
    payload = {
        "dataset_path": "data/raw/sales_problematic.csv",
        "approval_status": "PENDING",
        "use_llm": False,
    }
    response = client.post("/investigate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["workflow_status"] == "PENDING_APPROVAL"
    assert data["detected_issues"]["missing_values_count"] > 0


def test_upload_valid_csv_and_investigate():
    """POST /upload must save a valid CSV and allow investigation."""
    csv_content = (
        "transaction_id,customer_id,product_id,transaction_date,quantity,unit_price,total_amount,region,payment_method,sales_channel\n"
        "T10001,C001,P001,2026-09-01,2,100,200,North,Cash,Online\n"
        "T10002,C002,P002,2026-09-02,3,150,450,South,Credit Card,In-Store\n"
        "T10003,C003,P003,2026-09-03,1,200,200,East,PayPal,Online\n"
    )
    files = {"file": ("test_sample.csv", csv_content.encode("utf-8"), "text/csv")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "dataset_path" in data
    assert data["dataset_path"].startswith("data/uploads/")
    assert data["size_bytes"] == len(csv_content.encode("utf-8"))

    # Investigate the uploaded dataset
    inv_response = client.post("/investigate", json={"dataset_path": data["dataset_path"], "approval_status": "PENDING"})
    assert inv_response.status_code == 200
    inv_data = inv_response.json()
    assert inv_data["workflow_status"] == "NO_CORRECTION_REQUIRED"
    assert inv_data["detected_issues"]["missing_values_count"] == 0


def test_upload_non_csv_rejected():
    """POST /upload must reject non-CSV files with HTTP 400."""
    txt_content = b"This is a plain text file, not a CSV dataset."
    files = {"file": ("document.txt", txt_content, "text/plain")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Only .csv files are supported" in response.json()["detail"]


def test_upload_empty_csv_rejected():
    """POST /upload must reject empty files with HTTP 400."""
    files = {"file": ("empty.csv", b"", "text/csv")}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_path_traversal_sanitized():
    """POST /upload must sanitize path traversal attempts and store safely in data/uploads/."""
    csv_content = (
        "transaction_id,customer_id,product_id,transaction_date,quantity,unit_price,total_amount,region,payment_method,sales_channel\n"
        "T99999,C999,P999,2026-09-01,1,50,50,North,Cash,Online\n"
    )
    files = {"file": ("../../etc/passwd.csv", csv_content.encode("utf-8"), "text/csv")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert ".." not in data["filename"]
    assert "data/uploads/" in data["dataset_path"]
    assert not data["dataset_path"].startswith("..")


def test_upload_problematic_csv_and_investigate():
    """POST /upload with problematic dataset must trigger full diagnostic pipeline."""
    csv_content = (
        "transaction_id,customer_id,product_id,transaction_date,quantity,unit_price,total_amount,region,payment_method,sales_channel\n"
        "T001,C001,P001,2026-09-01,2,100,200,North,Cash,Online\n"
        "T001,C001,P001,2026-09-01,2,100,200,North,Cash,Online\n"
        "T002,,P002,2026-09-02,-5,150,-750,InvalidRegion,Bitcoin,Direct\n"
        "T003,C003,P003,invalid-date,1,200,200,East,PayPal,Online\n"
    )
    files = {"file": ("problematic_sample.csv", csv_content.encode("utf-8"), "text/csv")}
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    upload_path = response.json()["dataset_path"]

    # Run investigation
    inv_response = client.post("/investigate", json={"dataset_path": upload_path, "approval_status": "PENDING"})
    assert inv_response.status_code == 200
    inv_data = inv_response.json()
    assert inv_data["workflow_status"] == "PENDING_APPROVAL"
    assert inv_data["detected_issues"]["duplicate_rows_count"] >= 1
    assert inv_data["detected_issues"]["missing_values_count"] >= 1
    assert inv_data["detected_issues"]["schema_violations_count"] >= 1
    assert inv_data["root_cause"]["primary_cause"] != "No Data Quality Issues Detected"
    assert len(inv_data["correction_recommendation"]["actions"]) > 0

