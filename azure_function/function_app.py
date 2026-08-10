import json
import logging
import os

import azure.functions as func
from azure.data.tables import TableClient


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

TABLE_NAME = "detections"


@app.route(route="events", methods=["POST"])
def receive_event(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Detection event received.")

    try:
        data = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON"}),
            status_code=400,
            mimetype="application/json",
        )

    event_id = data.get("event_id")
    timestamp_utc = data.get("timestamp_utc")

    if not event_id or not timestamp_utc:
        return func.HttpResponse(
            json.dumps({"error": "event_id and timestamp_utc are required"}),
            status_code=400,
            mimetype="application/json",
        )

    connection_string = os.environ.get("AzureWebJobsStorage")

    if not connection_string:
        return func.HttpResponse(
            json.dumps({"error": "Storage is not configured"}),
            status_code=500,
            mimetype="application/json",
        )

    entity = {
        "PartitionKey": "detection",
        "RowKey": event_id,
        "timestamp_utc": timestamp_utc,
        "detector_backend": data.get("detector_backend", "unknown"),
        "target_detected": bool(data.get("target_detected", False)),
    }

    for field in ("confidence", "center_x", "center_y", "area"):
        value = data.get(field)
        if value is not None:
            entity[field] = value

    bbox = data.get("bbox")
    if bbox is not None:
        entity["bbox"] = json.dumps(bbox)

    try:
        table_client = TableClient.from_connection_string(
            conn_str=connection_string,
            table_name=TABLE_NAME,
        )
        table_client.upsert_entity(entity)

    except Exception:
        logging.exception("Failed to save detection event.")
        return func.HttpResponse(
            json.dumps({"error": "Failed to save event"}),
            status_code=500,
            mimetype="application/json",
        )

    return func.HttpResponse(
        json.dumps({
            "status": "accepted",
            "event_id": event_id,
        }),
        status_code=202,
        mimetype="application/json",
    )