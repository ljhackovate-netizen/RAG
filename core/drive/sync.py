"""
Google Drive integration: READ (sync files down) + WRITE (upload generated docs).

Authentication: Service Account JSON key (no OAuth browser flow).
Permission required: EDITOR on client Drive folders (to read AND write).

Setup (one-time, ~5 minutes):
  1. console.cloud.google.com → New Project
  2. APIs & Services → Enable "Google Drive API"
  3. IAM & Admin → Service Accounts → Create → Download JSON key
  4. Save JSON to ./secrets/cgn-service-account.json
  5. In Google Drive: right-click each client folder → Share →
     paste the service account email (e.g. cgn-brain@project.iam.gserviceaccount.com)
     → set permission to EDITOR (NOT Viewer — needs Editor to write back)

The Drive Folder ID is in the URL when you open a folder:
  https://drive.google.com/drive/folders/THIS_IS_THE_FOLDER_ID
"""
import os
import io
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive"]   # Full access (read + write)

SUPPORTED_MIME = {
    "text/plain":                                                    ".txt",
    "application/pdf":                                               ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":       ".xlsx",
    "text/csv":                                                      ".csv",
    "application/vnd.google-apps.document":                         ".txt",   # export
    "application/vnd.google-apps.spreadsheet":                      ".csv",   # export
}


def get_drive_service():
    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "./secrets/cgn-service-account.json")
    if not Path(sa_path).exists():
        raise FileNotFoundError(
            f"Service account JSON not found at: {sa_path}\n"
            "See setup instructions in core/drive/sync.py"
        )
    creds = service_account.Credentials.from_service_account_file(sa_path, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def verify_folder_access(service, folder_id: str) -> bool:
    """Quick check to confirm service account can access a folder."""
    try:
        service.files().get(fileId=folder_id, fields="id, name").execute()
        return True
    except HttpError as e:
        if e.resp.status in (403, 404):
            raise PermissionError(
                f"Cannot access Drive folder {folder_id}. "
                "Make sure you shared it with the service account email as Editor."
            )
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _list_subfolders(service, parent_id: str) -> list[dict]:
    query = (
        f"'{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false"
    )
    result = service.files().list(q=query, fields="files(id, name)").execute()
    return result.get("files", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def get_all_accessible_folders(service) -> list[dict]:
    """Fetch ALL folders the service account has access to."""
    query = "mimeType='application/vnd.google-apps.folder' and trashed=false"
    result = service.files().list(q=query, fields="files(id, name, parents)").execute()
    return result.get("files", [])

def get_master_folders() -> list[dict]:
    """Fetch all subfolders from the master CGN Drive folder, OR all accessible folders if master ID not set."""
    master_id = os.getenv("CGN_MASTER_DRIVE_FOLDER_ID")
    service = get_drive_service()
    if not master_id:
        return get_all_accessible_folders(service)
    return _list_subfolders(service, master_id)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _list_files(service, folder_id: str) -> list[dict]:
    query = f"'{folder_id}' in parents and trashed=false"
    result = service.files().list(
        q=query,
        fields="files(id, name, mimeType, size, modifiedTime)",
    ).execute()
    return [f for f in result.get("files", []) if f.get("mimeType") in SUPPORTED_MIME]


def _download_file(service, file_info: dict, dest_path: Path) -> bool:
    mime = file_info.get("mimeType", "")
    try:
        if mime == "application/vnd.google-apps.document":
            req = service.files().export_media(fileId=file_info["id"], mimeType="text/plain")
        elif mime == "application/vnd.google-apps.spreadsheet":
            req = service.files().export_media(fileId=file_info["id"], mimeType="text/csv")
        else:
            req = service.files().get_media(fileId=file_info["id"])

        buf = io.BytesIO()
        dl = MediaIoBaseDownload(buf, req, chunksize=1024 * 1024)
        done = False
        while not done:
            _, done = dl.next_chunk()
        dest_path.write_bytes(buf.getvalue())
        return True
    except Exception as e:
        logger.error(f"Download failed for {file_info['name']}: {e}")
        return False


def sync_client_folder(client_id: str, drive_folder_id: str, local_base_path: str) -> dict:
    """
    Download all files from a client's Google Drive folder into local cache.
    Walks all subfolders. Skips unchanged files. Writes .sync_status.json.
    Returns sync report.
    """
    service = get_drive_service()
    verify_folder_access(service, drive_folder_id)

    local_root = Path(local_base_path) / client_id
    report = {
        "client_id": client_id,
        "drive_folder_id": drive_folder_id,
        "folders_synced": [],
        "files_downloaded": [],
        "files_skipped": [],
        "errors": [],
        "total_size_mb": 0.0,
    }

    subfolders = _list_subfolders(service, drive_folder_id)

    # Also process root-level files
    all_folder_pairs = [
        {"id": drive_folder_id, "name": "_root"},
        *subfolders,
    ]

    for folder in all_folder_pairs:
        folder_name = folder["name"]
        if folder_name == "_root":
            local_dir = local_root
        else:
            local_dir = local_root / folder_name

        if folder_name == "auto_generated":
            continue  # Never sync our own outputs back down

        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            files = _list_files(service, folder["id"])
        except Exception as e:
            report["errors"].append(f"Could not list {folder_name}: {e}")
            continue

        for f in files:
            mime = f.get("mimeType", "")
            ext = SUPPORTED_MIME.get(mime, "")
            stem = Path(f["name"]).stem
            local_name = f["name"] if f["name"].endswith(ext) else stem + ext
            local_path = local_dir / local_name

            # Skip if file exists and is unchanged
            if local_path.exists() and f.get("modifiedTime"):
                drive_dt = datetime.fromisoformat(f["modifiedTime"].replace("Z", "+00:00"))
                local_dt = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)
                if local_dt >= drive_dt:
                    report["files_skipped"].append(local_name)
                    continue

            ok = _download_file(service, f, local_path)
            if ok:
                size_mb = local_path.stat().st_size / (1024 * 1024)
                report["files_downloaded"].append(local_name)
                report["total_size_mb"] += size_mb
            else:
                report["errors"].append(f"Failed: {f['name']}")
            time.sleep(0.3)  # Rate limit courtesy

        if folder_name != "_root":
            report["folders_synced"].append(folder_name)

    # Write sync status
    status_path = local_root / ".sync_status.json"
    status_path.write_text(json.dumps({
        "last_sync": datetime.utcnow().isoformat() + "Z",
        "drive_folder_id": drive_folder_id,
        "files_downloaded": len(report["files_downloaded"]),
        "last_error": report["errors"][-1] if report["errors"] else None,
    }, indent=2))

    logger.info(
        f"Drive sync complete: {client_id} — "
        f"{len(report['files_downloaded'])} downloaded, "
        f"{len(report['files_skipped'])} skipped, "
        f"{len(report['errors'])} errors"
    )
    return report
