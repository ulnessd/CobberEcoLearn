"""
CobberEcoFetcher.py

A standalone PyQt6 application for exploring ecological data from:
- GBIF (biodiversity occurrence records)
- iNaturalist (community observations)

Version 4 goals:
- Stable Chapter 5 classroom build
- No NEON in the main fetcher workflow
- No Build Dataset toggle
- Launch directly into the main workspace
- Keep a real map when Qt WebEngine is available
- Avoid the v3 hidden-tab WebEngine sizing/crash issues by lazy-loading the map only
  when the Spatial View tab is actually selected
- Fall back to a self-contained coordinate plot if Qt WebEngine is unavailable

Dependencies:
    pip install PyQt6 requests
Optional for live map support:
    pip install PyQt6-WebEngine
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QSpinBox,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except Exception:
    QWebEngineView = None  # type: ignore
    WEBENGINE_AVAILABLE = False

APP_TITLE = "CobberEcoFetcher"
USER_AGENT = "CobberEcoFetcher/4.0 (educational prototype; contact instructor)"
GBIF_BASE = "https://api.gbif.org/v1"
INAT_BASE = "https://api.inaturalist.org/v1"
REQUEST_TIMEOUT = 20

SAMPLE_QUERIES = {
    "GBIF – Bur oak in North America": {
        "source": "GBIF",
        "scientific_name": "Quercus macrocarpa",
        "common_name": "",
        "region": "",
        "start": "2000-01-01",
        "end": datetime.now().strftime("%Y-%m-%d"),
        "coords_only": True,
        "limit": 100,
    },
    "GBIF – Bur oak in Minnesota": {
        "source": "GBIF",
        "scientific_name": "Quercus macrocarpa",
        "common_name": "",
        "region": "Minnesota",
        "start": "2000-01-01",
        "end": datetime.now().strftime("%Y-%m-%d"),
        "coords_only": True,
        "limit": 100,
    },
    "iNaturalist – Monarch in Iowa": {
        "source": "iNaturalist",
        "taxon_name": "Danaus plexippus",
        "place": "Iowa",
        "start": "2025-08-01",
        "end": "2025-10-15",
        "photos_only": True,
        "research_grade": True,
        "limit": 50,
    },
    "iNaturalist – Bumblebees with photos": {
        "source": "iNaturalist",
        "taxon_name": "Bombus",
        "place": "Minnesota",
        "start": "2024-04-01",
        "end": datetime.now().strftime("%Y-%m-%d"),
        "photos_only": True,
        "research_grade": True,
        "limit": 60,
    },
}


@dataclass
class ResultPackage:
    source: str
    title: str
    query_summary: Dict[str, Any]
    fetched_at: str
    summary_html: str
    preview_rows: List[Dict[str, Any]]
    raw_metadata: Dict[str, Any]
    map_points: List[Dict[str, Any]] = field(default_factory=list)
    media_items: List[Dict[str, Any]] = field(default_factory=list)
    export_name: str = "dataset"


def safe_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def html_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def normalize_date_for_query(date_str: str) -> str:
    return date_str[:10]


class GBIFAdapter:
    @staticmethod
    def fetch_preview(query: Dict[str, Any], progress_cb=None) -> ResultPackage:
        search_name = query.get("scientific_name") or query.get("common_name")
        if not search_name:
            raise ValueError("Please enter a scientific name or common name for GBIF.")

        if progress_cb:
            progress_cb(f"Matching taxon '{search_name}' in GBIF...")
        match_json = safe_get_json(f"{GBIF_BASE}/species/match", {"name": search_name})
        usage_key = match_json.get("usageKey")
        accepted_name = (
            match_json.get("scientificName")
            or match_json.get("canonicalName")
            or search_name
        )

        params: Dict[str, Any] = {
            "limit": min(int(query.get("limit", 100)), 200),
            "offset": 0,
        }
        if usage_key:
            params["taxonKey"] = usage_key
        else:
            params["scientificName"] = search_name

        region = (query.get("region") or "").strip()
        if region:
            if len(region) == 2:
                params["country"] = region.upper()
            else:
                params["stateProvince"] = region

        start = query.get("start") or ""
        end = query.get("end") or ""
        if start or end:
            start_year = (start[:4] if start else "0000")
            end_year = (end[:4] if end else "9999")
            params["year"] = f"{start_year},{end_year}"

        if query.get("coords_only"):
            params["hasCoordinate"] = "true"

        if progress_cb:
            progress_cb("Fetching occurrence preview from GBIF...")
        occ_json = safe_get_json(f"{GBIF_BASE}/occurrence/search", params)
        results = occ_json.get("results", [])

        preview_rows: List[Dict[str, Any]] = []
        points: List[Dict[str, Any]] = []
        media_items: List[Dict[str, Any]] = []
        countries: Dict[str, int] = {}

        for item in results:
            country = item.get("country") or item.get("countryCode") or ""
            if country:
                countries[country] = countries.get(country, 0) + 1
            row = {
                "source": "GBIF",
                "record_id": item.get("key"),
                "scientific_name": item.get("scientificName") or item.get("species") or "",
                "accepted_name": accepted_name,
                "basis_of_record": item.get("basisOfRecord", ""),
                "event_date": item.get("eventDate") or item.get("year") or "",
                "country": country,
                "state_province": item.get("stateProvince", ""),
                "decimalLatitude": item.get("decimalLatitude", ""),
                "decimalLongitude": item.get("decimalLongitude", ""),
                "license": item.get("license", ""),
                "dataset": item.get("datasetName", ""),
            }
            preview_rows.append(row)
            lat = item.get("decimalLatitude")
            lon = item.get("decimalLongitude")
            if lat is not None and lon is not None:
                points.append({
                    "lat": lat,
                    "lon": lon,
                    "label": row["scientific_name"] or accepted_name,
                    "popup": f"GBIF #{item.get('key')}<br>{html_escape(row['scientific_name'] or accepted_name)}<br>{html_escape(country)}",
                })
            media = item.get("media") or []
            for m in media[:1]:
                media_items.append({
                    "title": row["scientific_name"] or accepted_name,
                    "url": m.get("identifier") or m.get("references") or "",
                    "type": m.get("type", "media"),
                })

        top_countries = ", ".join(
            f"{k} ({v})" for k, v in sorted(countries.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ) or "None in preview"
        summary_html = f"""
        <h2>GBIF preview</h2>
        <p><b>Matched taxon:</b> {html_escape(accepted_name)}</p>
        <p><b>GBIF taxon key:</b> {html_escape(usage_key)}</p>
        <p><b>Preview rows:</b> {len(preview_rows)} of {occ_json.get('count', 'unknown')}</p>
        <p><b>Top countries/regions in preview:</b> {html_escape(top_countries)}</p>
        <p><b>Teaching prompt:</b> What is the scientific object here: an occurrence record tied to place and time, or a species concept?</p>
        """

        return ResultPackage(
            source="GBIF",
            title=f"GBIF: {accepted_name}",
            query_summary=query,
            fetched_at=datetime.now().isoformat(timespec="seconds"),
            summary_html=summary_html,
            preview_rows=preview_rows,
            raw_metadata={"match": match_json, "occurrence_search": occ_json},
            map_points=points,
            media_items=media_items,
            export_name=f"gbif_{accepted_name.replace(' ', '_')}",
        )


class INaturalistAdapter:
    @staticmethod
    def _resolve_place(place_text: str) -> Tuple[Optional[int], Optional[str], Dict[str, Any]]:
        if not place_text.strip():
            return None, None, {}
        data = safe_get_json(f"{INAT_BASE}/places/autocomplete", {"q": place_text, "per_page": 1})
        results = data.get("results", [])
        if not results:
            return None, None, data
        place = results[0]
        return place.get("id"), place.get("display_name") or place.get("name"), data

    @staticmethod
    def fetch_preview(query: Dict[str, Any], progress_cb=None) -> ResultPackage:
        taxon_name = (query.get("taxon_name") or "").strip()
        if not taxon_name:
            raise ValueError("Please enter a taxon name for iNaturalist.")

        place_text = query.get("place") or ""
        place_id = None
        place_name = None
        place_lookup: Dict[str, Any] = {}
        if place_text.strip():
            if progress_cb:
                progress_cb(f"Resolving iNaturalist place '{place_text}'...")
            place_id, place_name, place_lookup = INaturalistAdapter._resolve_place(place_text)

        params: Dict[str, Any] = {
            "taxon_name": taxon_name,
            "per_page": min(int(query.get("limit", 100)), 200),
            "page": 1,
            "order_by": "observed_on",
            "order": "desc",
        }
        if query.get("photos_only"):
            params["photos"] = "true"
        if query.get("research_grade"):
            params["quality_grade"] = "research"
        if place_id is not None:
            params["place_id"] = place_id
        start = normalize_date_for_query(query.get("start") or "")
        end = normalize_date_for_query(query.get("end") or "")
        if start:
            params["d1"] = start
        if end:
            params["d2"] = end

        if progress_cb:
            progress_cb("Fetching observation preview from iNaturalist...")
        data = safe_get_json(f"{INAT_BASE}/observations", params)
        results = data.get("results", [])

        preview_rows: List[Dict[str, Any]] = []
        points: List[Dict[str, Any]] = []
        media_items: List[Dict[str, Any]] = []
        places_seen: Dict[str, int] = {}
        photo_count = 0

        for item in results:
            taxon = item.get("taxon") or {}
            user = item.get("user") or {}
            place_guess = item.get("place_guess") or ""
            if place_guess:
                places_seen[place_guess] = places_seen.get(place_guess, 0) + 1
            photos = item.get("photos") or []
            if photos:
                photo_count += 1
            observed_on = item.get("observed_on") or item.get("time_observed_at") or ""
            row = {
                "source": "iNaturalist",
                "record_id": item.get("id"),
                "scientific_name": (taxon.get("name") or taxon_name),
                "common_name": ((taxon.get("preferred_common_name") or "") if isinstance(taxon, dict) else ""),
                "observer": user.get("login") or "",
                "observed_on": observed_on,
                "place_guess": place_guess,
                "quality_grade": item.get("quality_grade", ""),
                "photo_count": len(photos),
                "latitude": item.get("geojson", {}).get("coordinates", [None, None])[1],
                "longitude": item.get("geojson", {}).get("coordinates", [None, None])[0],
                "license": item.get("license_code", ""),
                "uri": item.get("uri", ""),
            }
            preview_rows.append(row)
            lat = row["latitude"]
            lon = row["longitude"]
            if lat is not None and lon is not None:
                points.append({
                    "lat": lat,
                    "lon": lon,
                    "label": row["scientific_name"],
                    "popup": f"iNat #{item.get('id')}<br>{html_escape(row['scientific_name'])}<br>{html_escape(place_guess)}",
                })
            if photos:
                first = photos[0]
                thumb = first.get("url") or ""
                media_items.append({
                    "title": f"Observation {item.get('id')}",
                    "url": thumb.replace("square", "medium") if thumb else "",
                    "link": item.get("uri", ""),
                    "type": "photo",
                })

        top_places = ", ".join(
            f"{k} ({v})" for k, v in sorted(places_seen.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ) or "None in preview"
        summary_html = f"""
        <h2>iNaturalist preview</h2>
        <p><b>Taxon:</b> {html_escape(taxon_name)}</p>
        <p><b>Resolved place:</b> {html_escape(place_name or place_text or 'Not specified')}</p>
        <p><b>Preview observations:</b> {len(preview_rows)} of {data.get('total_results', 'unknown')}</p>
        <p><b>Observations with photos in preview:</b> {photo_count}</p>
        <p><b>Top places in preview:</b> {html_escape(top_places)}</p>
        <p><b>Teaching prompt:</b> How is a community observation different from a museum-like occurrence record?</p>
        """

        return ResultPackage(
            source="iNaturalist",
            title=f"iNat: {taxon_name}",
            query_summary=query,
            fetched_at=datetime.now().isoformat(timespec="seconds"),
            summary_html=summary_html,
            preview_rows=preview_rows,
            raw_metadata={"place_lookup": place_lookup, "observations": data},
            map_points=points,
            media_items=media_items,
            export_name=f"inat_{taxon_name.replace(' ', '_')}",
        )


class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)


class FetchWorker(QRunnable):
    def __init__(self, source: str, query: Dict[str, Any]):
        super().__init__()
        self.source = source
        self.query = query
        self.signals = WorkerSignals()

    def run(self):
        try:
            def progress(msg: str):
                self.signals.progress.emit(msg)

            if self.source == "GBIF":
                result = GBIFAdapter.fetch_preview(self.query, progress)
            else:
                result = INaturalistAdapter.fetch_preview(self.query, progress)
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class HtmlPanel(QWidget):
    def __init__(self, html: str = ""):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(html)
        layout.addWidget(self.browser)


class CoordinatePlotFallback(QWidget):
    def __init__(self, points: List[Dict[str, Any]]):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(self._build_html(points))
        layout.addWidget(browser)

    def _build_html(self, points: List[Dict[str, Any]]) -> str:
        valid = []
        for p in points:
            try:
                lat = float(p.get("lat"))
                lon = float(p.get("lon"))
                valid.append({**p, "lat": lat, "lon": lon})
            except Exception:
                continue

        if not valid:
            return "<h3>Spatial preview</h3><p>No coordinates were available for this result.</p>"

        lats = [p["lat"] for p in valid]
        lons = [p["lon"] for p in valid]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        lat_pad = max(1.0, (max_lat - min_lat) * 0.08)
        lon_pad = max(1.0, (max_lon - min_lon) * 0.08)
        min_lat -= lat_pad
        max_lat += lat_pad
        min_lon -= lon_pad
        max_lon += lon_pad

        width = 880
        height = 520
        left = 78
        top = 28
        plot_w = width - 140
        plot_h = height - 86

        def x_of(lon: float) -> float:
            return left + (lon - min_lon) / max(max_lon - min_lon, 1e-9) * plot_w

        def y_of(lat: float) -> float:
            return top + plot_h - (lat - min_lat) / max(max_lat - min_lat, 1e-9) * plot_h

        grid_lines: List[str] = []
        for i in range(6):
            x = left + i * plot_w / 5
            lon_val = min_lon + i * (max_lon - min_lon) / 5
            grid_lines.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + plot_h}' stroke='#d9d9d9' stroke-width='1'/>")
            grid_lines.append(f"<text x='{x:.1f}' y='{top + plot_h + 24}' text-anchor='middle' font-size='12' fill='#555'>{lon_val:.1f}°</text>")
        for j in range(6):
            y = top + j * plot_h / 5
            lat_val = max_lat - j * (max_lat - min_lat) / 5
            grid_lines.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='#d9d9d9' stroke-width='1'/>")
            grid_lines.append(f"<text x='{left - 10}' y='{y + 4:.1f}' text-anchor='end' font-size='12' fill='#555'>{lat_val:.1f}°</text>")

        circles: List[str] = []
        for p in valid[:500]:
            x = x_of(p["lon"])
            y = y_of(p["lat"])
            popup = html_escape(p.get("popup", p.get("label", "Record")))
            circles.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5.5' fill='#2b7bbb' fill-opacity='0.78' stroke='white' stroke-width='1.2'>"
                f"<title>{popup}</title></circle>"
            )

        return f"""
        <html>
        <body style='font-family:Arial, sans-serif; margin:0; padding:10px; background:#fafafa;'>
          <div style='margin-bottom:8px;'><b>Spatial preview</b> — coordinate plot fallback.</div>
          <svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' style='background:white; border:1px solid #d0d0d0;'>
            <rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='#f6fbff' stroke='#888' stroke-width='1.2'/>
            {''.join(grid_lines)}
            {''.join(circles)}
            <text x='{left + plot_w / 2:.1f}' y='{height - 12}' text-anchor='middle' font-size='13' fill='#444'>Longitude</text>
            <text x='18' y='{top + plot_h / 2:.1f}' text-anchor='middle' font-size='13' fill='#444' transform='rotate(-90 18 {top + plot_h / 2:.1f})'>Latitude</text>
          </svg>
          <div style='font-size:12px; color:#555; margin-top:8px;'>
            Bounding box: lat {min_lat:.2f} to {max_lat:.2f}, lon {min_lon:.2f} to {max_lon:.2f}. Showing {len(valid)} plotted point(s).
          </div>
        </body>
        </html>
        """


class LazyMapPanel(QWidget):
    """Build the live WebEngine map only when the Spatial View tab is selected.

    This avoids the hidden-tab sizing problems and the v3 instability from building/refreshing
    QWebEngine content before the tab is visible.
    """

    def __init__(self, points: List[Dict[str, Any]]):
        super().__init__()
        self.points = points
        self.built = False
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QLabel("Select the Spatial View tab to load the map.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.placeholder)

    def ensure_built(self):
        if self.built:
            return
        self.built = True
        self.layout.removeWidget(self.placeholder)
        self.placeholder.deleteLater()
        if WEBENGINE_AVAILABLE:
            view = QWebEngineView()
            view.setHtml(self._build_leaflet_html(self.points))
            self.layout.addWidget(view)
        else:
            self.layout.addWidget(CoordinatePlotFallback(self.points))

    def _build_leaflet_html(self, points: List[Dict[str, Any]]) -> str:
        valid = []
        for p in points:
            try:
                lat = float(p.get("lat"))
                lon = float(p.get("lon"))
                valid.append({**p, "lat": lat, "lon": lon})
            except Exception:
                continue

        markers_js = []
        bounds_entries = []
        for p in valid[:500]:
            markers_js.append(
                f"L.marker([{p['lat']}, {p['lon']}]).addTo(map).bindPopup({json.dumps(p.get('popup', p.get('label', 'Record')))});"
            )
            bounds_entries.append(f"[{p['lat']}, {p['lon']}]")

        if len(valid) == 1:
            center_script = f"map.setView([{valid[0]['lat']}, {valid[0]['lon']}], 8);"
        elif len(valid) > 1:
            center_script = (
                f"var bounds = L.latLngBounds([{', '.join(bounds_entries)}]);"
                "map.fitBounds(bounds.pad(0.12));"
            )
        else:
            center_script = "map.setView([39.5, -98.35], 3);"

        count_note = f"{len(valid)} mapped point(s)" if valid else "No coordinates available"

        return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; padding: 0; background: #f4f4f4; }}
    #map {{ height: 100%; width: 100%; }}
    .badge {{ position:absolute; top:12px; right:12px; z-index:9999; background:white; padding:8px 10px; border-radius:6px; border:1px solid #ddd; font-family:Arial,sans-serif; color:#555; }}
  </style>
</head>
<body>
  <div class="badge">{html_escape(count_note)}</div>
  <div id="map"></div>
  <script>
    var map = L.map('map', {{zoomControl: true, worldCopyJump: false}});
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      subdomains: 'abcd',
      maxZoom: 19,
      noWrap: true,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);
    {''.join(markers_js)}
    {center_script}
  </script>
</body>
</html>
        """


class SimpleTablePanel(QWidget):
    def __init__(self, rows: List[Dict[str, Any]]):
        super().__init__()
        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.populate(rows)

    def populate(self, rows: List[Dict[str, Any]]):
        self.table.clear()
        if not rows:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        columns = list(rows[0].keys())
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, col in enumerate(columns):
                self.table.setItem(r, c, QTableWidgetItem("" if row.get(col) is None else str(row.get(col))))
        self.table.resizeColumnsToContents()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)


class MediaPanel(QWidget):
    def __init__(self, items: List[Dict[str, Any]]):
        super().__init__()
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        if not items:
            browser.setHtml("<p>No media preview is available for this result.</p>")
        else:
            cards = []
            for item in items[:30]:
                url = item.get("url", "")
                link = item.get("link", url)
                title = html_escape(item.get("title", "Item"))
                if url:
                    cards.append(
                        f"<div style='margin-bottom:14px;'><b>{title}</b><br><a href='{html_escape(link)}'><img src='{html_escape(url)}' width='220'></a></div>"
                    )
                else:
                    cards.append(f"<div><b>{title}</b></div>")
            browser.setHtml("<h3>Media preview</h3>" + "".join(cards))
        layout.addWidget(browser)


class RawMetadataPanel(QWidget):
    def __init__(self, metadata: Dict[str, Any]):
        super().__init__()
        layout = QVBoxLayout(self)
        edit = QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(json.dumps(metadata, indent=2, ensure_ascii=False))
        edit.setFont(QFont("Courier New", 10))
        layout.addWidget(edit)


class SampleQueryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sample Queries")
        self.resize(520, 360)
        layout = QVBoxLayout(self)
        self.combo = QComboBox()
        self.combo.addItems(list(SAMPLE_QUERIES.keys()))
        self.details = QTextBrowser()
        self.details.setHtml(self._details_html(self.combo.currentText()))
        self.combo.currentTextChanged.connect(lambda text: self.details.setHtml(self._details_html(text)))
        layout.addWidget(QLabel("Choose a sample query:"))
        layout.addWidget(self.combo)
        layout.addWidget(self.details)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _details_html(self, name: str) -> str:
        payload = SAMPLE_QUERIES.get(name, {})
        bits = "".join(f"<li><b>{html_escape(k)}:</b> {html_escape(v)}</li>" for k, v in payload.items())
        return f"<h3>{html_escape(name)}</h3><ul>{bits}</ul>"

    def get_payload(self) -> Optional[Dict[str, Any]]:
        if self.exec() == QDialog.DialogCode.Accepted:
            return SAMPLE_QUERIES.get(self.combo.currentText())
        return None


class CobberEcoFetcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()
        self.results: List[ResultPackage] = []

        self.cobber_maroon = QColor(108, 29, 69)
        self.eco_green = QColor(38, 102, 66)
        self.base_font = QFont("Lato", 10)
        self.setFont(self.base_font)

        self.setWindowTitle(APP_TITLE)
        self._set_laptop_friendly_geometry()

        self._build_ui()
        self._build_menu()
        self.statusBar().showMessage("Ready. Choose a source and fetch a preview.")

    def _set_laptop_friendly_geometry(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1180, 720)
            return
        geom = screen.availableGeometry()
        width = min(1220, max(1040, int(geom.width() * 0.92)))
        height = min(720, max(640, int(geom.height() * 0.88)))
        self.resize(width, height)
        x = geom.x() + max(0, (geom.width() - width) // 2)
        y = geom.y() + max(0, (geom.height() - height) // 2)
        self.move(x, y)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        sample_action = QAction("Load Sample Query", self)
        sample_action.triggered.connect(self.load_sample_query)
        export_action = QAction("Export Current Result", self)
        export_action.triggered.connect(self.export_current_result)
        file_menu.addAction(sample_action)
        file_menu.addAction(export_action)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.addWidget(self._build_main_page())
        self.setStatusBar(QStatusBar())

    def _build_main_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(320)
        left.setMaximumWidth(400)
        left_layout = QVBoxLayout(left)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["GBIF", "iNaturalist"])
        self.source_combo.currentTextChanged.connect(self.on_source_changed)

        form_box = QGroupBox("Query Builder")
        form_layout = QVBoxLayout(form_box)
        form_layout.addWidget(QLabel("Select source:"))
        form_layout.addWidget(self.source_combo)

        self.form_stack = QWidget()
        form_stack_layout = QVBoxLayout(self.form_stack)
        form_stack_layout.setContentsMargins(0, 0, 0, 0)
        self.gbif_form = self._build_gbif_form()
        self.inat_form = self._build_inat_form()
        form_stack_layout.addWidget(self.gbif_form)
        form_stack_layout.addWidget(self.inat_form)
        form_layout.addWidget(self.form_stack)

        btn_row = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch Preview")
        self.fetch_btn.clicked.connect(self.start_fetch)
        self.fetch_btn.setStyleSheet(
            f"QPushButton {{ background-color: {self.cobber_maroon.name()}; color: white; padding: 8px; font-weight: bold; border-radius: 5px; }}"
        )
        self.export_btn = QPushButton("Export Dataset")
        self.export_btn.clicked.connect(self.export_current_result)
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)
        btn_row.addWidget(self.fetch_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.clear_btn)

        self.log_console = QTextBrowser()
        self.log_console.setOpenExternalLinks(True)

        left_layout.addWidget(form_box)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(QLabel("Report Log:"))
        left_layout.addWidget(self.log_console, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.results_tabs = QTabWidget()
        self.results_tabs.setTabsClosable(True)
        self.results_tabs.tabCloseRequested.connect(self.close_result_tab)
        empty = QTextBrowser()
        empty.setHtml(
            "<h2>Welcome to CobberEcoFetcher</h2>"
            "<p>Choose GBIF or iNaturalist, build a query, and fetch a preview of real ecological data.</p>"
            "<p><b>Canonical Chapter 5 examples:</b> Bur oak in GBIF, and monarch observations in Iowa from iNaturalist.</p>"
        )
        right_layout.addWidget(self.results_tabs)
        self.results_tabs.addTab(empty, "Start Here")

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 820])
        self.on_source_changed(self.source_combo.currentText())
        return page

    def _build_gbif_form(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        self.gbif_scientific = QLineEdit()
        self.gbif_common = QLineEdit()
        self.gbif_region = QLineEdit()
        self.gbif_region.setPlaceholderText("US, CA, Minnesota, etc.")
        self.gbif_start = QDateEdit()
        self.gbif_start.setCalendarPopup(True)
        self.gbif_start.setDate(date(2000, 1, 1))
        self.gbif_end = QDateEdit()
        self.gbif_end.setCalendarPopup(True)
        self.gbif_end.setDate(date.today())
        self.gbif_coords = QCheckBox("Only records with coordinates")
        self.gbif_coords.setChecked(True)
        self.gbif_limit = QSpinBox()
        self.gbif_limit.setRange(10, 200)
        self.gbif_limit.setValue(100)
        form.addRow("Scientific name:", self.gbif_scientific)
        form.addRow("Common name:", self.gbif_common)
        form.addRow("Country / region:", self.gbif_region)
        form.addRow("Start date:", self.gbif_start)
        form.addRow("End date:", self.gbif_end)
        form.addRow("Coordinates:", self.gbif_coords)
        form.addRow("Max preview rows:", self.gbif_limit)
        note = QLabel("GBIF is best for large-scale biodiversity occurrence records.")
        note.setWordWrap(True)
        form.addRow(note)
        return widget

    def _build_inat_form(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        self.inat_taxon = QLineEdit()
        self.inat_place = QLineEdit()
        self.inat_start = QDateEdit()
        self.inat_start.setCalendarPopup(True)
        self.inat_start.setDate(date(2025, 8, 1))
        self.inat_end = QDateEdit()
        self.inat_end.setCalendarPopup(True)
        self.inat_end.setDate(date.today())
        self.inat_photos = QCheckBox("Photos only")
        self.inat_photos.setChecked(True)
        self.inat_research = QCheckBox("Research grade only")
        self.inat_research.setChecked(True)
        self.inat_limit = QSpinBox()
        self.inat_limit.setRange(10, 200)
        self.inat_limit.setValue(50)
        form.addRow("Taxon name:", self.inat_taxon)
        form.addRow("Place:", self.inat_place)
        form.addRow("Start date:", self.inat_start)
        form.addRow("End date:", self.inat_end)
        form.addRow("Photos:", self.inat_photos)
        form.addRow("Quality:", self.inat_research)
        form.addRow("Max preview rows:", self.inat_limit)
        note = QLabel("iNaturalist is best for community observations, often with photographs.")
        note.setWordWrap(True)
        form.addRow(note)
        return widget

    def on_source_changed(self, source: str):
        self.gbif_form.setVisible(source == "GBIF")
        self.inat_form.setVisible(source == "iNaturalist")

    def _collect_query(self) -> Tuple[str, Dict[str, Any]]:
        source = self.source_combo.currentText()
        if source == "GBIF":
            return source, {
                "scientific_name": self.gbif_scientific.text().strip(),
                "common_name": self.gbif_common.text().strip(),
                "region": self.gbif_region.text().strip(),
                "start": self.gbif_start.date().toString("yyyy-MM-dd"),
                "end": self.gbif_end.date().toString("yyyy-MM-dd"),
                "coords_only": self.gbif_coords.isChecked(),
                "limit": self.gbif_limit.value(),
            }
        return source, {
            "taxon_name": self.inat_taxon.text().strip(),
            "place": self.inat_place.text().strip(),
            "start": self.inat_start.date().toString("yyyy-MM-dd"),
            "end": self.inat_end.date().toString("yyyy-MM-dd"),
            "photos_only": self.inat_photos.isChecked(),
            "research_grade": self.inat_research.isChecked(),
            "limit": self.inat_limit.value(),
        }

    def start_fetch(self):
        source, query = self._collect_query()
        self.fetch_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        worker = FetchWorker(source, query)
        worker.signals.progress.connect(self.log_progress)
        worker.signals.result.connect(self.add_result_tab)
        worker.signals.error.connect(self.log_error)
        worker.signals.finished.connect(self.on_fetch_finished)
        self.threadpool.start(worker)

    def on_fetch_finished(self):
        self.fetch_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.statusBar().showMessage("Fetch complete.", 5000)

    def log_progress(self, message: str):
        self.statusBar().showMessage(message)
        self.log_console.append(f"<span style='color:#555;'>{html_escape(message)}</span>")

    def log_error(self, message: str):
        self.statusBar().showMessage("Error.", 5000)
        self.log_console.append(f"<span style='color:red;'><b>Error:</b> {html_escape(message)}</span>")
        QMessageBox.warning(self, APP_TITLE, message)

    def log_success(self, message: str):
        self.log_console.append(f"<span style='color:green;'><b>Success:</b> {html_escape(message)}</span>")

    def add_result_tab(self, result: ResultPackage):
        if self.results_tabs.count() == 1 and self.results_tabs.tabText(0) == "Start Here":
            self.results_tabs.removeTab(0)
        self.results.append(result)

        container = QWidget()
        layout = QVBoxLayout(container)
        subtabs = QTabWidget()
        map_panel = LazyMapPanel(result.map_points)

        subtabs.addTab(HtmlPanel(result.summary_html + self._query_summary_html(result)), "Overview")
        subtabs.addTab(SimpleTablePanel(result.preview_rows), "Preview Table")
        subtabs.addTab(map_panel, "Spatial View")
        subtabs.addTab(MediaPanel(result.media_items), "Media / Package")
        subtabs.addTab(RawMetadataPanel(result.raw_metadata), "Metadata / Raw")

        def maybe_build_map(index: int, tabs=subtabs, panel=map_panel):
            if tabs.tabText(index) == "Spatial View":
                panel.ensure_built()

        subtabs.currentChanged.connect(maybe_build_map)

        export_row = QHBoxLayout()
        export_now = QPushButton("Export This Result")
        export_now.clicked.connect(lambda: self.export_result(result))
        export_row.addStretch(1)
        export_row.addWidget(export_now)

        layout.addWidget(subtabs)
        layout.addLayout(export_row)
        self.results_tabs.addTab(container, result.title)
        self.results_tabs.setCurrentWidget(container)
        self.log_success(f"Fetched {result.title}.")

    def _query_summary_html(self, result: ResultPackage) -> str:
        items = "".join(
            f"<li><b>{html_escape(k)}:</b> {html_escape(v)}</li>"
            for k, v in result.query_summary.items()
        )
        return f"<h3>Query summary</h3><ul>{items}</ul><p><b>Fetched at:</b> {html_escape(result.fetched_at)}</p>"

    def close_result_tab(self, index: int):
        self.results_tabs.removeTab(index)
        if self.results_tabs.count() == 0:
            empty = QTextBrowser()
            empty.setHtml(
                "<h2>Welcome to CobberEcoFetcher</h2>"
                "<p>Choose GBIF or iNaturalist, build a query, and fetch a preview of real ecological data.</p>"
                "<p><b>Canonical Chapter 5 examples:</b> Bur oak in GBIF, and monarch observations in Iowa from iNaturalist.</p>"
            )
            self.results_tabs.addTab(empty, "Start Here")

    def current_result(self) -> Optional[ResultPackage]:
        current_index = self.results_tabs.currentIndex()
        if current_index < 0:
            return None
        tab_title = self.results_tabs.tabText(current_index)
        for result in reversed(self.results):
            if result.title == tab_title:
                return result
        return None

    def export_current_result(self):
        result = self.current_result()
        if result is None:
            QMessageBox.information(self, APP_TITLE, "There is no result tab selected to export.")
            return
        self.export_result(result)

    def export_result(self, result: ResultPackage):
        target_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not target_dir:
            return
        export_dir = Path(target_dir)
        base = result.export_name or "dataset"
        csv_path = export_dir / f"{base}.csv"
        json_path = export_dir / f"{base}_metadata.json"
        txt_path = export_dir / f"{base}_query_summary.txt"

        try:
            if result.preview_rows:
                columns = list(result.preview_rows[0].keys())
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(result.preview_rows)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(result.raw_metadata, f, indent=2, ensure_ascii=False)
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(f"Source: {result.source}\n")
                f.write(f"Title: {result.title}\n")
                f.write(f"Fetched at: {result.fetched_at}\n\n")
                f.write("Query parameters:\n")
                for k, v in result.query_summary.items():
                    f.write(f"- {k}: {v}\n")
                f.write("\nNotes:\n")
                f.write("- Export created by CobberEcoFetcher version 4.\n")
                f.write("- Review source licensing and citation requirements before publication.\n")
            self.log_success(f"Exported files to {export_dir}")
            QMessageBox.information(self, APP_TITLE, f"Exported:\n{csv_path.name}\n{json_path.name}\n{txt_path.name}")
        except Exception as exc:
            self.log_error(f"Export failed: {exc}")

    def clear_all(self):
        self.gbif_scientific.clear()
        self.gbif_common.clear()
        self.gbif_region.clear()
        self.inat_taxon.clear()
        self.inat_place.clear()
        self.log_console.clear()
        self.results_tabs.clear()
        empty = QTextBrowser()
        empty.setHtml(
            "<h2>Welcome to CobberEcoFetcher</h2>"
            "<p>Choose GBIF or iNaturalist, build a query, and fetch a preview of real ecological data.</p>"
            "<p><b>Canonical Chapter 5 examples:</b> Bur oak in GBIF, and monarch observations in Iowa from iNaturalist.</p>"
        )
        self.results_tabs.addTab(empty, "Start Here")
        self.statusBar().showMessage("Ready. Fields cleared.")

    def load_sample_query(self):
        dialog = SampleQueryDialog(self)
        payload = dialog.get_payload()
        if not payload:
            return
        source = payload.get("source", "GBIF")
        self.source_combo.setCurrentText(source)
        if source == "GBIF":
            self.gbif_scientific.setText(payload.get("scientific_name", ""))
            self.gbif_common.setText(payload.get("common_name", ""))
            self.gbif_region.setText(payload.get("region", ""))
            self.gbif_coords.setChecked(bool(payload.get("coords_only", True)))
            self.gbif_limit.setValue(int(payload.get("limit", 100)))
            self.gbif_start.setDate(datetime.strptime(payload.get("start", "2000-01-01"), "%Y-%m-%d").date())
            self.gbif_end.setDate(datetime.strptime(payload.get("end", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d").date())
        else:
            self.inat_taxon.setText(payload.get("taxon_name", ""))
            self.inat_place.setText(payload.get("place", ""))
            self.inat_photos.setChecked(bool(payload.get("photos_only", True)))
            self.inat_research.setChecked(bool(payload.get("research_grade", True)))
            self.inat_limit.setValue(int(payload.get("limit", 50)))
            self.inat_start.setDate(datetime.strptime(payload.get("start", "2025-08-01"), "%Y-%m-%d").date())
            self.inat_end.setDate(datetime.strptime(payload.get("end", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d").date())
        self.log_success(f"Loaded sample query for {source}.")


def main():
    app = QApplication(sys.argv)
    window = CobberEcoFetcherApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
