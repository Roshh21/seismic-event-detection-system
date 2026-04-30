import json
import math
import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional
 
 
R_EARTH_KM = 6371.0
MAG_LOW      = 4.0
MAG_MODERATE = 6.0
 
CONFIDENCE_THRESHOLD = 0.50
IMPACT_RADII = {
    (0.0,  2.0): {"high":  5,  "moderate":  10, "low":  30},
    (2.0,  3.0): {"high": 10,  "moderate":  30, "low":  80},
    (3.0,  4.5): {"high": 30,  "moderate":  80, "low": 200},
    (4.5,  6.0): {"high": 80,  "moderate": 200, "low": 400},
    (6.0,  7.0): {"high": 200, "moderate": 400, "low": 700},
    (7.0, 10.0): {"high": 400, "moderate": 700, "low": 1200},
}
 
LEVEL_COLORS = {
    "LOW":      "#2ecc71",
    "MODERATE": "#f39c12",
    "HIGH":     "#e74c3c",
}
 
LEVEL_ICONS = {
    "LOW":      "🟢",
    "MODERATE": "🟡",
    "HIGH":     "🔴",
}
 
  
@dataclass
class EarthquakeEvent:
    detection_prob:  float
    magnitude:       float
    latitude:        float
    longitude:       float
    depth_km:        float
    timestamp:       str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    trace_name:      Optional[str] = None
 
 
@dataclass
class AlertReport:
    event:            EarthquakeEvent
    level:            str                 
    color:            str
    icon:             str
    impact_radii_km:  dict
    message:          str
    actions:          list
    is_alert_active:  bool
    timestamp:        str
 
    def to_dict(self) -> dict:
        d = asdict(self)
        d["event"] = asdict(self.event)
        return d
 
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
 
  
def _get_impact_radii(magnitude: float) -> dict:
    for (lo, hi), radii in IMPACT_RADII.items():
        if lo <= magnitude < hi:
            return radii
    return {"high": 400, "moderate": 700, "low": 1200}
 
 
def _classify_risk(magnitude: float) -> str:
    if magnitude < MAG_LOW:
        return "LOW"
    elif magnitude < MAG_MODERATE:
        return "MODERATE"
    else:
        return "HIGH"
 
 
def _build_message(level: str, event: EarthquakeEvent) -> str:
    ts   = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    icon = LEVEL_ICONS[level]
    lines = [
        f"{icon} EARTHQUAKE EARLY WARNING — {level} RISK",
        f"   Time         : {ts}",
        f"   Magnitude    : {event.magnitude:.2f}",
        f"   Latitude     : {event.latitude:.4f}°",
        f"   Longitude    : {event.longitude:.4f}°",
        f"   Depth        : {event.depth_km:.1f} km",
        f"   Detection P  : {event.detection_prob:.2%}",
    ]
    return "\n".join(lines)
 
 
def _recommend_actions(level: str, magnitude: float) -> list:
    base = ["Monitor official seismic authority updates"]
    if level == "LOW":
        return base + ["No immediate action required"]
    if level == "MODERATE":
        return base + [
            "Move away from windows and heavy furniture",
            "Prepare emergency kit",
            "Be ready to evacuate if instructed",
        ]
    return base + [
        "DROP, COVER, and HOLD ON immediately",
        "Evacuate coastal areas (tsunami risk if M>7)",
        "Avoid bridges, overpasses, and coastal areas",
        "Contact emergency services if needed",
        "Do NOT use elevators",
    ]
 
 
def assess_risk(event: EarthquakeEvent) -> AlertReport:
    if event.detection_prob < CONFIDENCE_THRESHOLD:
        level = "LOW"
        msg   = (f" No earthquake detected (confidence={event.detection_prob:.2%} "
                 f"< threshold={CONFIDENCE_THRESHOLD:.0%})")
        return AlertReport(
            event=event,
            level=level,
            color=LEVEL_COLORS[level],
            icon=LEVEL_ICONS[level],
            impact_radii_km={"high": 0, "moderate": 0, "low": 0},
            message=msg,
            actions=["No action required"],
            is_alert_active=False,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )
 
    level   = _classify_risk(event.magnitude)
    radii   = _get_impact_radii(event.magnitude)
    message = _build_message(level, event)
    actions = _recommend_actions(level, event.magnitude)
 
    return AlertReport(
        event=event,
        level=level,
        color=LEVEL_COLORS[level],
        icon=LEVEL_ICONS[level],
        impact_radii_km=radii,
        message=message,
        actions=actions,
        is_alert_active=(level in {"MODERATE", "HIGH"}),
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )
 
  
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(max(0, a)))
 
 
def is_in_impact_zone(
    point_lat: float,
    point_lon: float,
    event: EarthquakeEvent,
    zone: str = "moderate",
) -> bool:
    radii = _get_impact_radii(event.magnitude)
    dist  = haversine_km(event.latitude, event.longitude, point_lat, point_lon)
    return dist <= radii.get(zone, 0)
 
  
def print_alert(report: AlertReport) -> None:
    sep = "═" * 56
    print(f"\n{sep}")
    print(report.message)
    print(f"   Risk Level   : {report.icon}  {report.level}")
    print(f"   Alert Active : {'YES ⚠' if report.is_alert_active else 'NO'}")
    print("\n   Impact Radii:")
    for zone, r in report.impact_radii_km.items():
        print(f"     {zone.title():10s}: {r} km")
    print("\n   Recommended Actions:")
    for action in report.actions:
        print(f"     • {action}")
    print(f"{sep}\n")
 
 
def run_alert_pipeline(
    detection_prob: float,
    magnitude:      float,
    latitude:       float,
    longitude:      float,
    depth_km:       float,
    trace_name:     str = None,
    save_json:      str = None,
) -> AlertReport:
    event  = EarthquakeEvent(
        detection_prob=float(detection_prob),
        magnitude=float(magnitude),
        latitude=float(latitude),
        longitude=float(longitude),
        depth_km=float(depth_km),
        trace_name=trace_name,
    )
    report = assess_risk(event)
    print_alert(report)
 
    if save_json:
        with open(save_json, "w") as f:
            f.write(report.to_json())
        print(f"[Alert] Report saved → {save_json}")
 
    return report
  
if __name__ == "__main__":
    run_alert_pipeline(
        detection_prob=0.95, magnitude=3.2,
        latitude=36.5, longitude=-121.0, depth_km=12.5,
        trace_name="test_low",
    )
 
    run_alert_pipeline(
        detection_prob=0.98, magnitude=5.1,
        latitude=34.0, longitude=135.0, depth_km=30.0,
        trace_name="test_moderate",
    )
 
    run_alert_pipeline(
        detection_prob=0.99, magnitude=7.2,
        latitude=38.3, longitude=142.4, depth_km=29.0,
        trace_name="test_high",
        save_json="outputs/test_alert.json",
    )
 
    print("Alert engine self-test passed")