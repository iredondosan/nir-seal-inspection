"""
cvat.py — read/write CVAT-for-images-1.1 annotation XML.

Conventions used across the project:
  * polygon label "sellado"  -> the seal ring (TWO polygons per pack: outer + inner)
  * polygon label "defect"   -> a defect region
  * image-level tags         -> "reviewed" (human-verified GT), "good" / "defect"
                                (pack class), "exclude" (e.g. sticker over seal)
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np
import cv2


def parse_points(s: str) -> np.ndarray:
    """'x1,y1;x2,y2;...' -> (N,2) float array."""
    return np.array([[float(a) for a in p.split(",")] for p in s.strip().split(";")], np.float32)


def tags(image_node) -> set[str]:
    """Set of image-level tag labels (e.g. {'reviewed', 'good'})."""
    return {t.get("label") for t in image_node.findall("tag")}


def polygons(image_node, label: str) -> list[np.ndarray]:
    """All polygons with a given label, as a list of (N,2) arrays."""
    return [parse_points(p.get("points")) for p in image_node.findall("polygon") if p.get("label") == label]


def seal_outer_inner(image_node):
    """Return (outer, inner) seal polygons sorted largest-first, or None.

    The seal is two 'sellado' polygons; the bigger-area one is the outer flange
    edge, the smaller is the inner well edge.
    """
    pl = polygons(image_node, "sellado")
    if len(pl) < 2:
        return None
    pl = sorted(pl, key=lambda q: cv2.contourArea(q.astype(np.float32)), reverse=True)
    return pl[0], pl[1]


def iter_images(xml_path: str):
    """Yield every <image> node in a CVAT XML."""
    return ET.parse(xml_path).getroot().findall("image")


def points_str(poly: np.ndarray) -> str:
    """(N,2) array -> 'x1,y1;x2,y2;...' with 2 decimals (CVAT format)."""
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in poly)


def write_seal_xml(path: str, rows: list[tuple]):
    """Write a minimal CVAT-1.1 file of seal pre-annotations.

    `rows` is a list of (image_name, width, height, [outer, inner]) tuples;
    polygons may be empty when the model produced no ring.
    """
    parts = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>"]
    for name, w, h, polys in rows:
        parts.append(f'  <image name="{name}" width="{w}" height="{h}">')
        for poly in polys:
            parts.append(f'    <polygon label="sellado" source="auto" occluded="0" '
                         f'points="{points_str(poly)}" z_order="0"></polygon>')
        parts.append("  </image>")
    parts.append("</annotations>")
    with open(path, "w") as f:
        f.write("\n".join(parts) + "\n")
