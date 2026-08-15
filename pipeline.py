
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    detections: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    inference_time_ms: float = 0.0
    frame_id: Optional[int] = None
    camera_id: Optional[str] = None
    timestamp: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "detections": self.detections,
            "metadata": self.metadata,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
        }

class BasePipeline(ABC):

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_initialized = False
        self._frame_count = 0
        self._pool = None
        self._stream_id = None
        self._camera_id: Any = None
        self._tenant: str = ""
        self._engines: Dict[str, Any] = {}

    @abstractmethod
    def setup_models(self, registry, pool=None) -> None:
        pass

    @abstractmethod
    def process_image(self, image: np.ndarray) -> PipelineResult:
        pass

    def process_video_frame(self, image: np.ndarray, stream_id: str = "default") -> PipelineResult:
        self._frame_count += 1
        self._stream_id = stream_id
        return self.process_image(image)

    def process_batch(self, images: List[np.ndarray]) -> List[PipelineResult]:
        return [self.process_image(img) for img in images]

    def set_camera(self, camera_id: str, tenant: str = "") -> None:
        self._camera_id = camera_id
        self._tenant = (tenant or "").lower()

    def draw(self, image: np.ndarray, result: PipelineResult) -> np.ndarray:
        img = image.copy()
        for det in result.detections:
            box = det.get("box") or det.get("bbox")
            if box is None:
                continue
            x1, y1, x2, y2 = map(int, box)
            status = det.get("status", det.get("class", ""))
            conf = det.get("confidence", det.get("conf", 0.0))
            color = self._status_color(status)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{status} {conf:.0%}" if conf else status
            if det.get("track_id"):
                label = f"ID{det['track_id']} {label}"
            self._put_label(img, label, x1, y1, color)
        return img

    def reset(self) -> None:
        self._frame_count = 0

    def initialize(self, registry, pool=None) -> None:
        self._pool = pool
        self.setup_models(registry, pool)
        self.is_initialized = True
        logger.info(f"Pipeline '{self.name}' initialized (pool={'yes' if pool else 'no'})")

    def _safe_detect(self, model_name: str, image, **kwargs):
        if self._pool:
            locked = self._pool.get_detector(model_name)
            return locked.detect(image, **kwargs)
        return self._engines[model_name].detect(image, **kwargs)

    def _safe_track(self, model_name: str, image, stream_id: str = None, **kwargs):
        sid = stream_id or self._stream_id or "default"
        if self._pool:
            locked = self._pool.get_tracker(model_name, stream_id=sid)
            return locked.track(image, **kwargs)
        return self._engines[model_name].track(image, **kwargs)

    def _safe_classify(self, model_name: str, crops, **kwargs):
        if self._pool:
            locked = self._pool.get_classifier(model_name)
            return locked.classify_batch(crops, **kwargs)
        return self._engines[model_name].classify_batch(crops, **kwargs)

    @staticmethod
    def compute_iou(box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0

    @staticmethod
    def box_in_roi(box, roi) -> float:
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
        ox1, oy1 = max(x1, rx1), max(y1, ry1)
        ox2, oy2 = min(x2, rx2), min(y2, ry2)
        if ox1 >= ox2 or oy1 >= oy2:
            return 0.0
        return ((ox2 - ox1) * (oy2 - oy1)) / max(((x2 - x1) * (y2 - y1)), 1)

    @staticmethod
    def point_in_polygon(point, polygon) -> bool:
        x, y = point
        n, inside = len(polygon), False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @staticmethod
    def crop_with_mask(image, mask, box, padding=0.03):
        h, w = image.shape[:2]

        def _box_crop():
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            return image[y1:y2, x1:x2].copy() if x2 > x1 + 10 and y2 > y1 + 10 else None

        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask > 0.4).astype(np.uint8)
        if np.sum(mask_bin) < 400:
            return _box_crop()
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return _box_crop()
        xm, ym, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if cw < 20 or ch < 20:
            return _box_crop()
        pad = int(max(ch, cw) * padding)
        ym, xm = max(0, ym - pad), max(0, xm - pad)
        ye, xe = min(h - 1, ym + ch + pad), min(w - 1, xm + cw + pad)
        mc = mask_bin[ym:ye + 1, xm:xe + 1]
        ic = image[ym:ye + 1, xm:xe + 1]
        m3 = np.stack([mc] * 3, axis=-1).astype(bool)
        bg = ~m3[:, :, 0]
        bgc = np.median(ic[bg], axis=0).astype(np.uint8) if np.sum(bg) > 50 and len(ic[bg]) > 0 else np.array([220, 220, 220], dtype=np.uint8)
        return np.where(m3, ic, np.full_like(ic, bgc)).astype(np.uint8)

    @staticmethod
    def check_mask_overlap(table_mask, person_data, img_shape, threshold=0.05):
        h, w = img_shape[:2]
        if table_mask.shape != (h, w):
            table_mask = cv2.resize(table_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        tb = (table_mask > 0.3).astype(bool)
        ta = np.sum(tb)
        if ta == 0:
            return False
        for item in person_data:
            if isinstance(item, tuple) and len(item) == 2:
                box, mask = item
                if mask is not None:
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    if np.sum(np.logical_and(tb, (mask > 0.3).astype(bool))) / ta > threshold:
                        return True
                else:
                    yi, xi = np.where(tb)
                    if len(yi) == 0:
                        continue
                    tx1, ty1, tx2, ty2 = xi.min(), yi.min(), xi.max(), yi.max()
                    px1, py1, px2, py2 = box
                    ix1, iy1 = max(tx1, px1), max(ty1, py1)
                    ix2, iy2 = min(tx2, px2), min(ty2, py2)
                    if ix1 < ix2 and iy1 < iy2:
                        t_area = (tx2 - tx1) * (ty2 - ty1)
                        if t_area > 0 and (ix2 - ix1) * (iy2 - iy1) / t_area > threshold:
                            return True
        return False

    @staticmethod
    def _status_color(status):
        return {
            "TOZA": (0, 255, 0), "clean": (0, 255, 0), "TOZA EMAS": (0, 0, 255),
            "dirty": (0, 0, 255), "BAND": (0, 165, 255), "in_use": (0, 165, 255),
            "NOANIQ": (0, 255, 255), "unknown": (0, 255, 255), "farrosh": (0, 255, 0),
            "person": (0, 165, 255), "full": (0, 255, 0), "not_full": (0, 0, 255),
            "glove": (0, 255, 0), "no_glove": (0, 0, 255),
        }.get(status, (255, 255, 255))

    @staticmethod
    def _put_label(img, label, x, y, color):
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y), color, -1)
        cv2.putText(img, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
