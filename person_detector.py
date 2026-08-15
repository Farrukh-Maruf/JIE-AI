from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

class _TensorLike:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def __len__(self):
        return len(self._arr)

class _Boxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = _TensorLike(xyxy.astype(np.float32))
        self.conf = _TensorLike(conf.astype(np.float32))
        self.cls = _TensorLike(cls.astype(np.float32))
        self.id = None

    def __len__(self):
        return int(len(self.xyxy))

class _Result:
    def __init__(self, boxes: Optional[_Boxes], orig_shape: Tuple[int, int]):
        self.boxes = boxes
        self.masks = None
        self.keypoints = None
        self.probs = None
        self.orig_shape = orig_shape

def _read_engine_bytes(engine_path: str) -> bytes:
    with open(engine_path, "rb") as f:
        data = f.read()
    try:
        import json as _json
        meta_len = int.from_bytes(data[:4], byteorder="little")
        if 0 < meta_len < 10_000_000 and (4 + meta_len) < len(data):
            try:
                _json.loads(data[4:4 + meta_len].decode("utf-8"))
                return data[4 + meta_len:]
            except Exception:
                pass
    except Exception:
        pass
    return data

class PersonEngineDetector:

    def __init__(self, engine_path: str, input_size: int = 640, device: int = 0):
        import atexit
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        self.engine_path = str(engine_path)
        self.input_size = int(input_size)
        self.device = int(device)
        self._lock = threading.Lock()

        if not Path(self.engine_path).exists():
            raise FileNotFoundError(self.engine_path)

        raw = _read_engine_bytes(self.engine_path)
        logger_trt = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger_trt)
        self._engine = self._runtime.deserialize_cuda_engine(raw)
        if self._engine is None:
            raise RuntimeError(f"engine deserialize qaytardi: None ({self.engine_path})")
        self._exec = self._engine.create_execution_context()

        self._inputs, self._outputs = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            shape = list(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            if mode == trt.TensorIOMode.INPUT:
                self._inputs.append((name, shape, dtype))
            else:
                self._outputs.append((name, shape, dtype))

        atexit.register(self._release)

    def _release(self):
        self._exec = None
        self._engine = None
        self._runtime = None

    def _preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        h, w = bgr.shape[:2]
        s = min(self.input_size / h, self.input_size / w)
        nh, nw = int(round(h * s)), int(round(w * s))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_top = (self.input_size - nh) // 2
        pad_left = (self.input_size - nw) // 2
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None]
        return blob, s, pad_left, pad_top

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda
        in_name, _, in_dtype = self._inputs[0]
        data = np.ascontiguousarray(blob.astype(in_dtype))
        d_in = cuda.mem_alloc(data.nbytes)
        cuda.memcpy_htod(d_in, data)
        self._exec.set_tensor_address(in_name, int(d_in))

        out_bufs, d_outs = [], []
        for (on, _, od) in self._outputs:
            shp = [int(s) for s in self._exec.get_tensor_shape(on)]
            arr = np.empty(shp, dtype=od)
            d_out = cuda.mem_alloc(arr.nbytes)
            self._exec.set_tensor_address(on, int(d_out))
            out_bufs.append(arr); d_outs.append(d_out)

        stream = cuda.Stream()
        self._exec.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        for a, d in zip(out_bufs, d_outs):
            cuda.memcpy_dtoh(a, d)
        d_in.free()
        for d in d_outs:
            d.free()
        return out_bufs[0]

    def predict(
        self,
        source,
        conf: float = 0.25,
        classes: Optional[List[int]] = None,
        verbose: bool = False,
        **_: dict,
    ) -> List[_Result]:
        if isinstance(source, np.ndarray):
            images = [source]
        elif isinstance(source, list):
            images = source
        else:
            raise TypeError(f"source type qo'llab quvvatlanmaydi: {type(source)}")

        results: List[_Result] = []
        with self._lock:
            for bgr in images:
                blob, s, pad_l, pad_t = self._preprocess(bgr)
                raw = self._infer(blob)
                dets = raw.reshape(-1, raw.shape[-1])

                mask = dets[:, 4] >= float(conf)
                if classes is not None:
                    mask &= np.isin(dets[:, 5].astype(int), list(classes))
                sel = dets[mask]

                if len(sel) == 0:
                    results.append(_Result(None, orig_shape=bgr.shape[:2]))
                    continue

                xyxy = sel[:, :4].copy()
                xyxy[:, [0, 2]] -= pad_l
                xyxy[:, [1, 3]] -= pad_t
                xyxy /= max(s, 1e-9)
                h, w = bgr.shape[:2]
                xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w - 1)
                xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h - 1)

                boxes = _Boxes(xyxy=xyxy, conf=sel[:, 4], cls=sel[:, 5])
                results.append(_Result(boxes, orig_shape=bgr.shape[:2]))
        return results

    def __call__(self, source, **kwargs):
        return self.predict(source, **kwargs)
