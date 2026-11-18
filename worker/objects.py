"""CPU YOLOX-Nano object boxes for protecting portrait props."""

from pathlib import Path
import cv2
import numpy as np


def detect_objects(rgb):
    from runtime import session as model_session

    session = model_session(Path(__file__).with_name("models") / "yolox_nano.onnx")
    _, _, size, _ = session.get_inputs()[0].shape
    height, width = rgb.shape[:2]
    ratio = min(size / height, size / width)
    # The upstream ONNX export expects raw BGR pixels, letterboxed at the top left.
    resized = cv2.resize(rgb[:, :, ::-1], (int(width * ratio), int(height * ratio)))
    padded = np.full((size, size, 3), 114, np.float32)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    prediction = session.run(None, {
        session.get_inputs()[0].name: padded.transpose(2, 0, 1)[None].copy()
    })[0][0].copy()
    grids, scales = [], []
    for stride in (8, 16, 32):
        y, x = np.meshgrid(np.arange(size // stride), np.arange(size // stride), indexing="ij")
        grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
        scales.append(np.full((x.size, 1), stride))
    grid = np.concatenate(grids)
    stride = np.concatenate(scales)
    if prediction.shape != (len(grid), 85) or not np.isfinite(prediction).all():
        raise ValueError("Unexpected object detector output.")
    prediction[:, :2] = (prediction[:, :2] + grid) * stride
    prediction[:, 2:4] = np.exp(np.clip(prediction[:, 2:4], -10, 10)) * stride
    classes = prediction[:, 5:].argmax(axis=1)
    scores = prediction[:, 4] * prediction[np.arange(len(prediction)), 5 + classes]
    selected = np.flatnonzero(scores > .18)
    result = []
    for label in np.unique(classes[selected]):
        indexes = selected[classes[selected] == label]
        xywh = np.column_stack((prediction[indexes, :2] - prediction[indexes, 2:4] / 2,
                                prediction[indexes, 2:4])) / ratio
        kept = cv2.dnn.NMSBoxes(xywh.tolist(), scores[indexes].tolist(), .18, .45)
        for kept_index in np.asarray(kept).reshape(-1):
            index = indexes[kept_index]
            x, y, w, h = xywh[kept_index]
            box = [max(0, int(x)), max(0, int(y)),
                   min(width, int(np.ceil(x + w))), min(height, int(np.ceil(y + h)))]
            if box[2] > box[0] and box[3] > box[1]:
                result.append({"class": int(label), "confidence": float(scores[index]), "box": box})
    return result

