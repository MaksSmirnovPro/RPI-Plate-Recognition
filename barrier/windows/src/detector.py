
import cv2
import numpy as np
import onnxruntime as ort


class PlateDetector:
    def __init__(
        self,
        model_path,
        input_size=640,
        confidence=0.45,
        nms_threshold=0.45,
        output_format="auto",
    ):
        self.input_size = input_size
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.output_format = output_format

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        print("ONNX model:", model_path)
        print("Input:", self.session.get_inputs()[0].shape)
        print(
            "Outputs:",
            [item.shape for item in self.session.get_outputs()],
        )

    def detect(self, frame):
        height, width = frame.shape[:2]

        scale = min(
            self.input_size / width,
            self.input_size / height,
        )

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = (self.input_size - new_width) // 2
        pad_y = (self.input_size - new_height) // 2

        canvas = np.full(
            (self.input_size, self.input_size, 3),
            114,
            dtype=np.uint8,
        )

        canvas[
            pad_y:pad_y + new_height,
            pad_x:pad_x + new_width,
        ] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )

        outputs = self.session.run(
            None,
            {self.input_name: blob},
        )

        if not outputs:
            return []

        predictions = np.asarray(outputs[0])

        results = self._parse(
            predictions,
            width,
            height,
            scale,
            pad_x,
            pad_y,
        )

        print(
            f"Detection: frame={width}x{height}, "
            f"boxes={results}"
        )

        return results

    def _parse(
        self,
        output,
        original_width,
        original_height,
        scale,
        pad_x,
        pad_y,
    ):
        output = np.squeeze(output)

        if output.ndim != 2:
            raise ValueError(
                f"Unsupported ONNX output shape: {output.shape}"
            )

        # Автоматическое определение распространённых форматов.
        fmt = self.output_format

        if fmt == "auto":
            if output.shape[-1] == 6:
                fmt = "nms"
            elif output.shape[0] == 6:
                fmt = "nms"
            else:
                fmt = "yolo"

        if fmt == "nms":
            rows = output

            if rows.shape[0] == 6 and rows.shape[1] != 6:
                rows = rows.T

            if rows.shape[1] != 6:
                raise ValueError(
                    f"Expected Nx6 NMS output, got {rows.shape}"
                )

            candidates = []

            for row in rows:
                x1, y1, x2, y2, score, _class_id = row

                if score < self.confidence:
                    continue

                candidates.append((
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(score),
                ))

        elif fmt == "yolo":
            # YOLO: [4 + class_count, N] или [N, 4 + class_count].
            if output.shape[0] <= output.shape[1]:
                rows = output.T
            else:
                rows = output

            if rows.shape[1] < 5:
                raise ValueError(
                    f"Unsupported YOLO output: {rows.shape}"
                )

            candidates = []

            for row in rows:
                cx, cy, box_w, box_h = row[:4]
                score = float(np.max(row[4:]))

                if score < self.confidence:
                    continue

                candidates.append((
                    float(cx - box_w / 2),
                    float(cy - box_h / 2),
                    float(cx + box_w / 2),
                    float(cy + box_h / 2),
                    score,
                ))

        else:
            raise ValueError(f"Unknown output format: {fmt}")

        boxes = []
        scores = []

        for x1, y1, x2, y2, score in candidates:
            boxes.append([
                int(x1),
                int(y1),
                max(1, int(x2 - x1)),
                max(1, int(y2 - y1)),
            ])
            scores.append(score)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return []

        results = []

        for index in np.asarray(indices).flatten():
            x, y, w, h = boxes[int(index)]

            x1 = int((x - pad_x) / scale)
            y1 = int((y - pad_y) / scale)
            x2 = int((x + w - pad_x) / scale)
            y2 = int((y + h - pad_y) / scale)

            x1 = max(0, min(x1, original_width - 1))
            y1 = max(0, min(y1, original_height - 1))
            x2 = max(0, min(x2, original_width))
            y2 = max(0, min(y2, original_height))

            if x2 > x1 and y2 > y1:
                results.append((
                    x1,
                    y1,
                    x2,
                    y2,
                    float(scores[int(index)]),
                ))

        return results
