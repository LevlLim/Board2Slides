"""Display-only interaction and faithful reconstruction of existing OCR crops."""
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit.components.v1 as components

region_component = components.declare_component(
    'b2n_region_viewer', path=str(Path(__file__).parent / 'region_viewer'))


def region_photo(image, lines, selected, generation=0):
    preview = image.copy()
    preview.thumbnail((1400, 1400))  # display only; coordinates stay in source space
    buffer = BytesIO()
    preview.save(buffer, 'JPEG', quality=90)
    return region_component(image='data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode(),
                            width=image.width, height=image.height, lines=lines,
                            selected=selected, default=None, key=f'region_photo_{generation}')


def recognition_crop(image, box):
    """Same quadrilateral extraction as OCRService, before reader resizing."""
    from preprocessing.text_lines import quad_crop
    bgr = np.asarray(image.convert('RGB'))[:, :, ::-1].copy()
    crop = quad_crop(bgr, np.asarray(box, dtype=np.float32))
    return Image.fromarray(crop[:, :, ::-1])
