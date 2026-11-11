"""Task 4.2 - upload validation. Nothing is written to disk; bytes stay in memory."""
import io

from PIL import Image
from werkzeug.utils import secure_filename

Image.MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


class UploadError(ValueError):
    pass


def validate_upload(file_storage):
    if file_storage is None or not file_storage.filename:
        raise UploadError("No file was selected.")
    name = secure_filename(file_storage.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXT:
        raise UploadError(f"Unsupported file type '.{ext}'. Please upload PNG, JPG or WEBP.")
    data = file_storage.read()
    if not data:
        raise UploadError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()  # full decode: catches truncated / fake images
    except Exception as e:  # noqa: BLE001
        raise UploadError("This file has an image extension but could not be decoded as an image.") from e
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(f"Image content is {fmt}, which is not supported.")
    return name, data
