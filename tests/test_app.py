import io

from conftest import png_bytes


def post(client, name, data):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["database"] is True


def test_valid_upload_is_logged_and_shown_in_history(client):
    r = post(client, "cat.png", png_bytes())
    assert r.status_code == 200 and b"AI-generated" in r.data
    h = client.get("/history")
    assert b"cat.png" in h.data and b"1 request" in h.data


def test_wrong_extension_rejected(client):
    r = post(client, "notes.txt", b"hello")
    assert r.status_code == 400 and b"Unsupported file type" in r.data


def test_fake_image_rejected(client):
    r = post(client, "evil.png", b"this is not really a png")
    assert r.status_code == 400 and b"could not be decoded" in r.data


def test_oversized_rejected(client):
    r = post(client, "big.png", png_bytes() + b"0" * (300 * 1024))
    assert r.status_code == 413


def test_filename_is_sanitised(client):
    post(client, "../../etc/passwd.png", png_bytes())
    assert b"etc_passwd.png" in client.get("/history").data or b"passwd.png" in client.get("/history").data
    assert b".." not in client.get("/history").data.split(b"<tbody>")[1].split(b"</tbody>")[0]


def test_404_page(client):
    assert client.get("/nope").status_code == 404
