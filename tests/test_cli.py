import json
import sys

import cv2
import numpy as np
import pytest

from docscannerocr import cli


def _write_simple_image(path):
    image = np.full((150, 500, 3), 255, dtype=np.uint8)
    cv2.putText(image, "RECEIPT", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.imwrite(str(path), image)


@pytest.mark.network
def test_cli_prints_json_to_stdout(tmp_path, capsys, monkeypatch):
    image_path = tmp_path / "receipt.png"
    _write_simple_image(image_path)
    monkeypatch.setattr(sys, "argv", ["doc-scanner-ocr", str(image_path)])

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == str(image_path)
    assert any("RECEIPT" in line["text"].upper() for line in payload["lines"])


@pytest.mark.network
def test_cli_writes_output_file(tmp_path, monkeypatch):
    image_path = tmp_path / "receipt.png"
    _write_simple_image(image_path)
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["doc-scanner-ocr", str(image_path), "-o", str(out_path)])

    cli.main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source"] == str(image_path)


def test_cli_reports_missing_file_and_exits_nonzero(tmp_path, capsys, monkeypatch):
    missing = tmp_path / "nope.png"
    monkeypatch.setattr(sys, "argv", ["doc-scanner-ocr", str(missing)])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
    assert "Could not read image" in capsys.readouterr().err
