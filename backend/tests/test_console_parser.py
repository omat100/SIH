import math

from app import _extract_console_reading


def test_extract_console_reading_parses_sensor_block():
    block = [
        "==== SENSOR DATA RECEIVED ====",
        "Temperature: 28.9 °C",
        "Humidity: 26.1 %",
        "Accel X: -0.08 g",
        "Accel Y: 0.97 g",
        "Accel Z: 0.17 g",
        "Distance: 21.93 cm",
        "==============================",
    ]

    sample = _extract_console_reading(block)
    assert sample is not None
    assert sample["temp_c"] == 28.9
    assert sample["humidity_pct"] == 26.1
    assert sample["distance_mm"] == 219.3
    assert sample["tilt_deg"] > 0
    expected = math.degrees(math.atan2(math.sqrt((-0.08) ** 2 + (0.97) ** 2), 0.17))
    assert abs(sample["tilt_deg"] - expected) < 1e-6


def test_extract_console_reading_ignores_banner_and_status_lines():
    block = [
        "==== SENSOR DATA RECEIVED ====",
        "Status: OK",
        "Gyro: stable",
        "MPU Temp: 31.5°C",
        "Temperature: 28.9 °C",
        "Humidity: 26.1 %",
        "Accel X: -0.08 g",
        "Accel Y: 0.97 g",
        "Accel Z: 0.17 g",
        "Distance: 21.93 cm",
        "==============================",
    ]

    sample = _extract_console_reading(block)
    assert sample is not None
    assert sample["temp_c"] == 28.9
    assert sample["humidity_pct"] == 26.1
    assert sample["distance_mm"] == 219.3


def test_extract_console_reading_rejects_incomplete_block():
    block = [
        "Temperature: 28.9 °C",
        "Humidity: 26.1 %",
        "Accel X: -0.08 g",
        "Accel Y: 0.97 g",
        "Accel Z: 0.17 g",
    ]

    assert _extract_console_reading(block) is None
