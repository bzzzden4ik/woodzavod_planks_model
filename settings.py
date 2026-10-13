"""
Чтение настроек из config.toml.

Проверить настройки:
    .venv\\Scripts\\python.exe settings.py
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass
class Settings:
    marker_length_cm: float
    normal_length_cm: float          # N
    tolerance_percent: float
    model_path: Path
    model_url: str
    model_sha256: str
    device: str
    every_n_frames: int
    calibration_path: Path
    machine_entry_x: float
    machine_entry_y: float
    length_correction_cm: float

    @property
    def outlier_limit_cm(self) -> float:
        """Доска длиной меньше или равной этому значению — выброс."""
        return self.normal_length_cm * (1 - self.tolerance_percent / 100)

    def problems(self) -> list[str]:
        result = []
        if self.marker_length_cm <= 0:
            result.append("marker.length_cm должна быть больше 0")
        if self.normal_length_cm <= 0:
            result.append("boards.normal_length_cm (N) должна быть больше 0")
        if not 0 < self.tolerance_percent < 100:
            result.append("boards.tolerance_percent должна быть от 0 до 100")
        if self.every_n_frames < 1:
            result.append("model.every_n_frames должна быть 1 или больше")
        if not self.model_path.exists():
            hint = (r"запустите: .venv\Scripts\python.exe download_model.py" if self.model_url
                    else "скачайте best.pt из GitHub Releases в папку models (README.md, раздел 2)")
            result.append(f"нет файла модели {self.model_path} — {hint}")
        if not self.calibration_path.exists():
            result.append(f"нет файла калибровки {self.calibration_path}")
        return result


def load_settings(path: Path = HERE / "config.toml") -> Settings:
    with open(path, "rb") as f:
        d = tomllib.load(f)
    return Settings(
        marker_length_cm=float(d["marker"]["length_cm"]),
        normal_length_cm=float(d["boards"]["normal_length_cm"]),
        tolerance_percent=float(d["boards"]["tolerance_percent"]),
        model_path=HERE / d["model"]["path"],
        model_url=str(d["model"].get("url", "")).strip(),
        model_sha256=str(d["model"].get("sha256", "")).strip().lower(),
        device=str(d["model"].get("device", "auto")).strip(),
        every_n_frames=int(d["model"]["every_n_frames"]),
        calibration_path=HERE / d["geometry"]["calibration"],
        machine_entry_x=float(d["geometry"]["machine_entry_x"]),
        machine_entry_y=float(d["geometry"]["machine_entry_y"]),
        length_correction_cm=float(d["geometry"]["length_correction_cm"]),
    )


if __name__ == "__main__":
    st = load_settings()
    print(f"Маркер:          {st.marker_length_cm} см")
    print(f"N:               {st.normal_length_cm} см, погрешность {st.tolerance_percent} %")
    print(f"Выброс, если длина <= {st.outlier_limit_cm:.1f} см")
    print(f"Модель:          {st.model_path} (каждый {st.every_n_frames}-й кадр)")
    from detector import describe_device
    print(f"Нейросеть на:    {describe_device(st.device)}")
    print(f"Калибровка:      {st.calibration_path}")
    print(f"Вход в станок:   x={st.machine_entry_x:.0f}, y={st.machine_entry_y:.0f}")
    probs = st.problems()
    for p in probs:
        print(f"ВНИМАНИЕ: {p}")
    if not probs:
        print("Настройки в порядке.")
