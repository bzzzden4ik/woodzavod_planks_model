r"""
Скачать обученную модель (models/best.pt) из GitHub Releases.

Ссылка и контрольная сумма берутся из config.toml (раздел [model]).
Запуск (из папки machine_learning):
    .venv\Scripts\python.exe download_model.py
    .venv\Scripts\python.exe download_model.py --force      (скачать заново, даже если файл есть)
"""

import argparse
import hashlib
import sys
import urllib.request

from settings import load_settings


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, expected_sha256=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Скачиваю {url}")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            print(f"\r  {done / 1e6:5.1f} из {total / 1e6:.1f} МБ", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, progress)
    print()
    if expected_sha256:
        got = sha256_of(tmp)
        if got != expected_sha256:
            tmp.unlink()
            raise SystemExit(f"ОШИБКА: скачался не тот файл (контрольная сумма {got}, "
                             f"ожидалась {expected_sha256}). Проверьте model.url и model.sha256 в config.toml.")
    tmp.replace(path)
    print(f"Готово: {path}")


def main():
    ap = argparse.ArgumentParser(description="Скачать модель из GitHub Releases")
    ap.add_argument("--force", action="store_true", help="скачать заново, даже если файл есть")
    args = ap.parse_args()
    st = load_settings()
    if st.model_path.exists() and not args.force:
        ok = not st.model_sha256 or sha256_of(st.model_path) == st.model_sha256
        print(f"Модель уже есть: {st.model_path}" + ("" if ok else
              "\nВНИМАНИЕ: контрольная сумма не совпадает с config.toml — это другая версия модели. "
              "Скачать нужную: download_model.py --force"))
        return
    if not st.model_url:
        sys.exit("ОШИБКА: в config.toml не указана ссылка model.url. "
                 "Скачайте best.pt со страницы Releases вручную в папку models (README.md, раздел 2).")
    download(st.model_url, st.model_path, st.model_sha256)


if __name__ == "__main__":
    main()
