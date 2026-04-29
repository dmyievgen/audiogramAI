# Audiogram

Desktop додаток для macOS: завантажуєш аудіотрек (drag&drop або вибір з диска), бачиш ноти в часі (CQT-спектрограма з нотними мітками по Y) і можеш відтворити трек з лінією прогресу.

## Стек

- Python 3.10+
- PyQt6 — GUI
- pyqtgraph — швидке малювання спектрограми та плейхеду
- librosa — CQT та робота з аудіо
- soundfile — читання файлів
- sounddevice — програвання

## Запуск

```bash
cd audiogram
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python audiogram/__main__.py
```

(Викликати з кореня репозиторію або з папки `audiogram/` — обидва варіанти працюють завдяки `sys.path` тюнінгу в `__main__.py`.)

Перетягни `.wav`, `.mp3`, `.flac` тощо у вікно або натисни **Open…**, тоді **Play**.

## Архітектура

```
audiogram/
├── __main__.py              # entry point: python -m audiogram
└── src/audiogram/
    ├── app.py               # bootstrap QApplication + MainWindow
    ├── audio/
    │   ├── loader.py        # завантаження файлу → AudioTrack (моно float32 + sr)
    │   ├── analysis.py      # CQT спектрограма + нотні мітки осі
    │   └── player.py        # відтворення через sounddevice + позиція
    ├── ui/
    │   ├── main_window.py   # drop zone, тулбар, статус
    │   └── spectrogram_view.py  # pyqtgraph + плейхед
    └── core/
        ├── models.py        # dataclass-и AudioTrack, Spectrogram
        └── notes.py         # утиліти MIDI ↔ ім'я ноти
```

Кожен шар не знає про вищий: `audio/*` не імпортує `ui/*`, `core/*` чистий від Qt. Це потрібно, щоб далі легко додавати:

- детекцію акордів / темпу,
- виділення сегментів,
- експорт MIDI,
- альтернативні бекенди (Tone.js, web).
