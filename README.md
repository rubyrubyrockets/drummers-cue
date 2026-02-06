# Drum Cues Generator (Streamlit Cloud)

## Что делает
- Загружаешь трек
- all-in-one-fix определяет структуру и BPM
- ADTOF транскрибирует барабаны в MIDI
- По секциям делаем подсказки: что добавилось/убралось (kick/snare/hat/...)
- Генерируем mp3-трек с голосовыми подсказками заранее (за N тактов)

## Streamlit Cloud
- packages.txt ставит ffmpeg
- Piper модель скачивается автоматически из HuggingFace при первом запуске (src/download_models.py)

## Важно про ADTOF
`adtof_plus_drum_transcription` иногда не ставится/не запускается в Streamlit Cloud.
Если упадёт — открой логи Cloud и пришли сюда текст ошибки.
Я:
- подправлю команду запуска в src/transcribe_adtof.py
- или предложу запасной вариант (например демикс + onsets + классификация) который 100% ставится в Cloud.
