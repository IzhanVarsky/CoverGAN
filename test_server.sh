curl --progress-bar \
    -F "audio_file=@test_music.mp3" \
    "http://localhost:5001/generate?track_artist=Cool%20Band&track_name=Song&emotion=joy" \
    -o output1.json

curl --progress-bar \
    -F "audio_file=@test_music.mp3" \
    "http://localhost:5001/generate?track_artist=Cool%20Band&track_name=Song&emotion=joy&gen_type=1&use_captioner=True" \
    -o output2.json

curl --progress-bar \
    -F "audio_file=@test_music.mp3" \
    "http://localhost:5001/generate?track_artist=Cool%20Band&track_name=Song&emotion=joy&gen_type=2&use_captioner=True" \
    -o output3.json

curl --progress-bar \
    -F "audio_file=@test_music.mp3" \
    "http://localhost:5001/generate?track_artist=Cool%20Band&track_name=Song&emotion=joy&gen_type=1" \
    -o output4.json