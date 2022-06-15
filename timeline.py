import base64
import json
import pathlib
from io import BytesIO

import imageio.v3 as iio
import numpy as np
import streamlit as st
from streamlit_terran_timeline import terran_timeline

video_path = pathlib.Path(st.text_input("", "data/videos/BA_GNS.mp4"))


@st.cache(persist=True, ttl=86_400, suppress_st_warning=True, show_spinner=False)
def _generate_timeline(video_path):
    frame_rate = 1

    video_id = video_path.stem

    frame_count = len(list(iio.imiter(video_path, fps=frame_rate)))

    result_path = f"{video_id}.json"

    with open(result_path) as f:
        face_groups = json.load(f)

    appearance = {}
    for face_group_id, face_group in enumerate(face_groups):
        appearance[str(face_group_id)] = []
        for frame_id in range(frame_count):
            appear = False
            for face in face_group:
                if face["image_id"] == str(frame_id):
                    appear = True
            appearance[str(face_group_id)].append(appear)

    track_faces = {}
    for face_group_id, face_group in enumerate(face_groups):
        buffer = BytesIO()
        iio.imwrite(buffer, np.array(face_group[0]["thumbnail"]).astype(np.uint8), format="png")
        track_faces[str(face_group_id)] = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "id": video_id,
        "url": str(video_path),
        "appearance": appearance,
        "track_ids": list(appearance.keys()),
        "framerate": frame_rate,
        "start_time": 0,
        "end_time": frame_count,
        "track_faces": track_faces,
        "thumbnail_rate": None,
        "thumbnails": []
    }


with st.spinner("Generating timeline"):
    timeline = _generate_timeline(video_path)

start_time = terran_timeline(timeline)

st.write(f"User clicked on second {int(start_time)}")

st.video(str(video_path), start_time=int(start_time))
