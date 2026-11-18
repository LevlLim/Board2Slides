"""Student-facing photo → editable notes flow. No model choice is required."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile

from PIL import Image
import streamlit as st

from ocr_backend import OCRService
from student_flow import decode_photo, draw_regions, prepare_photo
from visual_review import recognition_crop, region_photo


MAX_UPLOAD_MB = 30
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# Lowest original whiteboard-validation CER, rather than a choice on Jy test.
DEFAULT_MODEL = os.environ.get("B2N_OCR_MODEL", "S1")
SAMPLE_PHOTO = Path(__file__).resolve().parents[1] / "fixtures" / "photos" / "jy_photo23.jpg"   # packaged copy of Jy Photo 23
LOGGER = logging.getLogger(__name__)


@st.cache_resource(max_entries=1, show_spinner=False)
def load_service(model_id: str) -> OCRService:
    return OCRService(model_id=model_id)


def run_ocr(image: Image.Image) -> dict:
    """Read the preview's exact pixels, without retaining an upload on disk."""
    service = load_service(DEFAULT_MODEL)
    with tempfile.TemporaryDirectory(prefix="board2notes-ocr-") as directory:
        path = Path(directory) / "whiteboard.png"
        image.save(path, format="PNG")
        return service.recognise(str(path))


def initialise() -> None:
    if st.session_state.pop("_reset_requested", False):
        generation = st.session_state.get("upload_generation", 0) + 1
        for key in ("photo", "photo_name", "result", "result_photo", "notes", "draft_notes",
                    "selected_line", "show_regions", "replace_edits", "region_event"):
            st.session_state.pop(key, None)
        st.session_state.update(phase="add", upload_generation=generation, turns=0,
                                trim_edges=False, horizontal_area=(0, 100),
                                vertical_area=(0, 100), trim_enabled=False,
                                crop_x=(0, 100), crop_y=(0, 100))
    for key, value in {
        "phase": "add", "upload_generation": 0, "turns": 0, "result_generation": 0,
        "trim_enabled": False, "crop_x": (0, 100), "crop_y": (0, 100),
        "trim_edges": st.session_state.get("trim_enabled", False),
        "horizontal_area": st.session_state.get("crop_x", (0, 100)),
        "vertical_area": st.session_state.get("crop_y", (0, 100)),
    }.items():
        st.session_state.setdefault(key, value)
    if st.session_state["phase"] == "check":
        # Restore before creating widgets: a hidden widget's stale default may
        # still be present even after its durable value survived cleanup.
        st.session_state.update(trim_edges=st.session_state["trim_enabled"],
                                horizontal_area=st.session_state["crop_x"],
                                vertical_area=st.session_state["crop_y"])


def accept_photo(data: bytes, name: str) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Please choose a photo smaller than 30 MB.")
    image = decode_photo(data)
    if min(image.size) < 16:
        raise ValueError("This photo is too small. Choose the original camera photo.")
    st.session_state.update(photo=image, photo_name=Path(name).name, phase="check",
                            turns=0, trim_edges=False,
                            horizontal_area=(0, 100), vertical_area=(0, 100),
                            trim_enabled=False, crop_x=(0, 100), crop_y=(0, 100))


def reset_photo() -> None:
    # Apply next pass, before controls exist, to respect widget state ownership.
    st.session_state["_reset_requested"] = True


def rotate_photo(turns: int) -> None:
    st.session_state["turns"] = (st.session_state["turns"] + turns) % 4
    # Percentage coordinates belong to the rotated image.
    st.session_state.update(horizontal_area=(0, 100), vertical_area=(0, 100),
                            crop_x=(0, 100), crop_y=(0, 100), replace_edits=False)


def save_crop() -> None:
    st.session_state.update(crop_x=st.session_state["horizontal_area"],
                            crop_y=st.session_state["vertical_area"])


def save_trim() -> None:
    st.session_state["trim_enabled"] = st.session_state["trim_edges"]


def has_edits() -> bool:
    result = st.session_state.get("result")
    return result is not None and current_notes() != str(result.get("text", ""))


def current_notes() -> str:
    return st.session_state.get("notes", st.session_state.get("draft_notes", ""))


def save_notes() -> None:
    # Widget keys are cleaned up when their page is no longer rendered.
    st.session_state["draft_notes"] = st.session_state.get("notes", "")


@st.dialog("Start another note?")
def confirm_new_photo() -> None:
    st.write("You have edited these notes. Save a copy before starting another photo.")
    st.download_button("Save my notes", current_notes(),
                       file_name="whiteboard_notes.txt", mime="text/plain", on_click="ignore")
    keep, leave = st.columns(2)
    if keep.button("Keep editing", width="stretch"):
        st.session_state["phase"] = "review"
        st.rerun()
    if leave.button("Start another note", type="primary", width="stretch"):
        reset_photo()
        st.rerun()


def new_photo() -> None:
    if has_edits():
        confirm_new_photo()
    else:
        reset_photo()
        st.rerun()


def notes_actions(text: str) -> None:
    """Export the live editor value, including edits not yet sent to Python.

    Streamlit sends textarea changes on blur. Reading our own visible textarea
    at the click avoids exporting a stale value during that round trip. The
    encoded server value is the fallback if parent DOM access is unavailable.
    """
    encoded = json.dumps(text, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    st.iframe("""
    <style>
      body{margin:0;font:15px system-ui,sans-serif;color:#334155}
      .actions{display:flex;gap:12px;padding:3px}
      button{flex:1;min-height:42px;border:1px solid #cbd5e1;background:white;color:#334155;
        border-radius:9px;font:600 15px system-ui,sans-serif;cursor:pointer}
      #copy{background:#0f766e;color:white;border-color:#0f766e}
      button:hover{filter:brightness(.95)}button:focus-visible{outline:3px solid #f59e0b;outline-offset:2px}
      #message{font-size:12px;margin:5px 3px 0;min-height:18px}
    </style>
    <div class="actions"><button id="copy" type="button">Copy notes</button>
      <button id="download" type="button">Download text</button></div>
    <p id="message" role="status" aria-live="polite"></p>
    <script>
      const savedNotes = """ + encoded + """;
      const message = document.getElementById('message');
      function currentNotes() {
        try {
          const editor = parent.document.querySelector('textarea[aria-label="Edit your notes"]');
          if (editor) return editor.value;
        } catch (_) {}
        return savedNotes;
      }
      document.getElementById('copy').addEventListener('click', async () => {
        const notes = currentNotes();
        try {
          try { await navigator.clipboard.writeText(notes); }
          catch (_) {
            const field = document.createElement('textarea');
            field.value=notes;field.style.position='fixed';field.style.opacity='0';
            document.body.appendChild(field);field.select();
            const ok=document.execCommand('copy');field.remove();
            if (!ok) throw new Error('Clipboard unavailable');
          }
          message.textContent='Copied. Ready to paste into your notes.';
        } catch (_) {
          message.textContent='Select the notes and copy, or use Download text.';
        }
      });
      document.getElementById('download').addEventListener('click', () => {
        const url=URL.createObjectURL(new Blob([currentNotes()], {type:'text/plain;charset=utf-8'}));
        const link=document.createElement('a');link.href=url;link.download='whiteboard_notes.txt';
        document.body.appendChild(link);link.click();link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
        message.textContent='Text file prepared. Check your browser’s downloads.';
      });
    </script>
    """, height=86, alt="Copy or download your edited notes")


def header() -> None:
    st.markdown("""<style>
      .stApp{background:#f7f9fc}
      .block-container{max-width:1180px;padding-top:4rem;padding-bottom:3rem}
      h1{letter-spacing:-.045em!important}h2,h3{letter-spacing:-.02em!important}
      .b2n-brand{font:750 18px system-ui;color:#0f766e;letter-spacing:-.03em;margin-bottom:18px}
      .b2n-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;list-style:none;padding:0;margin:0}
      .b2n-step{padding:10px 12px;border-bottom:3px solid #cbd5e1;color:#64748b;font-size:14px}
      .b2n-step.active{border-color:#0f766e;color:#0f766e;font-weight:650}
      [data-testid="stTextArea"] textarea{font-size:17px;line-height:1.65}
      [data-testid="stBaseButton-primary"]{background:#0f766e;border-color:#0f766e}
      [data-testid="stBaseButton-primary"]:hover{background:#115e59;border-color:#115e59}
      [data-testid="stFileUploader"]{background:white;border-radius:12px}
      @media(max-width:640px){.block-container{padding-top:4rem}.b2n-steps{gap:6px}.b2n-step{padding:8px 2px;font-size:12px}}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="b2n-brand">✎ &nbsp; Board2Notes</div>', unsafe_allow_html=True)
    labels = [("add", "1 · Add photo"), ("check", "2 · Check photo"), ("review", "3 · Review notes")]
    steps = "".join(f'<li class="b2n-step{" active" if st.session_state["phase"] == phase else ""}"'
                    f'{" aria-current=step" if st.session_state["phase"] == phase else ""}>{label}</li>'
                    for phase, label in labels)
    st.markdown('<nav aria-label="Your progress"><ol class="b2n-steps">' + steps + '</ol></nav>', unsafe_allow_html=True)
    st.write("")


def add_photo_page() -> None:
    st.title("Your whiteboard, ready to study.")
    st.write("Turn a whiteboard photo into text you can check, edit and keep.")
    upload_col, help_col = st.columns([1.7, 1], gap="large")
    with upload_col:
        upload = st.file_uploader("Add a whiteboard photo", type=["jpg", "jpeg", "png", "heic", "heif"],
                                  max_upload_size=MAX_UPLOAD_MB, key=f'upload_{st.session_state["upload_generation"]}')
        if upload is not None:
            try:
                accept_photo(upload.getvalue(), upload.name)
            except ValueError as error:
                st.error(str(error))
            else:
                st.rerun()
        st.caption("JPG, PNG or iPhone HEIC · up to 30 MB")
        if SAMPLE_PHOTO.is_file() and st.button("Try a sample photo", width="stretch"):
            accept_photo(SAMPLE_PHOTO.read_bytes(), "Sample whiteboard.jpg")
            st.rerun()
    with help_col:
        with st.container(border=True):
            st.subheader("A better photo helps")
            st.write("• Keep all the writing in frame.\n\n• Get close enough to see the letters.\n\n• Avoid glare over the words.")
        st.caption("Processed on this computer. Photos are not saved to a photo library by this app.")


def check_photo_page() -> None:
    st.title("Check your photo")
    st.write("Can you see every letter? Rotate or trim the edges if you need to.")
    preview_col, tools_col = st.columns([1.8, 1], gap="large")
    with tools_col:
        st.subheader("Adjust the photo")
        left, right = st.columns(2)
        left.button("↶ Rotate left", on_click=rotate_photo, args=(-1,), width="stretch")
        right.button("↷ Rotate right", on_click=rotate_photo, args=(1,), width="stretch")
        trim = st.toggle("Trim the edges", key="trim_edges", on_change=save_trim)
        if trim:
            st.caption("Move the handles to keep the part you want. The preview updates automatically.")
            st.slider("Keep left to right", 0, 100, key="horizontal_area", format="%d%%", on_change=save_crop)
            st.slider("Keep top to bottom", 0, 100, key="vertical_area", format="%d%%", on_change=save_crop)
        st.caption("Cropping keeps the original pixels. It does not sharpen blurry letters or fix a side view.")
    x = st.session_state["horizontal_area"] if trim else (0, 100)
    y = st.session_state["vertical_area"] if trim else (0, 100)
    try:
        prepared = prepare_photo(st.session_state["photo"], st.session_state["turns"], (x[0], y[0], x[1], y[1]))
    except ValueError:
        with preview_col:
            st.warning("The crop is too small. Move the handles apart to keep more of the photo.")
        prepared = None
    else:
        with preview_col:
            st.image(prepared, width="stretch", alt="Photo preview with your current rotation and crop")
            st.caption(st.session_state["photo_name"] + " · this is the photo we’ll read")
    with tools_col:
        if has_edits():
            st.checkbox("Replace my edited notes with the new extraction", key="replace_edits")
            st.caption("Your edits stay safe until you choose to replace them.")
        replace_allowed = not has_edits() or st.session_state.get("replace_edits", False)
        if st.button("Extract notes", type="primary", width="stretch", disabled=prepared is None or not replace_allowed):
            try:
                with st.spinner("Finding the writing and extracting your notes…", show_time=True):
                    result = run_ocr(prepared)
            except Exception:
                LOGGER.exception("Whiteboard extraction failed")
                st.error("We couldn’t read this photo just now. Try again, or choose another photo. Your photo and any edited notes are still here.")
            else:
                text = str(result.get("text", ""))
                st.session_state.update(result=result, result_photo=prepared.copy(), notes=text, draft_notes=text,
                                        phase="review")
                st.session_state.pop("selected_line", None)
                st.session_state.pop("region_event", None)
                st.session_state["show_regions"] = True
                st.session_state["result_generation"] += 1
                st.rerun()
        if "result" in st.session_state and st.button("Back to my notes", width="stretch"):
            st.session_state["phase"] = "review"
            st.rerun()
        if st.button("Choose a different photo", type="tertiary", width="stretch"):
            new_photo()


def review_page() -> None:
    result = st.session_state["result"]
    image = st.session_state["result_photo"]
    lines = result.get("lines", [])
    st.title("Make these notes yours")
    st.write("Select a numbered box to check its handwriting, then edit and keep your notes.")
    if lines:
        selected = st.session_state.get("selected_line", 0)
        if not isinstance(selected, int) or not 0 <= selected < len(lines):
            selected = 0
        st.session_state["selected_line"] = selected
    else:
        selected = None
    photo_col, notes_col = st.columns(2, gap="large")
    with photo_col:
        st.subheader("Writing found in your photo")
        st.session_state.setdefault("show_regions", True)
        show_regions = st.toggle("Show where text was found", key="show_regions")
        if show_regions and lines:
            event = region_photo(image, lines, selected, st.session_state["result_generation"])
            if isinstance(event, dict) and event != st.session_state.get("region_event"):
                index = event.get("index")
                if isinstance(index, int) and 0 <= index < len(lines):
                    st.session_state.update(selected_line=index, region_event=event)
                    st.rerun()
        else:
            st.image(image, width="stretch", alt="Your whiteboard photo for checking the extracted text")
        if lines:
            with st.container(border=True):
                st.subheader("Check a detected line")
                selected = st.selectbox("Choose a part of the photo", range(len(lines)),
                    format_func=lambda i: f'{i + 1}. {str(lines[i].get("text", ""))[:65] or "Unclear writing"}',
                    key="selected_line")
                try:
                    crop = recognition_crop(image, lines[selected]["box"])
                except (KeyError, ValueError):
                    st.info("Check this part in the full photo above.")
                else:
                    st.image(crop, width="stretch", alt="Selected handwriting crop extracted by the OCR pipeline")
                st.caption("Original reading for this region")
                st.text(str(lines[selected].get("text", "")))
                st.caption("Check the strokes here and correct your notes on the right. This is the extracted photo region, without sharpening or generated strokes.")
        if st.button("Adjust photo and read again", width="stretch"):
            save_notes()
            st.session_state.update(phase="check", replace_edits=False)
            st.rerun()
    with notes_col:
        st.subheader("Your notes")
        if not str(result.get("text", "")).strip():
            st.warning("We couldn’t find readable text. Try a closer photo with less glare. You can also type your notes below.")
        else:
            st.caption("These are a first draft. Check missing lines, names and numbers against the photo.")
        if "notes" not in st.session_state:
            st.session_state["notes"] = st.session_state.get("draft_notes", str(result.get("text", "")))
        notes = st.text_area("Edit your notes", key="notes", height=420, on_change=save_notes,
                             label_visibility="collapsed", placeholder="Your notes will appear here. You can also type them yourself.")
        words = len(notes.split())
        st.caption(f'{words} {"word" if words == 1 else "words"} · edits stay on this page until you close or refresh it')
        notes_actions(notes)
        if st.button("Add another photo", type="tertiary", width="stretch"):
            new_photo()
    st.caption("Please also check for writing the app may have missed. Your original photo is unchanged.")
    with st.expander("How your photo was read"):
        st.write("Your photo → find writing → extract each region → recognise characters")
        first, second, third = st.columns(3)
        with first:
            st.markdown("**1 · Photo we read**")
            st.image(image, width="stretch", alt="Actual photo supplied to OCR")
        with second:
            st.markdown("**2 · Writing locations**")
            st.image(draw_regions(image, lines, selected), width="stretch", alt="Detected handwriting regions")
            st.caption("Boxes show where writing was found; they can still miss or merge words.")
        with third:
            st.markdown("**3 · Region to text**")
            if selected is not None:
                try:
                    st.image(recognition_crop(image, lines[selected]["box"]), width="stretch", alt="Actual extracted region before reader resizing")
                except (KeyError, ValueError):
                    st.caption("Region preview unavailable.")
                st.text(str(lines[selected].get("text", "")))
                st.caption("The reader resizes this crop and predicts its characters. Your edits change the notes, not this original reading.")
            else:
                st.caption("No writing regions were returned for this photo.")


def main() -> None:
    st.set_page_config(page_title="Board2Notes · Photo to notes", page_icon="✍️", layout="wide")
    initialise()
    header()
    {"add": add_photo_page, "check": check_photo_page, "review": review_page}[st.session_state["phase"]]()


if __name__ == "__main__":
    main()
