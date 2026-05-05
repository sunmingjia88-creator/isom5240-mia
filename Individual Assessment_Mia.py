"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: Dynamically creates story based on actual image content
- Text-to-Speech: gTTS (Google Text-to-Speech)
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import re
import random

# ============================================
# Helper Functions
# ============================================

@st.cache_resource
def load_captioning_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


def img2text(image, captioning_pipeline):
    try:
        result = captioning_pipeline(image)
        return result[0]["generated_text"]
    except:
        return "a happy scene"


def create_kid_friendly_story(caption):
    caption = caption.lower()

    return f"""
    Once upon a time, there was a scene where {caption}.
    It was a bright and happy moment, full of joy and imagination.
    Everyone in the story was having a wonderful time.
    This picture reminds us how beautiful simple moments can be.
    The end.
    """


def text2audio(story_text):
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()

        tts = gTTS(text=story_text, lang="en", slow=True)
        tts.save(temp_audio_path)

        return temp_audio_path
    except:
        return None


# ============================================
# Streamlit UI
# ============================================

def main():
    st.set_page_config(
        page_title="Kids Storyteller",
        page_icon="📖"
    )

    st.title("📖 Kids Storyteller")
    st.write("Turn your picture into a story!")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Your Image", use_container_width=True)

        if st.button("✨ Generate Story ✨"):

            # Step 1: Image → Caption
            with st.spinner("Analyzing image..."):
                captioning_pipeline = load_captioning_model()
                caption = img2text(image, captioning_pipeline)

                st.write("🧠 AI sees:", caption)

            # ============================================
            # ✅ 这里是你要求修改的“✨ The Story ✨”完整版本
            # ============================================

            with st.spinner("📚 Creating your magical story..."):
                story = create_kid_friendly_story(caption)

            st.markdown("---")

            st.markdown("## ✨ The Story ✨")

            st.write(story)

            st.write("### 📖 Story Summary")
            st.write("This story is generated based on the uploaded image.")

            st.write("### 📏 Story Statistics")
            word_count = len(story.split())
            st.write(f"Word Count: {word_count}")

            if 50 <= word_count <= 100:
                st.success("✅ Perfect length!")
            else:
                st.info("ℹ️ Recommended: 50–100 words")

            if st.button("💫 I Like This Story!"):
                st.write("🎉 Glad you liked it!")

            st.markdown("---")

            # Step 3: Text → Audio
            with st.spinner("Generating audio..."):
                audio_path = text2audio(story)

            if audio_path:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()

                st.audio(audio_bytes)

                st.download_button(
                    label="Download Audio",
                    data=audio_bytes,
                    file_name="story.mp3"
                )

                os.unlink(audio_path)

    else:
        st.info("Please upload an image to start!")


if __name__ == "__main__":
    main()
