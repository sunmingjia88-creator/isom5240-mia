"""
Storytelling Application for Kids (Age 3-10)

Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: google/flan-t5-base
- Text-to-Speech: gTTS
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os


# =========================================================
# Load Models (Cached)
# =========================================================

@st.cache_resource
def load_caption_model():
    """
    Load image captioning model
    """
    return pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )


@st.cache_resource
def load_story_model():
    """
    Load text generation model
    """
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )


# =========================================================
# Image -> Caption
# =========================================================

def img2text(image, caption_model):
    """
    Generate image caption
    """

    try:
        result = caption_model(image)

        caption = result[0]["generated_text"]

        # Clean simple unwanted words
        caption = caption.replace("illustration", "")
        caption = caption.replace("cartoon", "")
        caption = caption.replace("drawing", "")
        caption = caption.strip()

        return caption

    except Exception as e:

        st.error(f"Caption generation failed: {e}")

        return "children playing happily outside"


# =========================================================
# Caption -> Story
# =========================================================

def text2story(caption, story_model):
    """
    Generate a creative and non-repetitive story
    """

    prompt = f"""
Write ONE short children's story about this picture:

Picture: {caption}

Rules:
- Use simple English for children age 3-10
- Story must be 60-90 words
- Describe the picture clearly
- Include actions, feelings, colors, and fun details
- Make the story cheerful and imaginative
- DO NOT repeat sentences
- DO NOT repeat phrases
- DO NOT use lists
- DO NOT say "The end"
- Write only the story

Story:
"""

    try:

        result = story_model(
            prompt,
            max_new_tokens=120,
            temperature=0.9,
            do_sample=True
        )

        story = result[0]["generated_text"].strip()

        # Additional cleanup
        story = story.replace("\n", " ")
        story = story.replace("Story:", "")

        # Remove repeated sentences
        sentences = story.split(". ")

        cleaned_sentences = []
        seen = set()

        for sentence in sentences:

            sentence = sentence.strip()

            if sentence and sentence not in seen:
                cleaned_sentences.append(sentence)
                seen.add(sentence)

        story = ". ".join(cleaned_sentences)

        # Ensure proper ending
        if not story.endswith("."):
            story += "."

        return story

    except Exception as e:

        st.error(f"Story generation failed: {e}")

        return (
            f"On a sunny day, {caption} laughed and played together. "
            f"Bright flowers danced in the wind while happy birds sang in the sky. "
            f"Everyone smiled and enjoyed the warm and beautiful day outside."
        )


# =========================================================
# Story -> Audio
# =========================================================

def text2audio(story_text):
    """
    Convert story text to speech
    """

    try:

        temp_audio = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        tts = gTTS(
            text=story_text,
            lang="en",
            slow=False
        )

        tts.save(temp_audio.name)

        return temp_audio.name

    except Exception as e:

        st.error(f"Audio generation failed: {e}")

        return None


# =========================================================
# Streamlit UI
# =========================================================

def main():

    st.set_page_config(
        page_title="Kids AI Storyteller",
        page_icon="📚",
        layout="centered"
    )

    # -----------------------------------------------------

    st.title("📚 Kids AI Storyteller")

    st.write(
        "Upload a picture and let AI create a fun story with audio!"
    )

    # -----------------------------------------------------

    with st.sidebar:

        st.header("🌈 How It Works")

        st.markdown("""
1. Upload a picture  
2. Click the button  
3. Read the story  
4. Listen to the audio  
""")

        st.divider()

        st.markdown("### 🎨 Try uploading:")
        st.markdown("- Animals 🐶")
        st.markdown("- Parks 🌳")
        st.markdown("- Beaches 🏖️")
        st.markdown("- Family photos 👨‍👩‍👧")
        st.markdown("- Cartoons 🎈")

    # -----------------------------------------------------

    uploaded_image = st.file_uploader(
        "📸 Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    # -----------------------------------------------------

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # -------------------------------------------------

        if st.button("✨ Generate Story ✨"):

            # Load models
            with st.spinner("Loading AI models..."):

                caption_model = load_caption_model()

                story_model = load_story_model()

            # Generate caption
            with st.spinner("Looking at the picture..."):

                caption = img2text(image, caption_model)

            st.subheader("🌟 Picture Description")

            st.write(caption)

            # Generate story
            with st.spinner("Writing a fun story..."):

                story = text2story(caption, story_model)

            st.subheader("📖 Fun Story")

            st.write(story)

            # Word count
            word_count = len(story.split())

            st.caption(f"📏 Story Length: {word_count} words")

            # Generate audio
            with st.spinner("Creating story audio..."):

                audio_path = text2audio(story)

            if audio_path:

                st.subheader("🔊 Listen to the Story")

                with open(audio_path, "rb") as audio_file:

                    audio_bytes = audio_file.read()

                st.audio(audio_bytes)

                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name="kids_story.mp3",
                    mime="audio/mpeg"
                )

                # Remove temp file
                try:
                    os.remove(audio_path)
                except:
                    pass

    else:

        st.info(
            "👆 Upload an image to begin your storytelling adventure!"
        )

    # -----------------------------------------------------

    st.markdown("---")

    st.markdown(
        "<div style='text-align:center; color:gray;'>"
        "Made with ❤️ using Hugging Face and Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================

if __name__ == "__main__":
    main()
