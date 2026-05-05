import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import re

# -----------------------------------
# Load Hugging Face Models
# -----------------------------------

# Image Caption Model
image_captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

# Story Generation Model
story_generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

# -----------------------------------
# Function: Generate Image Caption
# -----------------------------------
def img2text(image):
    """
    Generate a clean image description
    """

    result = image_captioner(image)

    caption = result[0]["generated_text"]

    # Clean unnecessary words
    caption = caption.replace("illustration", "")
    caption = caption.replace("cartoon", "")
    caption = caption.replace("drawing", "")
    caption = caption.strip()

    return caption


# -----------------------------------
# Function: Generate Story
# -----------------------------------
def text2story(caption):
    """
    Generate a short children's story
    """

    prompt = (
        f"Children story: "
        f"A happy and cute story about {caption}. "
        f"The story should use simple words for kids. "
        f"Keep it short and fun."
    )

    result = story_generator(
        prompt,
        max_new_tokens=60,
        temperature=0.7,
        do_sample=True,
        truncation=True
    )

    generated_text = result[0]["generated_text"]

    # Remove prompt from output
    story = generated_text.replace(prompt, "").strip()

    # Remove strange symbols
    story = re.sub(r'[^A-Za-z0-9 ,.?!]', '', story)

    # Backup story if generation is bad
    if len(story) < 20:
        story = (
            f"One sunny day, {caption} had lots of fun together. "
            f"They laughed, ran, and played happily in the warm sunshine. "
            f"Everyone smiled and enjoyed the wonderful day."
        )

    return story


# -----------------------------------
# Function: Convert Story to Audio
# -----------------------------------
def text2audio(story_text):
    """
    Convert story into speech audio
    """

    tts = gTTS(text=story_text, lang="en")

    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    tts.save(temp_audio.name)

    return temp_audio.name


# -----------------------------------
# Streamlit UI
# -----------------------------------

st.title("📚 AI Storytelling App for Kids")

st.write(
    "Upload a picture and enjoy a cute AI-generated story!"
)

# Upload image
uploaded_image = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# If image uploaded
if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Generate Story Button
    if st.button("Generate Story"):

        # Step 1: Image Caption
        with st.spinner("Looking at the picture..."):

            caption = img2text(image)

        st.subheader("🎨 Picture Description")

        st.write(caption)

        # Step 2: Story Generation
        with st.spinner("Creating a fun story..."):

            story = text2story(caption)

        st.subheader("📖 Fun Story")

        st.write(story)

        # Step 3: Audio Generation
        with st.spinner("Making story audio..."):

            audio_file = text2audio(story)

        st.subheader("🔊 Listen to the Story")

        st.audio(audio_file)
   
