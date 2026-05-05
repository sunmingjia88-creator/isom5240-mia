import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile

# -------------------------------
# Function: Image to Text
# -------------------------------
def img2text(image):
    """
    Generate image caption using BLIP model
    """

    image_to_text = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base"
    )

    caption = image_to_text(image)[0]["generated_text"]

    return caption


# -------------------------------
# Function: Text to Story
# -------------------------------
def text2story(caption):
    """
    Generate a short story based on image caption
    """

    generator = pipeline(
        "text-generation",
        model="gpt2"
    )

    prompt = f"""
    Create a short and fun story for children aged 3 to 10.
    The story should be simple, happy, and easy to understand.

    Image description: {caption}

    Story:
    """

    story = generator(
        prompt,
        max_length=100,
        num_return_sequences=1,
        temperature=0.9
    )

    story_text = story[0]["generated_text"]

    return story_text


# -------------------------------
# Function: Text to Audio
# -------------------------------
def text2audio(story_text):
    """
    Convert story text into audio using gTTS
    """

    tts = gTTS(text=story_text, lang='en')

    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

    tts.save(temp_audio.name)

    return temp_audio.name


# -------------------------------
# Streamlit UI
# -------------------------------

st.title("📚 AI Storytelling App for Kids")

st.write(
    """
    Upload an image and let AI create
    a fun story with audio narration!
    """
)

uploaded_image = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# Process image
if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Generate Story Button
    if st.button("Generate Story"):

        # Image Captioning
        with st.spinner("Analyzing image..."):

            caption = img2text(image)

        st.subheader("🖼 Image Caption")

        st.write(caption)

        # Story Generation
        with st.spinner("Creating story..."):

            story = text2story(caption)

        st.subheader("📖 Generated Story")

        st.write(story)

        # Text to Speech
        with st.spinner("Generating audio..."):

            audio_file = text2audio(story)

        st.subheader("🔊 Story Audio")

        st.audio(audio_file)
