import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile

# -----------------------------------
# Load Models
# -----------------------------------

# Image Captioning Model
image_to_text = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

# Text Generation Model
story_generator = pipeline(
    "text-generation",
    model="gpt2"
)

# -----------------------------------
# Function: Image to Text
# -----------------------------------
def img2text(image):
    """
    Generate a simple image caption
    """

    caption = image_to_text(image)[0]["generated_text"]

    return caption


# -----------------------------------
# Function: Text to Story
# -----------------------------------
def text2story(caption):
    """
    Generate a short and kid-friendly story
    """

    prompt = (
        f"Write a cute, simple, and happy story for children "
        f"between 3 and 10 years old based on this picture: {caption}. "
        f"The story should only describe the picture content. "
        f"Use easy words, fun actions, and cheerful emotions. "
        f"Keep the story between 50 and 80 words."
    )

    story = story_generator(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )

    generated_text = story[0]["generated_text"]

    # Remove prompt part
    story_text = generated_text.replace(prompt, "").strip()

    # Clean unwanted symbols
    story_text = story_text.replace("\n", " ")

    return story_text


# -----------------------------------
# Function: Text to Audio
# -----------------------------------
def text2audio(story_text):
    """
    Convert text story into audio
    """

    tts = gTTS(text=story_text, lang='en')

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
    "Upload a picture and enjoy a fun AI-generated story with audio!"
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

    # Generate button
    if st.button("Generate Story"):

        # Generate caption
        with st.spinner("Looking at the picture..."):

            caption = img2text(image)

        st.subheader("🌟 Picture Description")

        st.write(caption)

        # Generate story
        with st.spinner("Creating a fun story..."):

            story = text2story(caption)

        st.subheader("📖 Fun Story")

        st.write(story)

        # Generate audio
        with st.spinner("Making story audio..."):

            audio_file = text2audio(story)

        st.subheader("🔊 Listen to the Story")

        st.audio(audio_file)
