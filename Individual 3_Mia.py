"""
Storytelling Application for Kids (Age 3-10)
Using Hugging Face Transformers Pipelines
- Image Captioning: Salesforce/blip-image-captioning-base
- Story Generation: Template-based + keyword matching (accurate and kid-friendly)
- Text-to-Speech: gTTS (Google Text-to-Speech)
"""

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import random

# ============================================
# Helper Functions
# ============================================

@st.cache_resource
def load_captioning_model():
    """Load the image captioning model (cached for performance)"""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


def img2text(image, captioning_pipeline):
    """
    Convert image to text caption using BLIP model
    """
    try:
        result = captioning_pipeline(image)
        caption = result[0]["generated_text"]
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "a group of people eating together"


def extract_keywords(caption):
    """
    Extract key elements from caption to understand the scene
    """
    caption_lower = caption.lower()
    
    # Define keyword categories
    keywords = {
        'eating': ['eating', 'dinner', 'lunch', 'food', 'meal', 'restaurant', 'table', 'plate', 'fork', 'knife'],
        'playing': ['playing', 'game', 'toy', 'ball', 'run', 'jump', 'playground', 'park'],
        'animal': ['dog', 'cat', 'bird', 'rabbit', 'horse', 'cow', 'pig', 'duck', 'elephant', 'lion', 'tiger', 'bear', 'monkey'],
        'outdoor': ['park', 'beach', 'forest', 'garden', 'outside', 'sun', 'grass', 'tree', 'flower', 'nature'],
        'indoor': ['inside', 'room', 'house', 'home', 'building', 'office', 'school', 'classroom'],
        'celebration': ['birthday', 'party', 'cake', 'candle', 'gift', 'present', 'balloon', 'celebration'],
        'family': ['family', 'parent', 'mother', 'father', 'mom', 'dad', 'sister', 'brother'],
        'friends': ['friend', 'friends', 'group', 'people', 'together'],
        'sleeping': ['sleep', 'sleeping', 'bed', 'nap', 'rest'],
        'reading': ['read', 'reading', 'book', 'story'],
        'walking': ['walk', 'walking', 'stroll', 'street', 'road']
    }
    
    matched = []
    for category, words in keywords.items():
        for word in words:
            if word in caption_lower:
                matched.append(category)
                break
    
    # Remove duplicates
    matched = list(set(matched))
    
    return matched


def create_story_from_caption(caption, keywords):
    """
    Create a natural, first-person story based on the image content
    """
    caption_lower = caption.lower()
    
    # === EATING / DINING / RESTAURANT stories ===
    if 'eating' in keywords or 'dinner' in caption_lower or 'lunch' in caption_lower or 'food' in caption_lower or 'table' in caption_lower:
        stories = [
            f"Today I went to eat with my family and friends. {caption}. We ordered yummy food like pizza and noodles. Everyone was smiling and talking. We shared so many funny stories. What a great night together!",
            f"Look at this picture! {caption}. The food looks so delicious. I had a burger and my friend had salad. We laughed and ate until our tummies were full. I love eating with people I care about.",
            f"This is a happy memory. {caption}. The restaurant was warm and cozy. We celebrated my friend's birthday with cake and candles. Being together with good food makes everything better!",
            f"I see {caption}. We were all hungry so we went to eat. The table was full of tasty dishes. We talked about our day and planned fun things for tomorrow. Eating together is the best!"
        ]
        return random.choice(stories)
    
    # === BIRTHDAY / PARTY / CELEBRATION stories ===
    if 'celebration' in keywords or 'birthday' in caption_lower or 'cake' in caption_lower:
        stories = [
            f"Wow, it's a party! {caption}. We sang the birthday song and blew out the candles. Everyone clapped and cheered. I got a big piece of chocolate cake. This was the best birthday ever!",
            f"This picture shows {caption}. Balloons were floating everywhere. We played games and danced to fun music. My friends gave me nice presents. I felt so loved and happy!",
            f"I see {caption}. We were celebrating something special. There was a big cake with colorful icing. We took lots of photos and made happy memories. What a wonderful day!"
        ]
        return random.choice(stories)
    
    # === PLAYING / OUTDOOR / PARK stories ===
    if 'playing' in keywords or 'outdoor' in keywords:
        stories = [
            f"I see {caption}. The sun was shining and the air was fresh. We ran and played games on the soft green grass. I made new friends and we laughed a lot. Playing outside is so much fun!",
            f"Look at this picture! {caption}. We went to the park after school. Some kids were on the swings, some were playing ball. I love days when I can play outside with everyone.",
            f"This is a picture of {caption}. We were having a race to see who was fastest. The wind felt good on my face. Even when we got tired, we kept smiling. What a great day to play!"
        ]
        return random.choice(stories)
    
    # === ANIMAL / PET stories ===
    if 'animal' in keywords:
        if 'dog' in caption_lower:
            stories = [
                f"I see a cute dog in this picture! {caption}. The dog was wagging its tail and wanted to play. I gave the dog a treat and it licked my hand. Dogs are our best friends!",
                f"Look at this furry friend! {caption}. The dog ran around the yard and chased a butterfly. Then it came back and sat next to me. Having a pet makes every day happier!"
            ]
        elif 'cat' in caption_lower:
            stories = [
                f"I see a soft cat in this picture! {caption}. The cat was taking a nap in a warm spot. Then it woke up and stretched its little paws. Cats are so cute and fluffy!",
                f"Look at this pretty cat! {caption}. The cat watched birds from the window. Then it came to me for some petting. I love spending time with this sweet cat!"
            ]
        else:
            stories = [
                f"I see an animal in this picture! {caption}. The animal looked so cute and friendly. I watched it play and run around. Animals make the world a happier place!",
                f"This picture shows {caption}. The animal was eating some food and looking around. I think it was happy and safe. I'm glad I got to see this lovely animal today!"
            ]
        return random.choice(stories)
    
    # === FAMILY / FRIENDS / GROUP stories ===
    if 'family' in keywords or 'friends' in keywords:
        stories = [
            f"This picture shows {caption}. We were all together, smiling and laughing. My mom made delicious food and my dad told funny jokes. Being with family is the best feeling in the world!",
            f"I see {caption}. My friends and I spent the whole afternoon together. We played, we ate snacks, and we shared secrets. True friends make life so much better. I love them so much!",
            f"Look at this happy picture! {caption}. Everyone was having a good time. We took this photo to remember the fun day. I will keep this memory in my heart forever!"
        ]
        return random.choice(stories)
    
    # === SLEEPING / REST stories ===
    if 'sleeping' in keywords:
        stories = [
            f"I see {caption}. After a long day of playing, it was time to rest. I closed my eyes and felt so peaceful. Sleeping helps us grow big and strong. Good night, everyone!",
            f"This picture shows {caption}. The room was quiet and dark. I snuggled under my warm blanket and fell asleep. Sweet dreams come to those who rest well!"
        ]
        return random.choice(stories)
    
    # === READING / BOOK stories ===
    if 'reading' in keywords:
        stories = [
            f"I see {caption}. I picked up my favorite book and started reading. The story was about a brave little mouse. Reading takes you to magical places without leaving your chair!",
            f"This picture shows {caption}. Books are full of adventure and fun. Every page teaches us something new. I love to read every single day!"
        ]
        return random.choice(stories)
    
    # === WALKING / STREET stories ===
    if 'walking' in keywords:
        stories = [
            f"I see {caption}. The sun was warm and the breeze was soft. I walked down the street and looked at all the pretty houses. A walk outside always makes me feel happy and calm.",
            f"This picture shows {caption}. I put on my shoes and went for a walk. I saw flowers, trees, and friendly people. Walking is good exercise and it's fun too!"
        ]
        return random.choice(stories)
    
    # === DEFAULT / GENERAL stories (for any other picture) ===
    default_stories = [
        f"I look at this picture and I see {caption}. It makes me feel happy inside. Every picture tells a little story. This one is about joy and beauty. I'm glad I got to see it today!",
        f"This is a nice picture. {caption}. The colors are so pretty. I like looking at things that make me smile. Thank you for sharing this lovely photo with me!",
        f"I see {caption}. Sometimes the simple things are the most beautiful. A picture can capture a special moment. I will remember this happy scene for a long time."
    ]
    return random.choice(default_stories)


def text2audio(story_text):
    """
    Convert story text to audio using gTTS
    """
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Slow speed for kids to understand better
        tts = gTTS(text=story_text, lang="en", slow=True)
        tts.save(temp_audio_path)
        
        return temp_audio_path
    except Exception as e:
        st.error(f"Error converting text to audio: {e}")
        return None


# ============================================
# Streamlit UI
# ============================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Kids Storyteller",
        page_icon="📖",
        layout="centered"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .story-box {
            background-color: #FFF8DC !important;
            padding: 25px !important;
            border-radius: 20px !important;
            border-left: 8px solid #FF6B6B !important;
            border-right: 8px solid #4ECDC4 !important;
            font-size: 20px !important;
            line-height: 1.8 !important;
            font-family: 'Comic Neue', 'Comic Sans MS', 'Chalkboard SE', cursive !important;
            color: #1A1A2E !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        }
        .story-title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            color: #FF6B6B !important;
            margin-bottom: 20px;
        }
        .caption-box {
            background-color: #E8F8F5 !important;
            padding: 12px !important;
            border-radius: 15px !important;
            font-size: 16px !important;
            margin: 10px 0 !important;
            color: #2C3E50 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App title
    st.markdown('<div class="story-title">📖 Kids Storyteller 📖</div>', unsafe_allow_html=True)
    st.markdown("### Turn any picture into a happy story!")
    st.markdown("🎈 For kids aged 3-10 | 📖 Easy to read | 🔊 Listen along")
    
    # Sidebar
    with st.sidebar:
        st.header("🌟 How to Use")
        st.markdown("""
        1. **Upload a picture** (JPG or PNG)
        2. **Click 'Make a Story'**
        3. **Read the story** about your picture
        4. **Listen to the audio**
        5. **Download** to keep it!
        """)
        st.divider()
        st.markdown("**📸 Try these pictures:**")
        st.markdown("- 🍕 People eating together")
        st.markdown("- 🎂 Birthday party with cake")
        st.markdown("- 🐕 A dog or cat")
        st.markdown("- 👨‍👩‍👧 Family or friends")
        st.markdown("- 🌳 Park or playground")
    
    # File uploader
    uploaded_image = st.file_uploader(
        "🎨 **Upload a picture**", 
        type=["jpg", "jpeg", "png"],
        help="Pick any picture and I'll tell a story about it!"
    )
    
    # Display uploaded image
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Your picture", use_container_width=True)
        
        # Generate Story Button
        if st.button("📖 Make a Story 📖", type="primary", use_container_width=True):
            
            # Step 1: Look at the picture
            with st.spinner("👀 Looking at your picture..."):
                captioning_pipeline = load_captioning_model()
                caption = img2text(image, captioning_pipeline)
                
                st.markdown(f"""
                <div class="caption-box">
                🖼️ <strong>I see:</strong> {caption}
                </div>
                """, unsafe_allow_html=True)
            
            # Step 2: Extract keywords
            keywords = extract_keywords(caption)
            
            # Step 3: Create story based on keywords
            with st.spinner("✏️ Writing a story for you..."):
                story = create_story_from_caption(caption, keywords)
            
            # Show the story
            st.markdown("---")
            st.markdown('<div class="story-title">✨ The Story ✨</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="story-box">
            {story}
            </div>
            """, unsafe_allow_html=True)
            
            # Word count
            word_count = len(story.split())
            st.caption(f"📖 {word_count} words")
            st.markdown("---")
            
            # Step 4: Make audio
            with st.spinner("🔊 Making audio..."):
                audio_path = text2audio(story)
            
            if audio_path:
                st.subheader("🎧 Listen to the story")
                
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                
                st.download_button(
                    label="💾 Download Audio",
                    data=audio_bytes,
                    file_name="my_story.mp3",
                    mime="audio/mpeg"
                )
                
                try:
                    os.unlink(audio_path)
                except:
                    pass
            else:
                st.warning("⚠️ Audio not ready. You can still read the story!")
    
    else:
        st.info("👆 **Upload a picture to start!**")
        
        with st.expander("📷 Example", expanded=False):
            st.markdown("""
            **If you upload a picture of people eating together:**
            
            > *"Today I went to eat with my family and friends. We ordered yummy food like pizza and noodles. Everyone was smiling and talking. We shared so many funny stories. What a great night together!"*
            
            The AI looks at your picture and tells a story that matches what it sees!
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #95A5A6; font-size: 13px;'>"
        "Made with ❤️ for kids | Stories that match your pictures"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
