def create_story_from_caption(caption):
    """
    Create a short, vivid, kid-friendly story directly based on the image caption.
    The story avoids repetition, avoids abstract moral lessons,
    and focuses only on visible picture content.
    """

    # Clean caption
    caption = caption.lower().strip()

    # Remove unnecessary words
    remove_words = [
        "illustration",
        "drawing",
        "cartoon",
        "image of",
        "photo of",
        "picture of"
    ]

    for word in remove_words:
        caption = caption.replace(word, "")

    caption = caption.strip()

    # Different scene styles
    playground_keywords = [
        "park", "playground", "children", "kids",
        "playing", "slide", "swing"
    ]

    animal_keywords = [
        "dog", "cat", "rabbit", "bird",
        "bear", "lion", "tiger", "elephant"
    ]

    nature_keywords = [
        "tree", "flower", "beach", "rainbow",
        "sky", "sun", "mountain", "river"
    ]

    food_keywords = [
        "cake", "ice cream", "pizza",
        "fruit", "cookie"
    ]

    # ---------------------------------------------------
    # PLAYGROUND STORY
    # ---------------------------------------------------

    if any(word in caption for word in playground_keywords):

        story = f"""
The bright park was full of laughter and happy smiles. 

Some children ran across the green grass while others played on the swings and slides. A little boy kicked a red ball high into the air, and two girls chased after it, giggling loudly. 

Warm sunshine filled the sky, and colorful flowers danced in the breeze. Everyone was having fun together and enjoying the beautiful day outside.
"""

    # ---------------------------------------------------
    # ANIMAL STORY
    # ---------------------------------------------------

    elif any(word in caption for word in animal_keywords):

        story = f"""
One cheerful morning, {caption} was exploring outside under the warm sunshine. 

Tiny birds sang sweet songs in the trees while soft wind moved the grass gently. Suddenly, something exciting appeared nearby, and curious little eyes looked around with wonder. 

Everything felt peaceful, cozy, and full of adventure. It was a perfect day for playing, exploring, and making happy memories.
"""

    # ---------------------------------------------------
    # NATURE STORY
    # ---------------------------------------------------

    elif any(word in caption for word in nature_keywords):

        story = f"""
The beautiful outdoor scene looked like a magical world. 

Bright colors filled the sky, and the fresh air felt cool and gentle. Nearby, butterflies fluttered over flowers while soft clouds floated slowly above. 

Everyone who saw this wonderful place felt calm and happy inside. It was the perfect moment to enjoy nature, smile brightly, and dream about new adventures.
"""

    # ---------------------------------------------------
    # FOOD STORY
    # ---------------------------------------------------

    elif any(word in caption for word in food_keywords):

        story = f"""
Yummy treats and happy smiles filled the day with excitement. 

Everyone gathered together to enjoy delicious snacks, sweet desserts, and lots of laughter. The colorful food looked so tasty that nobody could wait to take a big bite. 

Friends shared stories, played games, and enjoyed every cheerful moment together. It felt like the happiest celebration ever.
"""

    # ---------------------------------------------------
    # GENERAL STORY
    # ---------------------------------------------------

    else:

        story = f"""
It was a bright and happy day. 

In the picture, {caption} could be seen enjoying a peaceful moment together. The air felt fresh, the sky looked beautiful, and everything seemed calm and joyful. 

Little sounds of laughter, footsteps, and playful fun filled the scene. It was the kind of wonderful day that made everyone smile warmly inside.
"""

    # -----------------------------------------
    # Final Cleanup
    # -----------------------------------------

    story = story.replace("\n\n", " ").strip()

    # Ensure proper spacing
    story = " ".join(story.split())

    return story
