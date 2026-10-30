"""Ready-made episodes for young children (about 3 to 6 years old), written
for Peeko but usable with any hero: ``{name}`` is replaced by its name.

The main one is season 1 of "Peeko & Friends" (:mod:`season1`): twelve
episodes of about three minutes with Lulu, Kiko and Bobo, a teaser before the
show's opening, conversations, counting, picture quizzes, the lesson, a
flower for the collection and a look at the next episode.

The mini episodes (about a minute and a half, Peeko mostly alone) each teach
one simple thing, in three groups: learning (numbers, colours, shapes, how
plants grow), good habits (brushing teeth, washing hands, healthy food,
bedtime, crossing the street) and friends & feelings.
"""

from __future__ import annotations

from typing import Dict, List

from . import season1
from .cartoon import Story

SEASON_GROUP = f"Season {season1.SEASON}: {season1.TITLE}"
GROUPS = (SEASON_GROUP, "Mini: learning", "Mini: good habits", "Mini: friends & feelings")
_MINI = {"Learning": GROUPS[1], "Good habits": GROUPS[2], "Friends & feelings": GROUPS[3]}


def _s(setting, time, narration, actions, say="", effect="none", weather="clear", **extra) -> Dict:
    return {"setting": setting, "time": time, "weather": weather, "narration": narration,
            "actions": actions, "say": say, "effect": effect, **extra}


EPISODES: List[Dict] = [
    {"id": "counting-apples", "group": "Learning", "title": "{name} Counts Apples",
     "lesson": "We can count things all around us: one, two, three, four, five!",
     "question": "How many apples can you count?",
     "scenes": [
         _s("meadow", "day", "Hello, friends! This is {name}, a little sprout who loves to learn new things.",
            ["enter", "wave"], "Hi! Will you learn with me today?"),
         _s("forest", "day", "{name} walked into the forest and found an apple tree. The apples were red "
            "and shiny!", ["walk", "surprised", "point"], "Wow! So many apples!", board="APPLES", item="apple"),
         _s("meadow", "day", "Let's count the apples together. Ready? Count with {name}!", ["point"],
            "Five apples! Hooray!", "stars", board="Let's count!", item="apple", count=5),
         _s("meadow", "day", "{name} was hungry, so {name} ate one apple. Crunch, crunch!", ["eat", "happy"],
            "Mmm, so sweet!", holds="apple"),
         _s("meadow", "day", "Now, how many apples are left? Let's count again!", ["think"],
            "Four! Five take away one is four!", "confetti", board="5 take away 1", item="apple", count=4),
         _s("meadow", "sunset", "{name} gave an apple to a little bird. Sharing makes everyone happy!",
            ["walk", "wave"], "Here you go, little bird!", "hearts", holds="apple", friend="bird"),
         _s("meadow", "sunset", "What a wonderful day of counting!", ["dance", "clap"],
            "Counting is fun! Bye-bye!", "music"),
     ]},
    {"id": "rainbow-colors", "group": "Learning", "title": "{name} and the Rainbow",
     "lesson": "Red, orange, yellow, green, blue and purple: the colors of the rainbow!",
     "question": "What is your favorite color?",
     "scenes": [
         _s("meadow", "day", "Pitter, patter! It was raining on the meadow, and {name} had a big red umbrella.",
            ["enter", "idle"], "I love the rain!", weather="rain", holds="umbrella"),
         _s("meadow", "day", "Then the sun came out, and look! A rainbow appeared in the sky!",
            ["surprised", "happy"], "Wow! A rainbow!", "stars", board="RAINBOW", item="rainbow"),
         _s("meadow", "day", "A rainbow has six colors. Let's say them with {name}!", ["point"],
            "Red, orange, yellow!", board="RED ORANGE YELLOW"),
         _s("meadow", "day", "And three more colors. Can you say them too?", ["point", "clap"],
            "Green, blue, purple!", board="GREEN BLUE PURPLE"),
         _s("forest", "day", "Can you find something red? {name} found a red apple!", ["walk", "happy"],
            "Red, like an apple!", holds="apple", board="RED", item="apple"),
         _s("forest", "day", "And something yellow? A yellow banana!", ["happy"], "Yellow, like a banana!",
            holds="banana", board="YELLOW", item="banana"),
         _s("forest", "day", "And something orange? A crunchy orange carrot!", ["eat"],
            "Orange, like a carrot!", holds="carrot", board="ORANGE", item="carrot"),
         _s("meadow", "sunset", "Colors are everywhere. What a beautiful, colorful world!", ["dance", "spin"],
            "I love all the colors!", "confetti"),
     ]},
    {"id": "shapes", "group": "Learning", "title": "{name} Finds Shapes",
     "lesson": "A circle is round, a square has four sides, and a triangle has three.",
     "question": "What shapes can you see around you?",
     "scenes": [
         _s("city", "day", "{name} went to the big city to look for shapes.", ["enter", "walk"],
            "Shapes, where are you?"),
         _s("city", "day", "Look at the wheels on the cars. They are circles! A circle is round, round, round.",
            ["point", "happy"], "A circle is round!", board="CIRCLE", item="circle"),
         _s("city", "day", "The windows are squares. A square has four sides, all the same.", ["point"],
            "A square has four sides!", board="SQUARE", item="square"),
         _s("city", "sunset", "The roofs look like triangles. A triangle has three sides and three corners.",
            ["walk", "point"], "Triangle! Three sides!", board="TRIANGLE", item="triangle"),
         _s("city", "night", "At night, the stars come out. Twinkle, twinkle, little star!", ["float"],
            "Stars are my favorite shape!", "stars", board="STAR", item="star"),
         _s("city", "night", "Let's count all the shapes we found today!", ["clap"], "Four shapes! Hooray!",
            "confetti", board="Circle, square, triangle, star", count=4),
     ]},
    {"id": "how-plants-grow", "group": "Learning", "title": "How {name} Grew",
     "lesson": "Plants need water, sunshine and time to grow.",
     "question": "What would you like to grow in a garden?",
     "scenes": [
         _s("meadow", "day", "Long ago, {name} was just a tiny seed, sleeping under the ground.", ["sleep"],
            board="A tiny seed"),
         _s("meadow", "day", "Then the rain came down. Pitter, patter! The seed drank the water.",
            ["sleep", "surprised"], "Mmm, water!", weather="rain", board="WATER", item="cup"),
         _s("meadow", "day", "Then the warm sun came out and shone on the ground.", ["happy"], "So warm!",
            board="SUN", item="sun"),
         _s("meadow", "day", "And the little seed grew, and grew, and grew... into {name}!", ["jump", "happy"],
            "Hello, world!", "stars"),
         _s("meadow", "day", "Plants need three things to grow. Let's count them!", ["point"],
            "Water, sun and time!", board="Water, sun, time", count=3),
         _s("meadow", "sunset", "Now {name} helps the flowers grow, with a cup of water every day.",
            ["walk", "happy"], "Grow, little flowers, grow!", "hearts", holds="cup"),
     ]},
    {"id": "brush-your-teeth", "group": "Good habits", "title": "Brush, Brush, Brush!",
     "lesson": "Brush your teeth every morning and every night.",
     "question": "Did you brush your teeth today?",
     "scenes": [
         _s("room", "night", "It was evening, and {name} had a big, sweet cookie for dessert.", ["eat", "happy"],
            "Yum! I love cookies!", holds="cookie"),
         _s("room", "night", "But sugar likes to stick to our teeth. Uh-oh! What should {name} do?",
            ["surprised", "think"], "Hmm... what should I do?", "question"),
         _s("room", "night", "Of course! It's time to brush! {name} picked up a toothbrush and some toothpaste.",
            ["happy"], "I know! Brush my teeth!", "idea", holds="toothbrush"),
         _s("room", "night", "Up and down, round and round. Brush the front, and brush the back!", ["brush"],
            "Brush, brush, brush!", board="Up and down, round and round!"),
         _s("room", "night", "Let's count to five while we brush. Brush along with {name}!", ["brush"],
            "Five! All clean!", "stars", holds="toothbrush", board="Brush and count!", item="tooth", count=5),
         _s("room", "night", "Then a sip of water. Swish, swish, and spit!", ["eat", "happy"],
            "Sparkly clean teeth!", "stars", holds="cup"),
         _s("room", "night", "Now {name} is ready for bed. Good night, little sprout!", ["yawn", "talk", "sleep"],
            "Good night, friends!"),
     ]},
    {"id": "wash-your-hands", "group": "Good habits", "title": "Wash Your Hands!",
     "lesson": "Wash your hands with soap before you eat and after you play.",
     "question": "When do you wash your hands?",
     "scenes": [
         _s("meadow", "day", "{name} was playing outside after the rain. Splish, splash, in the mud!",
            ["jump", "happy", "jump"], "Mud is so much fun!"),
         _s("meadow", "day", "Oh no! {name}'s hands are all muddy. Tiny germs are hiding there!",
            ["look", "surprised"], "Eww! My hands are dirty!"),
         _s("room", "day", "Time to wash! First, warm water. Then, lots of soap!", ["enter", "happy"],
            "Bubbles, bubbles!", holds="soap"),
         _s("room", "day", "Rub the fronts, rub the backs, and between the fingers too. Let's count to ten!",
            ["wash"], "Ten! Squeaky clean!", "stars", board="Scrub, scrub, scrub!", count=10),
         _s("room", "day", "Rinse off the bubbles, and dry your hands with a soft towel.", ["wash", "happy"],
            "So fresh and clean!", holds="towel"),
         _s("room", "day", "Clean hands make a happy tummy. Now {name} can have a snack!", ["eat", "dance"],
            "Clean hands, yummy snack!", "hearts", holds="strawberry"),
     ]},
    {"id": "eat-your-veggies", "group": "Good habits", "title": "{name} Eats Healthy Food",
     "lesson": "Eat lots of fruit and vegetables to grow big and strong.",
     "question": "What is your favorite vegetable?",
     "scenes": [
         _s("room", "day", "{name} was hungry. On the table there were candies... and vegetables!",
            ["enter", "think"], "Hmm... candy, please!", "question", board="Candy or veggies?", item="candy"),
         _s("room", "day", "{name} ate lots and lots of candy. But soon, {name}'s tummy did not feel good.",
            ["eat", "sad"], "Ooh... my tummy hurts.", holds="candy"),
         _s("room", "day", "Candy is a treat, just for sometimes. Our bodies need healthy food to grow!",
            ["think", "surprised"], "I need healthy food!", "idea"),
         _s("room", "day", "Carrots are crunchy and good for your eyes. Crunch, crunch!", ["eat", "happy"],
            "Crunchy!", holds="carrot", board="CARROT", item="carrot"),
         _s("room", "day", "Broccoli looks like tiny trees, and it makes us strong!", ["eat", "jump"],
            "I feel so strong!", holds="broccoli", board="BROCCOLI", item="broccoli"),
         _s("room", "day", "And fruit is sweet and juicy. Let's count the strawberries!", ["point"],
            "Five strawberries!", board="Yummy fruit!", item="strawberry", count=5),
         _s("meadow", "day", "Now {name} feels healthy and full of energy!", ["run", "jump", "dance"],
            "Healthy food makes me happy!", "stars"),
     ]},
    {"id": "good-night", "group": "Good habits", "title": "Good Night, {name}!",
     "lesson": "Tidy up, brush your teeth, read a story, and sleep well.",
     "question": "What do you do before bed?",
     "scenes": [
         _s("room", "night", "The moon is up and the stars are twinkling. It's almost bedtime for {name}.",
            ["yawn", "idle"], "I'm getting sleepy...", board="Bedtime!", item="moon"),
         _s("room", "night", "First, tidy up! {name} puts the toys away.", ["walk", "happy"],
            "Everything in its place!", holds="teddy", board="1. Tidy up", item="teddy"),
         _s("room", "night", "Second, brush your teeth. Up and down, round and round!", ["brush"],
            "Brush, brush, brush!", board="2. Brush teeth", item="tooth"),
         _s("room", "night", "Third, story time! {name} reads a book about the stars.", ["read"],
            "Once upon a time...", board="3. Read a story", item="book"),
         _s("room", "night", "Let's count the stars before we sleep.", ["point"], "Five little stars!",
            board="Count the stars", item="star", count=5),
         _s("room", "night", "{name} closes its eyes. Good night, little sprout. Sweet dreams!",
            ["yawn", "sleep"], effect="stars"),
     ]},
    {"id": "look-left-look-right", "group": "Good habits", "title": "Look Left, Look Right!",
     "lesson": "Stop, look left and right, and cross the street with a grown-up.",
     "question": "What does a red light mean?",
     "scenes": [
         _s("city", "day", "{name} wanted to go to the park across the street.", ["enter", "point"],
            "The park is over there!"),
         _s("city", "day", "But cars go fast on the road! {name} stopped at the edge of the sidewalk.",
            ["surprised"], "Whoa! Stop!", board="STOP!", item="red light"),
         _s("city", "day", "The traffic light is red. Red means stop. We wait!", ["idle"], "Red means stop!",
            board="RED means STOP", item="red light"),
         _s("city", "day", "Look left, look right, and look left again. Is a car coming?",
            ["look", "look", "look"], "Left, right, left!"),
         _s("city", "day", "The light is green. Green means go! Hold a grown-up's hand and walk across.",
            ["walk"], "Green means go!", board="GREEN means GO", item="green light"),
         _s("meadow", "day", "{name} made it safely to the park. Hooray!", ["happy", "dance"],
            "We did it! Hooray!", "confetti"),
     ]},
    {"id": "sharing-is-caring", "group": "Friends & feelings", "title": "Sharing Is Caring",
     "lesson": "When we share, everyone is happy.",
     "question": "What do you like to share with your friends?",
     "scenes": [
         _s("meadow", "day", "{name} had a shiny new ball and loved to play with it.", ["enter", "jump", "happy"],
            "This is my favorite ball!", holds="ball"),
         _s("meadow", "day", "Then {name} saw a little bunny sitting all alone. The bunny looked sad.",
            ["surprised", "sad"], "Why are you sad, little bunny?", holds="ball", friend="sad bunny"),
         _s("meadow", "day", "The bunny had no toys to play with. What should {name} do?", ["think"],
            "Hmm... should I share my ball?", "question", holds="ball", friend="sad bunny"),
         _s("meadow", "day", "{name} decided to share! The bunny was so happy.", ["happy", "wave"],
            "Let's play together, bunny!", "hearts", friend="bunny"),
         _s("meadow", "day", "They rolled the ball back and forth, and laughed, and laughed!", ["jump", "dance"],
            "Playing together is more fun!", "music", holds="ball", friend="bunny"),
         _s("meadow", "sunset", "When the sun went down, the bunny said thank you for sharing.", ["wave"],
            "You're welcome! See you tomorrow!", "hearts", friend="bunny"),
     ]},
    {"id": "feelings", "group": "Friends & feelings", "title": "How Do You Feel?",
     "lesson": "All feelings are okay. Tell someone how you feel.",
     "question": "How do you feel today?",
     "scenes": [
         _s("meadow", "day", "Everyone has feelings. Today, {name} will show us some!", ["enter", "wave"],
            "Let's talk about feelings!"),
         _s("meadow", "day", "When {name} gets a present, {name} feels happy!", ["happy", "dance"],
            "I feel happy!", "hearts", holds="gift", board="HAPPY"),
         _s("meadow", "day", "When the rain spoils the picnic, {name} feels sad.", ["sad"], "I feel sad...",
            weather="rain", board="SAD"),
         _s("forest", "night", "In the dark forest, a strange noise makes {name} feel scared.", ["scared", "look"],
            "I feel scared!", board="SCARED"),
         _s("meadow", "sunset", "Then a friend comes to give a big hug, and {name} feels better.", ["happy"],
            "A hug makes me feel better!", "hearts", friend="bunny", board="LOVED"),
         _s("meadow", "sunset", "All feelings are okay. When you feel sad or scared, tell someone you love.",
            ["wave"], "How do you feel today?"),
     ]},
]

for _k, _e in enumerate(season1.EPISODES, 1):
    _e.update(group=SEASON_GROUP, episode=_k, season=len(season1.EPISODES), hook=True)
for _e in EPISODES:
    _e["group"] = _MINI.get(_e["group"], _e["group"])
EPISODES = season1.EPISODES + EPISODES
_BY_ID = {e["id"]: e for e in EPISODES}


def _fill(value, name: str):
    if isinstance(value, str):
        return value.replace("{name}", name)
    if isinstance(value, list):
        return [_fill(v, name) for v in value]
    return value


def catalogue(name: str = "") -> List[Dict]:
    """The episodes, for choosing one: id, group, title, lesson and number in the season."""
    n = name.strip() or "Peeko"
    return [{"id": e["id"], "group": e["group"], "title": _fill(e["title"], n), "lesson": _fill(e["lesson"], n),
             "episode": e.get("episode", 0)} for e in EPISODES]


def episode(episode_id: str, name: str = "") -> Story:
    """The episode ``episode_id`` starring ``name`` (Peeko if empty)."""
    if episode_id not in _BY_ID:
        raise KeyError(episode_id)
    n = name.strip() or "Peeko"
    e = _BY_ID[episode_id]
    data = {k: _fill(e.get(k, ""), n) for k in ("title", "lesson", "question", "next")}
    data.update(hook=e.get("hook", False), episode=e.get("episode", 0), season=e.get("season", 0),
                music=e.get("music", "happy"),
                scenes=[{k: _fill(v, n) for k, v in scene.items()} for scene in e["scenes"]])
    return Story.from_dict(data)
