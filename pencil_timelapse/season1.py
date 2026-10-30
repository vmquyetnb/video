"""Peeko & Friends, season 1: "Sunny Garden" (12 episodes, about 3 minutes each).

Peeko, a little sprout who wants to learn everything, lives in a flowerpot
house in Sunny Garden with its friends: shy Lulu the bunny, who paints;
Kiko the bluebird, who says "Did you know...?" and is often a little bit
wrong; and Bobo the bear cub, who is always hungry. Every time Peeko learns
something, a flower blooms on its sprout; when twelve flowers have bloomed,
it is time for the Sunny Garden Festival (episode 12).

Each episode opens with a teaser (a funny or surprising moment), then the
show's opening; it has a problem, two or three moments where children join
in (counting out loud, a picture quiz with a 3-2-1 countdown, saying a word),
a happy ending, the lesson, the newest flower and a look at the next episode.

Lines are written "Who: what they say (what they do)"; a line without a name
is Peeko's. ``{name}`` is the hero's name.
"""

from __future__ import annotations

from typing import Dict, List

SEASON = 1
TITLE = "Sunny Garden"


def _s(setting: str, time: str, narration: str, actions: List[str], lines: str = "", cast=(), **extra) -> Dict:
    return {"setting": setting, "time": time, "weather": extra.pop("weather", "clear"), "narration": narration,
            "actions": actions, "lines": lines.strip("\n"), "cast": list(cast), **extra}


ALL = ("Lulu", "Kiko", "Bobo")

EPISODES: List[Dict] = [
    {"id": "s1e01-hello-sunny-garden", "title": "Hello, Sunny Garden!",
     "lesson": "Friends make every day more fun!", "next": "Bobo's Big Breakfast",
     "question": "Who is your best friend?",
     "scenes": [
         _s("garden", "day", "", ["float"], """
Leaves, spin! Wheee! (float)
Kiko: Wait! Sprouts can't fly!
This sprout can! (float)""", ["Kiko"], effect="stars"),
         _s("garden", "day", "Welcome to Sunny Garden! This little sprout is {name}. {name} lives in a flowerpot "
            "house and wants to learn everything!", ["enter", "wave"], """
Hello, friends! I'm {name}! (wave)
Do you want to meet my friends? Let's go! (jump)"""),
         _s("garden", "day", "First, they found Lulu, a little bunny who loves to paint. She is a bit shy.",
            ["walk", "wave"], """
Hi, Lulu! (wave)
Lulu: Oh! H-hello... I'm Lulu.
Lulu: I like painting flowers. (happy)
Wow! Your flowers are so pretty, Lulu!""", ["Lulu"], holds="flower"),
         _s("meadow", "day", "Then a little bluebird zoomed down from the sky.", ["look", "surprised"], """
Kiko: Hello! I'm Kiko, and I know everything! (happy)
Kiko: Did you know... the sky is made of blueberry jam?
Lulu: Um... I don't think so, Kiko.
Kiko: Hmm. Maybe not. (think)""", ["Lulu", "Kiko"]),
         _s("forest", "day", "Suddenly, the ground went thump, thump, thump! Something big was coming!",
            ["scared"], """
Bobo: Hi, everybody! I'm Bobo! (wave)
Bobo: Is it snack time yet?
Ha ha! Bobo, you're always hungry! (happy)""", ALL),
         _s("garden", "day", "{name} has so many friends now. Let's count everyone together!", ["point"], """
Four friends! Hooray! (clap)
Bobo: Four friends, and zero snacks... (sad)""", ALL, board="How many friends?", count=4),
         _s("garden", "day", "Bobo's tummy is rumbling. Which one would Bobo like to eat? The ball, the cookie, "
            "or the book?", ["think"], """
The cookie! (happy)
Bobo: Yum! Thank you, friends! (eat)""", ["Bobo"], board="What does Bobo want?",
            quiz=["ball", "cookie", "book"], answer="cookie"),
         _s("garden", "sunset", "Every time {name} learns something new, a flower blooms on its sprout. And when "
            "twelve flowers bloom, it's time for the big Sunny Garden Festival!", ["surprised", "happy"], """
Today I learned something too!
Lulu: Oh, how pretty!
Kiko: Did you know... the festival has music and games!
Bobo: And snacks? Please say snacks! (jump)""", ALL, effect="stars"),
     ]},
    {"id": "s1e02-bobos-big-breakfast", "title": "Bobo's Big Breakfast",
     "lesson": "Eat fruit and vegetables every day. Sweets are just for sometimes.",
     "next": "Kiko Counts the Stars", "question": "What is your favorite fruit?",
     "scenes": [
         _s("kitchen", "day", "", ["surprised"], """
Bobo: Honey cakes for breakfast! Honey cakes for lunch! Honey cakes for dinner! (dance)
Bobo! That's way too many honey cakes! (surprised)""", ["Bobo"]),
         _s("garden", "day", "It was breakfast time in Sunny Garden, and everyone was hungry.", ["enter", "wave"], """
Good morning, everyone! (wave)
Bobo: My tummy says grumble, grumble!
Kiko: Did you know... the best breakfast is a big bowl of candy?
Lulu: Hmm, I don't think so, Kiko.""", ALL),
         _s("kitchen", "day", "Bobo ate honey cakes, cookies and candy. Lots and lots of candy!", ["surprised"], """
Bobo: Ooh... my tummy feels funny. (sad)
Oh no! Poor Bobo. (sad)""", ["sad Bobo"]),
         _s("kitchen", "day", "Sweets are yummy, but only sometimes. Our bodies need healthy food to grow big and "
            "strong.", ["think", "point"], """
Which foods are healthy? Let's find out!""", ["sad Bobo"], board="Healthy food!", item="apple"),
         _s("kitchen", "day", "Which one is a healthy snack? The candy, the carrot, or the cookie?", ["think"], """
The carrot! Crunch, crunch! (happy)
Bobo: Carrots are crunchy! I like them! (eat)""", ["Bobo"], board="Which one is healthy?",
            quiz=["candy", "carrot", "cookie"], answer="carrot"),
         _s("garden", "day", "Lulu found sweet strawberries in the garden. Let's count them together!", ["point"], """
Lulu: Six red strawberries. So pretty! (happy)
Bobo: And so yummy! (eat)""", ["Lulu", "Bobo"], board="Strawberries!", item="strawberry", count=6),
         _s("garden", "day", "Quiz time! Which one is a fruit? The broccoli, the banana, or the ball?", ["think"], """
Kiko: The banana! I got it right! (happy)
Great job, Kiko! (clap)""", ["Kiko"], board="Which one is a fruit?", quiz=["broccoli", "banana", "ball"],
            answer="banana"),
         _s("garden", "day", "Bobo tried a little bit of everything: an apple, a carrot, and some broccoli.",
            ["eat", "happy"], """
Bobo: Broccoli looks like tiny trees!
Bobo: Now I feel super strong! (jump)
Healthy food makes us strong! (happy)""", ["Bobo"], holds="broccoli"),
         _s("meadow", "day", "With full, happy tummies, the friends ran and played all morning!", ["run", "jump"], """
Bobo: Race you to the big tree!
Kiko: I'm the fastest! (happy)""", ALL, effect="confetti"),
     ]},
    {"id": "s1e03-kiko-counts-the-stars", "title": "Kiko Counts the Stars",
     "lesson": "Count slowly, one at a time, and you'll get it right.", "next": "Lulu's Missing Colors",
     "question": "How many stars can you count tonight?", "music": "calm",
     "scenes": [
         _s("meadow", "night", "", ["surprised"], """
Kiko: One, two, a hundred, a million! I counted all the stars! (spin)
Kiko! That was way too fast! (surprised)""", ["Kiko"], effect="stars"),
         _s("garden", "sunset", "The sun was going down in Sunny Garden. Tonight, the friends had a plan: a star "
            "party!", ["enter", "jump"], """
Tonight, we count the stars! (jump)
Bobo: Can we bring snacks?
Lulu: I'll bring my paints!""", ALL),
         _s("meadow", "night", "When it got dark, the first stars began to twinkle.", ["look", "point"], """
Kiko: I'm the best counter in the whole garden! Watch me!
Kiko: One, two, five, nine, twenty, a million! (spin)
Lulu: Kiko... you skipped some numbers.""", ["Lulu", "Kiko"], effect="stars"),
         _s("meadow", "night", "Counting is easier when we go slowly, one at a time. Let's count the stars "
            "together!", ["point"], """
Kiko: Five stars! Counting slowly is fun! (happy)""", ["Kiko"], board="Count slowly!", item="star", count=5),
         _s("meadow", "night", "Now let's try a bigger number. Can you count all the way to ten?", ["point"], """
Bobo: Ten! I counted with my paws! (clap)
Kiko: Wow! Everybody can count!""", ALL, board="Count to ten!", count=10),
         _s("meadow", "night", "Quiz time! Which one comes out at night? The sun, the moon, or the rainbow?",
            ["think"], """
Lulu: The moon! It's so round and bright. (happy)""", ["Lulu"], board="What comes out at night?",
            quiz=["sun", "moon", "rainbow"], answer="moon"),
         _s("meadow", "night", "Kiko looked up at the sky and felt a little bit silly.", ["idle"], """
Kiko: I don't know everything. But I can learn! (sad)
Me too, Kiko! Let's learn together! (happy)""", ["sad Kiko"], effect="hearts"),
         _s("meadow", "night", "{name} spun its leaves and flew up high, to say good night to the stars.",
            ["float"], """
Good night, stars! (float)
Kiko: Good night, one, two, three... all of you! (happy)""", ["Kiko"], effect="stars"),
     ]},
    {"id": "s1e04-lulus-missing-colors", "title": "Lulu's Missing Colors",
     "lesson": "Mixing two colors makes a brand new color!", "next": "Brush, Brush, Bobo!",
     "question": "What is your favorite color?",
     "scenes": [
         _s("garden", "day", "", ["surprised"], """
Lulu: Oh no! My orange paint is all gone! How can I paint the sun? (sad)
Don't worry, Lulu. We'll help you! (wave)""", ["sad Lulu"]),
         _s("garden", "day", "Lulu was painting a big picture of Sunny Garden. She needed every color of the "
            "rainbow.", ["enter", "look"], """
Lulu: I have red, yellow and blue. But no orange. (sad)
Kiko: Did you know... orange paint grows on orange trees?
Lulu: Hmm... I don't think so, Kiko.""", ["Lulu", "Kiko"], board="RED YELLOW BLUE"),
         _s("garden", "day", "{name} had an idea. What happens if we mix two colors together?", ["think",
                                                                                                "surprised"], """
Let's find out! (jump)""", ["Lulu", "Kiko"], effect="idea"),
         _s("garden", "day", "Red and yellow. Mix, mix, mix... and it makes orange!", ["point"], """
Lulu: Orange! Just like a carrot! (happy)""", ["Lulu"], board="RED + YELLOW = ORANGE", effect="stars"),
         _s("garden", "day", "What about blue and yellow? Mix, mix, mix... it makes green!", ["point"], """
Bobo: Green, like broccoli! (happy)
Lulu: Bobo, you always think of food!""", ["Lulu", "Bobo"], board="BLUE + YELLOW = GREEN"),
         _s("garden", "day", "And red and blue make... purple!", ["point", "dance"], """
Purple, just like me! (dance)""", ["Lulu"], board="RED + BLUE = PURPLE", effect="music"),
         _s("garden", "day", "Can you find something green? The broccoli, the carrot, or the strawberry?",
            ["think"], """
Bobo: The broccoli! Green, like a tiny tree! (eat)""", ["Bobo"], board="Which one is green?",
            quiz=["broccoli", "carrot", "strawberry"], answer="broccoli"),
         _s("garden", "day", "Quiz time! Which one is yellow? The apple, the banana, or the broccoli?", ["think"], """
Kiko: The banana! I got it right! (happy)""", ["Kiko", "Lulu"], board="Which one is yellow?",
            quiz=["apple", "banana", "broccoli"], answer="banana"),
         _s("meadow", "day", "Then the sky lit up, and look! A real rainbow, with every color!", ["surprised",
                                                                                                "happy"], """
Lulu: Oh, how pretty!
Lulu: Now I can paint every color! (dance)""", ALL, board="RAINBOW", item="rainbow", effect="confetti"),
         _s("garden", "sunset", "Lulu finished her painting, and everyone loved it.", ["clap"], """
Lulu: Thank you for helping me, friends. (happy)
Your painting is beautiful, Lulu!""", ALL, effect="hearts"),
     ]},
    {"id": "s1e05-brush-brush-bobo", "title": "Brush, Brush, Bobo!",
     "lesson": "Brush your teeth every morning and every night.", "next": "Bubble Trouble",
     "question": "Did you brush your teeth today?", "music": "calm",
     "scenes": [
         _s("room", "night", "", ["surprised"], """
Bobo: Brush my teeth? Nooo! I'm too sleepy! (yawn)
But Bobo, the sugar bugs will have a party! (surprised)""", ["Bobo"]),
         _s("room", "night", "It was a sleepover at {name}'s flowerpot house, and it was almost bedtime.",
            ["enter", "wave"], """
Bobo: I just ate a big honey cake! (happy)
Kiko: Did you know... teeth brush themselves at night?
Lulu: No, Kiko! We have to brush them.""", ALL),
         _s("room", "night", "When we eat sweets, tiny sugar bugs can stay on our teeth. Brushing washes them "
            "away!", ["point"], """
Bobo: Sugar bugs? Eek! (scared)""", ["Bobo"], board="Brush away sugar bugs!", item="tooth"),
         _s("room", "night", "{name} showed everyone how to brush. Up and down, round and round, front and back!",
            ["brush"], """
Brush, brush, brush! (brush)""", ["Bobo"], holds="toothbrush", board="Up, down, round and round!"),
         _s("room", "night", "Let's brush and count to ten together. Ready? Go!", ["brush"], """
Bobo: Ten! My teeth feel so clean! (happy)""", ["Bobo"], holds="toothbrush", board="Brush and count!",
            count=10),
         _s("room", "night", "Quiz time! What do we use to brush our teeth? The soap, the toothbrush, or the "
            "towel?", ["think"], """
Kiko: The toothbrush! Even I knew that one! (happy)""", ["Kiko"], board="What do we brush with?",
            quiz=["soap", "toothbrush", "towel"], answer="toothbrush"),
         _s("room", "night", "Then a sip of water. Swish, swish, and spit!", ["eat", "happy"], """
Bobo: Look at my shiny teeth! (happy)
Lulu: So sparkly, Bobo!""", ["Lulu", "Bobo"], holds="cup", effect="stars"),
         _s("room", "night", "With clean teeth and cozy blankets, the friends were ready to sleep.",
            ["yawn", "talk", "sleep"], """
Good night, everyone!
Bobo: Good night, teeth! (yawn)""", ["sleepy Lulu", "sleepy Kiko", "Bobo"], effect="stars"),
     ]},
    {"id": "s1e06-bubble-trouble", "title": "Bubble Trouble",
     "lesson": "Wash your hands with soap after you play and before you eat.", "next": "Shapes in the Sky",
     "question": "When do you wash your hands?",
     "scenes": [
         _s("garden", "day", "", ["surprised"], """
Kiko: Bobo! Your paws are all muddy!
Bobo: Mud cookies! Yum! (eat)
Stop, Bobo! Wash your hands first! (surprised)""", ["Kiko", "Bobo"], effect="exclaim"),
         _s("garden", "day", "After the rain, Sunny Garden was full of puddles. Splish, splash!",
            ["jump", "happy", "jump"], """
Bobo: Let's make mud cookies! (jump)
Lulu: I love jumping in puddles! (happy)""", ALL),
         _s("garden", "day", "Everyone's hands were brown and muddy. And mud has tiny germs that we cannot see!",
            ["look"], """
Lulu: Germs? I don't see any. (look)
They are too tiny to see! Let's wash them away.""", ["Lulu", "Bobo"], board="Germs are tiny!"),
         _s("kitchen", "day", "First, wet your hands. Then add soap and make lots of bubbles!", ["wash"], """
Bobo: Bubbles! So many bubbles! (happy)""", ["Bobo"], holds="soap"),
         _s("kitchen", "day", "Rub the fronts, rub the backs, and in between your fingers. Let's count to ten!",
            ["wash"], """
Bobo: Ten! My paws are squeaky clean! (clap)""", ["Bobo"], board="Scrub, scrub, scrub!", count=10),
         _s("kitchen", "day", "Quiz time! What do we dry our hands with? The towel, the book, or the ball?",
            ["think"], """
Kiko: The towel! Pat, pat, pat. (happy)""", ["Kiko", "Lulu"], board="What dries our hands?",
            quiz=["towel", "book", "ball"], answer="towel"),
         _s("kitchen", "day", "With clean hands, it was finally snack time!", ["eat", "happy"], """
Bobo: My favorite time of day! (dance)
Clean hands, happy tummies!""", ALL, holds="apple", effect="hearts"),
     ]},
    {"id": "s1e07-shapes-in-the-sky", "title": "Shapes in the Sky",
     "lesson": "A circle is round, a square has four sides, and a triangle has three.",
     "next": "My Turn, Your Turn", "question": "What shapes can you find in your room?",
     "scenes": [
         _s("meadow", "day", "", ["look"], """
Kiko: Look! That cloud is a giant flying cupcake!
Lulu: No, Kiko. I think it's a bunny! (happy)""", ["Lulu", "Kiko"]),
         _s("meadow", "day", "It was a breezy day, perfect for watching clouds.", ["enter", "point"], """
Clouds come in so many shapes! Let's find out about shapes. (point)""", ALL),
         _s("meadow", "day", "A circle is round, round, round, like the sun and like a ball.", ["point"], """
Bobo: Like a round cookie! (happy)""", ["Bobo"], board="CIRCLE", item="circle"),
         _s("meadow", "day", "A square has four sides that are all the same. Let's count the sides!", ["point"], """
Lulu: One, two, three, four. A window is a square!""", ["Lulu"], board="A SQUARE has 4 sides", count=4),
         _s("meadow", "day", "A triangle has three sides and three pointy corners.", ["point"], """
Kiko: Did you know... a triangle has five sides?
Let's count, Kiko. One, two, three!
Kiko: Oops! Three sides! (surprised)""", ["Kiko"], board="TRIANGLE", item="triangle"),
         _s("meadow", "day", "Shape quiz! Which one is a circle?", ["think"], """
Bobo: The circle! Round, like my tummy! (happy)""", ["Bobo"], board="Which one is a circle?",
            quiz=["square", "triangle", "circle"], answer="circle"),
         _s("meadow", "day", "Some shapes are extra special. A star twinkles, and a heart means I love you.",
            ["happy"], """
Lulu: I'll paint a heart for all my friends. (happy)""", ["Lulu"], board="HEART", item="heart",
            effect="hearts"),
         _s("meadow", "sunset", "As the sun went down, the friends found shapes everywhere they looked.",
            ["dance"], """
Circle, square, triangle, star! (dance)
Kiko: Shapes are everywhere! (happy)""", ALL, effect="confetti"),
     ]},
    {"id": "s1e08-my-turn-your-turn", "title": "My Turn, Your Turn",
     "lesson": "When we take turns and share, everyone gets to play.", "next": "Peeko Feels Grumpy",
     "question": "What do you like to share with your friends?",
     "scenes": [
         _s("playground", "day", "", ["surprised"], """
Kiko: The slide is mine!
Bobo: No, it's mine! (angry)
Uh-oh. Everyone wants the slide! (scared)""", ["Kiko", "Bobo"]),
         _s("playground", "day", "The friends went to the playground. There was a slide, swings, and a sandbox!",
            ["enter", "happy"], """
Lulu: Wow! Look at the big red slide!
Kiko: Me first! Me first! (jump)""", ALL),
         _s("playground", "day", "Kiko and Bobo both wanted to go first. Soon, nobody was having fun.", ["sad"], """
Bobo: I want a turn too! (sad)
Kiko: But I was here first! (angry)""", ["sad Lulu", "Kiko", "Bobo"]),
         _s("playground", "day", "{name} had an idea. What if everyone takes turns?", ["think", "surprised"], """
My turn, your turn! Then everybody gets to play! (point)""", ["Kiko", "Bobo"], effect="idea"),
         _s("playground", "day", "Let's count while Kiko slides. When we get to five, it's Bobo's turn!",
            ["point"], """
Bobo: Five! My turn! Wheee! (jump)""", ["Kiko", "Bobo"], board="Take turns!", count=5),
         _s("playground", "day", "Then Lulu shared her ball with everyone.", ["happy"], """
Lulu: Would you like to play with my ball?
Bobo: Yes, please! Thank you, Lulu! (happy)""", ["Lulu", "Bobo"], holds="ball", effect="hearts"),
         _s("playground", "day", "Some things are fun to share, and some things are just for you. Which one can we "
            "share with friends?", ["think"], """
Kiko: The ball! We never share toothbrushes. Eww! (happy)""", ["Kiko"], board="What can we share?",
            quiz=["toothbrush", "ball"], answer="ball"),
         _s("playground", "sunset", "Taking turns made the whole day fun. Everyone got to play!", ["dance"], """
Kiko: Sorry I didn't share, Bobo. (sad)
Bobo: That's okay, Kiko. Friends forever! (happy)""", ALL, effect="confetti"),
     ]},
    {"id": "s1e09-peeko-feels-grumpy", "title": "{name} Feels Grumpy",
     "lesson": "When you feel angry, take slow breaths and tell someone how you feel.",
     "next": "Seeds and Sunshine", "question": "What do you do when you feel grumpy?",
     "scenes": [
         _s("room", "day", "", ["angry"], """
No, no, NO! Everything is going wrong today! (angry)
Lulu: Oh dear. {name} looks so grumpy.""", ["Lulu"]),
         _s("garden", "day", "It was a rainy day. {name} wanted to fly outside, but everything was too wet.",
            ["sad", "angry"], """
I wanted to fly today. It's not fair! (angry)""", weather="rain"),
         _s("room", "day", "Then {name}'s tower of blocks fell down. Crash!", ["surprised", "angry"], """
Grr! I'm so grumpy! (angry)
Lulu: It's okay to feel angry, {name}.""", ["Lulu"]),
         _s("room", "day", "Feeling angry is okay. Everybody feels angry sometimes. But what can we do to feel "
            "better?", ["think"], """
Bobo: When I'm grumpy, I take big, slow breaths.""", ["Lulu", "Bobo"], board="It's okay to feel angry"),
         _s("room", "day", "Let's breathe together. Breathe in, and out. Let's count five slow breaths.",
            ["idle"], """
Five. I feel much calmer now. (happy)""", ["Lulu", "Bobo"], board="Slow breaths", count=5),
         _s("room", "day", "Another good idea: tell a friend how you feel.", ["talk", "happy"], """
I feel sad because I can't play outside.
Kiko: Did you know... we can play inside too!
Lulu: Let's build a new tower, together! (happy)""", ALL),
         _s("room", "day", "Quiz time! Which shape means love and big hugs?", ["think"], """
Bobo: The heart! Big bear hugs for everyone! (happy)""", ["Bobo"], board="Which one means love?",
            quiz=["triangle", "heart", "square"], answer="heart", effect="hearts"),
         _s("garden", "day", "And then, the rain stopped, and the sun came out again!", ["surprised", "float"], """
Leaves, spin! Let's play! (float)
Bobo: Hooray! Sunshine! (jump)""", ALL, effect="confetti"),
     ]},
    {"id": "s1e10-seeds-and-sunshine", "title": "Seeds and Sunshine",
     "lesson": "Plants need soil, water, sunshine and time to grow.", "next": "Safe Steps to the Park",
     "question": "What would you like to grow?",
     "scenes": [
         _s("garden", "day", "", ["think"], """
Kiko: {name}, where did you come from? Did you hatch from an egg?
No, Kiko! I grew from a tiny seed! (happy)""", ["Kiko"]),
         _s("garden", "day", "Today, the friends wanted to plant a garden of their own.", ["enter", "wave"], """
Bobo: Let's plant a honey tree!
Lulu: Let's plant flowers! (happy)
Let's find out how seeds grow! (jump)""", ALL),
         _s("garden", "day", "First, we put a tiny seed into the soil. Dig, dig, dig!", ["point"], """
Bobo: I'm a great digger! (happy)""", ["Bobo"], board="1. Plant the seed"),
         _s("garden", "day", "Second, seeds need water. Pitter, patter!", ["happy"], """
Lulu: Here's a drink, little seed.""", ["Lulu"], holds="cup", board="2. Water", item="cup"),
         _s("garden", "day", "Third, seeds need warm sunshine.", ["point"], """
Kiko: Did you know... seeds wear tiny sunglasses?
Ha ha! No, Kiko. They just love the warm sun! (happy)""", ["Kiko"], board="3. Sunshine", item="sun"),
         _s("garden", "day", "And last, seeds need time. Let's count the days while they grow!", ["point"], """
Bobo: Seven days is a long time to wait! (sad)""", ["Bobo"], board="Days of growing", count=7),
         _s("garden", "day", "Then, one morning... pop! Little green sprouts!", ["surprised", "happy"], """
Lulu: Oh, how pretty!
They look just like me! (dance)""", ALL, effect="stars"),
         _s("garden", "day", "Quiz time! What do plants need to grow? Candy, water, or a ball?", ["think"], """
Bobo: Water! Not candy. I knew that! (happy)""", ["Bobo"], board="What do plants need?",
            quiz=["candy", "cup", "ball"], answer="cup"),
     ]},
    {"id": "s1e11-safe-steps-to-the-park", "title": "Safe Steps to the Park",
     "lesson": "Stop, look left and right, and cross the street with a grown-up.",
     "next": "The Sunny Garden Festival", "question": "What does a red light mean?",
     "scenes": [
         _s("city", "day", "", ["surprised"], """
Kiko: The park is right there! Let's zoom across the road! (happy)
Stop, Kiko! Cars are coming! (surprised)""", ["Kiko"], effect="exclaim"),
         _s("city", "day", "The friends were going to the big park in town. To get there, they had to cross a "
            "busy street.", ["enter", "walk"], """
Bobo: The park has an ice cream cart! (jump)""", ALL),
         _s("city", "day", "At the street, there was a traffic light. Red means stop!", ["point"], """
Red means stop. We wait! (point)
Lulu: Stop, and wait.""", ["Lulu"], board="RED means STOP", item="red light"),
         _s("city", "day", "Before we cross, we look left, look right, and look left again.",
            ["look", "look", "look"], """
Left, right, left!
Bobo: No cars! (happy)""", ["Bobo"]),
         _s("city", "day", "While we wait for the green light, let's count to five together!", ["point"], """
Bobo: Five! Look, the light is changing! (surprised)""", ["Lulu", "Bobo"], board="Wait for green...", count=5),
         _s("city", "day", "And green means go! Hold a grown-up's hand and walk across.", ["walk"], """
Lulu: Green means go! (happy)""", ["Lulu", "Bobo"], board="GREEN means GO", item="green light"),
         _s("city", "day", "Quiz time! Which light tells us to stop?", ["think"], """
Kiko: The red light! Stop, look and listen! (happy)
Kiko: Sorry I almost flew across, {name}. (sad)
That's okay, Kiko. Now you know!""", ["Kiko"], board="Which light means stop?",
            quiz=["green light", "red light"], answer="red light"),
         _s("meadow", "day", "They made it safely to the park, just in time for a picnic!", ["run", "jump"], """
Bobo: We did it! Now, where's that ice cream? (dance)""", ALL, effect="confetti"),
     ]},
    {"id": "s1e12-the-sunny-garden-festival", "title": "The Sunny Garden Festival",
     "lesson": "Learning is more fun with friends!", "next": "",
     "question": "Which episode did you like best?",
     "scenes": [
         _s("garden", "sunset", "", ["jump"], """
Eleven flowers! Just one more, and it's festival time! (jump)
Kiko: What will the last lesson be? (think)""", ["Kiko"], effect="stars"),
         _s("garden", "day", "{name} had learned so many things. Just one more flower, and it would be time for "
            "the big Sunny Garden Festival!", ["enter", "wave"], """
Lulu: Let's play a game! A quiz about everything we learned! (happy)""", ALL),
         _s("garden", "day", "First question. Which one helps us grow big and strong?", ["think"], """
Bobo: The carrot! Easy peasy! (eat)""", ["Bobo"], board="Grow big and strong!", quiz=["candy", "carrot"],
            answer="carrot"),
         _s("garden", "day", "Next question. What do we use every morning and every night?", ["think"], """
Kiko: The toothbrush! Brush, brush, brush! (happy)""", ["Kiko"], board="Every morning and night",
            quiz=["toothbrush", "ball", "teddy"], answer="toothbrush"),
         _s("garden", "day", "Last question! Which one is a triangle?", ["think"], """
Lulu: The triangle! Three sides! (happy)""", ["Lulu"], board="Which one is a triangle?",
            quiz=["circle", "triangle", "star"], answer="triangle"),
         _s("garden", "day", "Now let's count all the friends in Sunny Garden, one last time!", ["point"], """
Four friends, together! (clap)""", ALL, board="Our friends", count=4),
         _s("garden", "sunset", "{name} thought about everything they had learned together, and then {name} "
            "understood the last lesson.", ["think", "surprised"], """
The best thing I learned is... learning is more fun with friends! (happy)
Lulu: Oh, {name}! (happy)""", ALL, effect="idea"),
         _s("garden", "night", "Pop! The twelfth flower bloomed. It was time for the Sunny Garden Festival!",
            ["dance", "spin"], """
Bobo: There are snacks! (dance)
Lulu: And music! (dance)
Kiko: Did you know... I'm a great dancer? (dance)
Yes, Kiko! That one is true! (happy)""", ALL, effect="confetti"),
         _s("garden", "night", "The friends danced under the stars until it was time to say good night.",
            ["wave", "talk", "sleep"], """
Thank you for learning with us! See you next season! (wave)
Bobo: Good night, everyone! (yawn)""", ["sleepy Lulu", "sleepy Kiko", "Bobo"], effect="stars"),
     ]},
]
