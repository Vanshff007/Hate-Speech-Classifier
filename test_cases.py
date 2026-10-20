"""500 labelled sentences used to check the model on everyday messages.

Each case is (text, is_toxic, category). The compliment, negation and slang
words here are kept out of the fine-tuning data (see finetune.py), so the
suite measures whether the model generalises instead of memorising.
"""

COMPLIMENT_WORDS = [
    "cute", "beautiful", "kind", "smart", "funny", "amazing", "lovely", "sweet",
    "brilliant", "talented", "wonderful", "gorgeous", "handsome", "charming",
    "awesome", "generous", "caring", "hardworking", "creative", "inspiring",
]
COMPLIMENT_TEMPLATES = [
    "you are {}", "you're so {}", "ur {}", "I think you are really {}", "you look {} today",
]

NEUTRAL = [
    "Can you send me the report by Friday?", "The meeting has been moved to 3pm.",
    "I had pasta for dinner tonight.", "What time does the train leave?",
    "Please review my pull request when you get a chance.", "The weather is nice today.",
    "I am going to the gym after work.", "Thanks for the quick reply.",
    "Does anyone know a good book about history?", "The article needs more citations.",
    "I added a reference to the second paragraph.", "Let's grab coffee tomorrow morning.",
    "My cat sleeps all day.", "The library opens at nine.", "I just finished my homework.",
    "Could you explain this formula again?", "We should update the documentation.",
    "The football match was exciting.", "I moved to a new apartment last week.",
    "Happy birthday, have a great year!", "Good morning everyone.", "See you at lunch.",
    "This recipe needs more salt.", "I'm learning to play the guitar.",
    "The bus was late again today.", "Can we reschedule our call?",
    "I like reading science fiction.", "Our team won the quiz.", "The store is closed on Sunday.",
    "I watched a documentary about whales.", "Please merge the branch after the tests pass.",
    "The new phone has a great camera.", "My sister is visiting next month.",
    "I fixed the typo in the heading.", "We planted tomatoes in the garden.",
    "The museum has a new exhibition.", "I need to buy groceries.", "Traffic was light this morning.",
    "The movie starts at eight.", "Did you get my email?", "I agree with your edit.",
    "This page could use a picture.", "The source you cited is reliable.",
    "I will be on vacation next week.", "Let me know if you need any help.",
    "The printer is out of paper.", "Hello, how are you?", "Thank you so much!",
    "Nice to meet you.", "Have a safe trip.", "I baked cookies for the office.",
    "The concert was sold out.", "My dog loves long walks.", "It is raining again.",
    "I prefer tea over coffee.", "The deadline is next Monday.", "We are out of milk.",
    "The children are playing outside.", "Welcome to the team!", "Congratulations on your promotion!",
    "Can you pass the salt?", "I'm reading a book about space.", "The flowers are blooming.",
    "Please sign the form at the bottom.", "The train was very crowded.",
    "I'm proud of what we built.", "Let's discuss this on the talk page.",
    "The sources disagree on the date.", "I reverted the edit because it lacked a source.",
    "Good night, sleep well.", "I think the section should be shorter.", "Enjoy your weekend!",
    "The results look promising.", "What do you think about the new design?",
    "I appreciate your feedback.", "We need more volunteers.", "The café has free wifi.",
    "I'm excited for the holidays.", "The bridge was built in 1920.", "Our flight lands at noon.",
]

IDENTITIES = [
    "gay", "lesbian", "muslim", "jewish", "christian", "black", "white", "asian",
    "mexican", "indian", "trans", "a woman", "an immigrant", "hindu", "african",
]
IDENTITY_TEMPLATES = [
    "I am {}", "my best friend is {}", "being {} is nothing to be ashamed of",
    "my neighbor is {} and very nice",
]

NEGATED_WORDS = [
    "stupid", "dumb", "ugly", "an idiot", "a loser", "annoying", "useless",
    "pathetic", "worthless", "a fool",
]
NEGATION_TEMPLATES = [
    "you are not {}", "you're not {} at all", "I never said you were {}", "nobody thinks you are {}",
]

SLANG = [
    "this is sick", "what the hell", "I hate mondays", "you killed it bro", "damn that's good",
    "that party was insane", "this song is fire", "I'm dying of laughter", "that's crazy good",
    "you nailed it", "hell yeah", "this game is so addictive", "I'm starving",
    "that exam killed me", "I could murder a pizza right now", "the traffic is killing me",
    "she slayed that performance", "he's a beast at chess", "I hate it when it rains",
    "oh my god this is amazing", "damn, I forgot my keys", "this bug is driving me crazy",
    "that joke was savage", "we crushed the competition", "stop, you're killing me haha",
    "that is wicked cool", "I'm dead tired", "this weather sucks", "the movie was bloody brilliant",
    "that shot was deadly", "I'm mad about this new album", "what a crazy day",
    "this pizza is the bomb", "he's a monster on the court", "I hate spiders",
    "that was a stupid mistake on my part", "I feel so dumb for forgetting", "ugh, mondays",
    "the ending was brutal", "holy cow that's big",
]

INSULTS = [
    "a stupid idiot", "a worthless loser", "a pathetic moron", "an ugly pig", "a dumb clown",
    "a disgusting creep", "a useless piece of trash", "a brainless fool", "a lying scumbag",
    "a filthy rat", "an absolute idiot", "a total moron", "a fat slob", "a miserable jerk",
    "a spineless coward", "a braindead troll", "a waste of space", "a pathetic joke",
    "an ignorant fool", "a dumbass",
]
INSULT_TEMPLATES = [
    "you are {}", "you're such {}", "everyone knows you are {}", "go away, you are {}",
]

THREAT_VERBS = [
    "kill", "murder", "stab", "shoot", "beat", "strangle", "choke", "hurt", "punch", "burn",
]
THREAT_TEMPLATES = [
    "I will {} you", "I am going to {} you tonight", "watch your back or I will {} you",
    "someone should {} you",
]

HATED_GROUPS = [
    "gays", "muslims", "jews", "black people", "immigrants", "mexicans", "asians", "women",
    "trans people", "indians",
]
HATE_TEMPLATES = [
    "all {} should die", "{} are disgusting animals", "I hate {} they are subhuman",
]

OBSCENE = [
    "fuck you", "go fuck yourself", "you piece of shit", "suck my dick", "shut the fuck up",
    "fuck off and die", "you stupid bitch", "eat shit", "what a fucking idiot", "you dumb fuck",
    "kiss my ass", "you are a cunt", "fucking loser", "screw you asshole", "you little shit",
    "motherfucker get lost", "you are a dickhead", "bullshit you liar", "stfu you whore",
    "you fucking moron", "go to hell you bastard", "damn you bitch", "piss off wanker",
    "shut your fucking mouth", "you absolute prick", "fuck this and fuck you", "you slut",
    "eat a dick", "you are full of shit", "fuck your stupid opinion",
]


def build_cases():
    cases = []
    cases += [(t.format(w), False, "compliment") for t in COMPLIMENT_TEMPLATES for w in COMPLIMENT_WORDS]
    cases += [(s, False, "neutral") for s in NEUTRAL]
    cases += [(t.format(w), False, "identity_neutral") for t in IDENTITY_TEMPLATES for w in IDENTITIES]
    cases += [(t.format(w), False, "negation") for t in NEGATION_TEMPLATES for w in NEGATED_WORDS]
    cases += [(s, False, "benign_slang") for s in SLANG]
    cases += [(t.format(w), True, "insult") for t in INSULT_TEMPLATES for w in INSULTS]
    cases += [(t.format(w), True, "threat") for t in THREAT_TEMPLATES for w in THREAT_VERBS]
    cases += [(t.format(w), True, "identity_hate") for t in HATE_TEMPLATES for w in HATED_GROUPS]
    cases += [(s, True, "obscene") for s in OBSCENE]
    return cases


CASES = build_cases()
assert len(CASES) == 500, len(CASES)
