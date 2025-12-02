#goal: to analyze the datasets to compute basic stats
#of the newly cleaned data sets but also to filter out
#the stories that would be best for my goal of training on a grimdark fantasy
#using the newly created threshold

#importing necessary packages
import json

#loading the just scored data sets
with open("train_cleaned.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("valid_cleaned.json", "r", encoding="utf-8") as f:
    valid_data = json.load(f)

with open("test_cleaned.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

#these are the fantasy keywords we are filtering out for
fantasy_keywords = {
    #using same metrics as scoring
    "magic": 4, "dragon": 5, "witch": 4, "wizard": 4, "sorcerer": 4,
    "fairy": 3, "lich": 5, "orc": 4, "kingdom": 3, "sword": 3, "dungeon": 4,
    "curse": 4, "blade": 3, "undead": 4, "realm": 3, "haunted": 3, "dark": 3,
    "sin": 3, "vile": 3, "corrupt": 3, "treason": 3, "rotten": 3, "dogma": 2,
    "siege": 3, "skirmish": 3, "war-torn": 4, "grim": 4, "saber": 3, "tyrant": 4,
    "whip": 3, "filth": 2, "gothic": 3, "citadel": 3, "eerie": 3, "sullen": 2,
    "foul": 2, "demon": 5, "horror": 5, "power": 3, "powers": 3, "voice": 2,
    "flayed": 4, "torment": 4, "tormented": 4, "regime": 3, "blood": 4,
    "spear": 3, "dagger": 3, "wound": 3, "wounds": 3, "scar": 2, "scars": 2,
    "cadaver": 4, "cadavers": 4, "corpse": 4, "corpses": 4, "grave": 3,
    "tomb": 3, "gore": 4, "tyranny": 4, "oppression": 4, "oppressive": 4,
    "throne": 3, "crown": 3, "thrones": 3, "dominion": 3, "empire": 3,
    "usurper": 4, "deceit": 3, "conspiracy": 3, "overlord": 4, "lord": 3,
    "lords": 3, "greed": 3, "altar": 2, "shrine": 2, "idol": 2, "relic": 2,
    "martyr": 3, "penance": 2, "zombie": 4, "ghoul": 4, "wraith": 4,
    "fiend": 4, "devil": 4, "hellspawn": 5, "evil": 4, "otherworldly": 4,
    "forbidden": 3, "monster": 4, "monsters": 4, "haunting": 3, "hex": 3,
    "malediction": 4, "eldritch": 5, "possession": 4, "possessed": 4, "misery": 3,
    "despair": 3, "chain": 2, "chains": 2, "bleak": 3, "brutal": 3, "savage": 3,
    "tragic": 3, "fatal": 3, "foreboding": 3, "nihilistic": 4, "cynical": 3,
    "sinister": 4, "malevolent": 4, "spirit": 3, "spirits": 3, "spell": 3,
    "spells": 3, "witches": 4, "wizards": 4, "damned": 4, "doomed": 4,
    "cursed": 4, "curses": 4, "magical": 3, "forest": 3, "despairing": 3,
    "ominous": 3, "warlock": 4, "graves": 3, "cemetery": 3, "necromancy": 5,
    "necromancer": 5, "blasted": 3, "ruin": 3, "ruined": 3, "vampiric": 4,
    "vampire": 4, "runes": 3, "ancient": 3, "acolyte": 3, "leeches": 2,
    "golem": 4, "gargoyle": 4, "powerful": 3, "apothecary": 2, "hollow": 2,
    "zealot": 3, "fanatic": 3, "nether": 3, "underworld": 4, "catacomb": 4,
    "defile": 3, "defiled": 3, "sacrifice": 4, "elf": 3, "king": 3, "queen": 3,
    "elves": 3, "human": 2, "humans": 2, "dwarf": 3, "dwarves": 3, "fairies": 3,
    "fae": 3, "goblin": 3, "goblins": 3, "centaur": 3, "centaurs": 3,
    "bugbear": 3, "gnome": 3, "gnomes": 3
}

prompt_weight = 0.6
story_weight = 0.4

#calculating fantasy scores
def calculate_fantasy_score(item):
    prompt_text = item["prompt"].lower()
    story_text = item["story"].lower()
    score = 0.0
    for word, weight in fantasy_keywords.items():
        score += prompt_weight * weight * prompt_text.count(word)
        score += story_weight * weight * story_text.count(word)
    return score

#filtering through datasets by my threshold of 25
def filter_by_threshold(data, threshold=25):
    filtered = []
    for item in data:
        score = calculate_fantasy_score(item)
        if score >= threshold:
            filtered.append(item)
    return filtered

#applying the filtering
threshold_value = 25

train_filtered = filter_by_threshold(train_data, threshold_value)
valid_filtered = filter_by_threshold(valid_data, threshold_value)
test_filtered = filter_by_threshold(test_data, threshold_value)

#now saving new datasets
with open("train_fantasy.json", "w", encoding="utf-8") as f:
    json.dump(train_filtered, f, ensure_ascii=False, indent=2)

with open("valid_fantasy.json", "w", encoding="utf-8") as f:
    json.dump(valid_filtered, f, ensure_ascii=False, indent=2)

with open("test_fantasy.json", "w", encoding="utf-8") as f:
    json.dump(test_filtered, f, ensure_ascii=False, indent=2)

print(f"Filtered datasets saved. Train: {len(train_filtered)}, Valid: {len(valid_filtered)}, Test: {len(test_filtered)}")
