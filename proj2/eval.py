import os

# מאפשר לפייתון לעבוד עם התעודות של Windows (נדרש בגלל NetFree)
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from langsmith import Client

# משתמשים במה שכבר כתבנו ב-agent.py
from agent import TOPICS, BASE_PROMPT, list_images, ask_model, classify_app

load_dotenv()
client = Client()

# ---------- הגדרות ----------
DATASET_FOLDER = r"C:\Users\User\Documents\Dvori\AI\proj2\dataset"  # תיקיית הדאטהסט (שם כל תיקייה = התשובה הנכונה)
DATASET_NAME = "screenshot-topics"
RUN_EXPERIMENT = 2  # איזה ניסוי להריץ: 1 = פרומפט פשוט בלי הענף, 2 = הגדרות לכל נושא ועם הענף

# הפרומפט הפשוט של ניסוי 1: רק רשימת הנושאים, בלי הגדרות ובלי unsure
SIMPLE_PROMPT = BASE_PROMPT + (
    "Use 'other' if none of the other topics fit. Answer with the topic name only."
)


# בונה ב-LangSmith דאטהסט מתיקיית dataset: לכל תמונה נשמר הנתיב שלה והנושא הנכון (שם התיקייה)
# התמונה עצמה לא נשמרת, ובהמשך המודל יקבל רק אותה, לעולם לא את שם התיקייה
def build_dataset():
    # אם הדאטהסט כבר קיים לא נוגעים בו, כדי ששני הניסויים ירוצו על אותו דאטהסט בדיוק
    if client.has_dataset(dataset_name=DATASET_NAME):
        print("dataset already exists, not changing it")
        return

    examples = []
    skipped_folders = set()
    for path in list_images(DATASET_FOLDER):  # תיקייה ריקה לא מוסיפה כלום
        topic = os.path.basename(os.path.dirname(path))  # שם התיקייה של התמונה
        if topic in TOPICS:
            examples.append({"inputs": {"image_path": path}, "outputs": {"topic": topic}})
        else:
            skipped_folders.add(topic)  # תיקייה ששמה לא ברשימת הנושאים

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Screenshots with the correct topic (the folder name)",
    )
    client.create_examples(dataset_id=dataset.id, examples=examples)

    print("examples created:", len(examples))
    for topic in TOPICS:
        count = sum(1 for e in examples if e["outputs"]["topic"] == topic)
        print(f"  {topic}: {count}")
    if skipped_folders:
        print("skipped (folder name not in TOPICS):", skipped_folders)


# ניסוי 1: קריאה אחת למודל עם הפרומפט הפשוט, בלי גרף ובלי הענף
# מקבלת את ה-inputs של דוגמה מהדאטהסט, והמודל רואה רק את התמונה
def target_simple(inputs: dict):
    topic = ask_model(inputs["image_path"], SIMPLE_PROMPT).lower()
    return {"topic": topic}


# ניסוי 2: הגרף המלא מ-agent.py (פרומפט עם הגדרות לכל נושא, ועם הענף של unsure)
def target_full(inputs: dict):
    result = classify_app.invoke({"image_path": inputs["image_path"]})
    return {"topic": result["topic"]}


# ה-evaluator: האם הנושא שהוחזר זהה בדיוק לנושא הנכון (True או False לכל תמונה)
def exact_match(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    return outputs["topic"] == reference_outputs["topic"]


if __name__ == "__main__":
    build_dataset()
    if RUN_EXPERIMENT == 1:
        client.evaluate(
            target_simple,
            data=DATASET_NAME,
            evaluators=[exact_match],
            experiment_prefix="1-simple-prompt-no-branch",
        )
    else:
        client.evaluate(
            target_full,
            data=DATASET_NAME,
            evaluators=[exact_match],
            experiment_prefix="2-definitions-with-branch",
        )