import os
import base64
import random
import mimetypes
from typing import TypedDict
import shutil
# מאפשר לפייתון לעבוד עם התעודות של Windows (נדרש בגלל NetFree)
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

load_dotenv()
client = OpenAI()

# ---------- הגדרות: כל מה שאפשר לשנות נמצא כאן ----------
MODEL = "gpt-4.1"
TOPICS = ["meeting", "code", "site_not_loading", "site_guide", "shop_item", "other"]
SCAN_FOLDER = r"C:\Users\User\Pictures\Screenshots"  # התיקייה לסריקה
IMAGES_TO_RUN = None # כמה תמונות אקראיות להריץ (None = כל התמונות בתיקייה)
BATCH_SIZE = 7  # כל כמה תמונות מבקשים ממך אישור
MOVE_FILES=True
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# תחילת ההוראה למודל, משותפת לשתי הקריאות
BASE_PROMPT = (
    "Classify this screenshot into exactly one topic from this list: "
    + ", ".join(TOPICS) + ". "
)
# הגדרה קצרה לכל נושא, כדי שהמודל יבין מה כל נושא אומר (משמש בניסוי 2 ובסוכן עצמו)
TOPIC_DEFINITIONS = (
    "Topic definitions:\n"
    "- meeting: a screenshot of a meeting or a video call (for example Zoom).\n"
    "- code: a screenshot related to programming: code, error messages, "
    "a code editor or IDE (for example VS Code), a terminal, or developer tools "
    "and extensions.\n"
    "- site_not_loading: a screenshot of a website that does not open or load "
    "(error pages, blank pages, connection errors).\n"
    "- site_guide: a screenshot of a website kept to know what to click or where the "
    "buttons are; it shows the general page or layout, not one specific item.\n"
    "- shop_item: a screenshot of one specific item on a shopping site; "
    "the focus is on that single item.\n"
    "- other: anything that does not fit the topics above.\n"
)
# סיווג ראשון: other הוא נושא אמיתי, ואם המודל לא בטוח הוא עונה unsure
FIRST_PROMPT = BASE_PROMPT + TOPIC_DEFINITIONS + (
    "Use 'other' for screenshots that clearly don't belong to any other topic. "
    "If you are not sure which topic fits, answer 'unsure'. "
    "Answer with the topic name only."
)
# סיווג מחדש (אחרי קריאת הטקסט): חייבים לבחור נושא, unsure כבר לא אפשרי
SECOND_PROMPT = BASE_PROMPT + TOPIC_DEFINITIONS + (
    "Use 'other' only if none of the other topics fit. "
    "Answer with the topic name only."
)


# ---------- גרף 1: סיווג תמונה אחת (רץ בלי עצירות, וזה הגרף שנעריך) ----------

# המידע שעובר בין הצמתים: רק נתיב התמונה והנושא (קצרים, כך שה-traces נשארים קטנים)
class State(TypedDict):
    image_path: str
    topic: str


# מחזירה רשימת נתיבים של כל התמונות בתיקייה (כולל תת-תיקיות)
# תיקייה ריקה פשוט לא מוסיפה כלום, ולכן אין קריסה
def list_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                images.append(os.path.join(root, name))
    return images


# ממירה קובץ תמונה לטקסט base64, הצורה שבה שולחים תמונה למודל
def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)  # למשל image/png
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# שולחת למודל תמונה והוראה, ומחזירה את התשובה שלו כטקסט
# לעולם לא שולחים את שם הקובץ או התיקייה, כדי שהמודל לא "יעתיק" את התשובה מהם
def ask_model(image_path, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        }],
    )
    return response.choices[0].message.content.strip()


# צומת 1 (סריקה): בודק שהקובץ הוא תמונה תקינה, ומאפס את הנושא
def scan_node(state: State):
    with Image.open(state["image_path"]) as img:
        img.verify()  # זורק שגיאה אם הקובץ פגום
    return {"topic": ""}


# צומת 2 (סיווג): שואל את המודל איזה נושא מתאים לתמונה (או unsure אם הוא לא בטוח)
def classify_node(state: State):
    topic = ask_model(state["image_path"], FIRST_PROMPT).lower()
    return {"topic": topic}


# צומת 3 (קריאת טקסט): רץ רק כשהמודל ענה unsure
# קודם מבקש מהמודל לקרוא את הטקסט שבתמונה, ואז מסווג שוב כשהטקסט מצורף
def read_text_node(state: State):
    text = ask_model(
        state["image_path"],
        "Read all the text visible in this screenshot. Return only that text.",
    )
    prompt = f"The text found in the screenshot is:\n{text}\n\n{SECOND_PROMPT}"
    topic = ask_model(state["image_path"], prompt).lower()
    return {"topic": topic}


# הענף המותנה: רק כשהמודל לא בטוח עוברים לקריאת טקסט, אחרת מסיימים
def route_after_classify(state: State):
    if state["topic"] == "unsure":
        return "read_text"
    return END

# בניית גרף הסיווג: התחלה ← סריקה ← סיווג ← (אם unsure: קריאת טקסט) ← סוף
classify_builder = StateGraph(State)
classify_builder.add_node("scan", scan_node)
classify_builder.add_node("classify", classify_node)
classify_builder.add_node("read_text", read_text_node)
classify_builder.add_edge(START, "scan")
classify_builder.add_edge("scan", "classify")
classify_builder.add_conditional_edges("classify", route_after_classify)
classify_builder.add_edge("read_text", END)
classify_app = classify_builder.compile()


# ---------- גרף 2: אישור אדם לקבוצת תמונות ----------

# המידע בגרף האישור: מילון של נתיב תמונה ← הנושא שהוצע לה
class ReviewState(TypedDict):
    suggestions: dict[str, str]


# צומת האישור: עוצר את הגרף ומחכה לרשימה הסופית אחרי התיקונים שלך
# כשממשיכים, הצומת רץ מההתחלה, ולכן לפני ה-interrupt אין כאן שום פעולה
def human_review_node(state: ReviewState):
    final_suggestions = interrupt(state["suggestions"])
    return {"suggestions": final_suggestions}


# בניית גרף האישור: התחלה ← אישור אדם ← סוף
# ה-checkpointer שומר את המצב בזיכרון, וזה מה שמאפשר לעצור ולהמשיך
review_builder = StateGraph(ReviewState)
review_builder.add_node("human_review", human_review_node)
review_builder.add_edge(START, "human_review")
review_builder.add_edge("human_review", END)
review_app = review_builder.compile(checkpointer=InMemorySaver())


# ---------- הפעלה ----------

# מדפיסה את ההצעות ממוינות לפי נושא, ומחזירה את הנתיבים לפי אותו סדר (המספר = המקום ברשימה)
# שם הקובץ מוצג רק לך, והוא לא נשלח למודל
def show_batch(suggestions):
    ordered = sorted(suggestions, key=lambda path: suggestions[path])
    last_topic = None
    for number, path in enumerate(ordered, start=1):
        if suggestions[path] != last_topic:
            last_topic = suggestions[path]
            print(f"\n{last_topic}")
        print(f"  {number}. {os.path.basename(path)}")
    return ordered

def move_image(path,topic):
    target_folder=os.path.join(SCAN_FOLDER,topic)
    os.makedirs(target_folder,exist_ok=True)
    name,extension=os.path.splitext(os.path.basename(path))
    target=os.path.join(target_folder,name+extension)
    number=2
    while os.path.exists(target):
        target = os.path.join(target_folder, f"{name} ({number}){extension}")
        number += 1
    shutil.move(path, target)
if __name__ == "__main__":

    images = list_images(SCAN_FOLDER)
    if MOVE_FILES:
        images = [p for p in images if os.path.dirname(p) == SCAN_FOLDER]
    if IMAGES_TO_RUN is not None:
        images = random.sample(images, min(IMAGES_TO_RUN, len(images)))

    # עוברים על התמונות בקבוצות של BATCH_SIZE
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start:start + BATCH_SIZE]

        # שלב א: מסווגים כל תמונה בקבוצה בנפרד, בלי עצירות
        suggestions = {}
        for path in batch:
            suggestions[path] = classify_app.invoke({"image_path": path})["topic"]

        # שלב ב: גרף האישור רץ עד ה-interrupt ונעצר
        config = {"configurable": {"thread_id": f"batch-{start}"}}
        review_app.invoke({"suggestions": suggestions}, config)

        # שלב ג: מציגים לך את הקבוצה, ואת מתקנת מה שצריך
                # שלב ג: מציגים לך את הקבוצה, ואת מתקנת מה שצריך
        while True:
            ordered = show_batch(suggestions)
            answer = input("\nEnter = approve all, number = fix, v number = view image: ").strip()
            if answer == "":
                break
            # v ומספר: פותחים את התמונה לצפייה (פעולה של Windows בלבד), ושום דבר לא משתנה
            if answer.lower().startswith("v "):
                number = answer[2:].strip()
                if number.isdigit() and 1 <= int(number) <= len(ordered):
                    os.startfile(ordered[int(number) - 1])
                else:
                    print("Not a valid number")
                continue
            if not answer.isdigit() or not 1 <= int(answer) <= len(ordered):
                print("Not a valid number")
                continue
            new_topic = input(f"Correct topic {TOPICS}: ").strip().lower()
            if new_topic in TOPICS:
                suggestions[ordered[int(answer) - 1]] = new_topic
            else:
                print("Not a valid topic")

        # שלב ד: ממשיכים את גרף האישור עם הרשימה הסופית
        final = review_app.invoke(Command(resume=suggestions), config)
        if MOVE_FILES:
            for path, topic in final["suggestions"].items():
                if topic in TOPICS:  # תשובה לא תקינה של המודל לא מועברת
                    move_image(path, topic)
            print("\n--- Final (files were moved) ---")
        else:
            print("\n--- Final (nothing was moved) ---")
        show_batch(final["suggestions"])
