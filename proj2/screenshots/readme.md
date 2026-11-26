# סוכן מיון צילומי מסך (LangGraph + LangSmith)

## מה הפרויקט עושה
סוכן שסורק תיקייה של צילומי מסך ומציע לכל תמונה נושא: `meeting`, `code`, `site_not_loading`, `site_guide`, `shop_item` או `other`.
הסוכן רק מציע. הוא לא מעביר קבצים. את הצעות הסיווג מאשרים או מתקנים בעצמך, בקבוצות של 10 תמונות.
המודל (`gpt-4.1`) מקבל רק את התמונה, אף פעם לא את שם הקובץ או התיקייה.

## איך זה בנוי
שני גרפים ב-`agent.py`:
1. **גרף סיווג** (לכל תמונה): `scan` ← `classify` ← (אם המודל ענה `unsure`: `read_text`, קריאת הטקסט בתמונה וסיווג מחדש) ← סוף. זה הענף המותנה.
2. **גרף אישור**: עוצר עם `interrupt` ומחכה לאישור או תיקון שלך, וממשיך עם `Command(resume=...)`.

סכמה: `screenshots/graph_flow.png`

## איך מריצים
- `python agent.py` סורק את התיקייה שבשורת `SCAN_FOLDER` (בראש הקובץ) ומציג הצעות. Enter מאשר הכול, ומספר מאפשר תיקון.
- `python eval.py` בונה דאטהסט ב-LangSmith מתיקיית `dataset` (שם התיקייה = הנושא הנכון) ומריץ ניסוי. מגדירים את `RUN_EXPERIMENT` (1 או 2) בראש הקובץ.
- מפתחות ב-`.env`: `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=proj2`.

## ניסויים (36 תמונות, evaluator אחד: `exact_match`)
| ניסוי | מה שונה | ציון |
|---|---|---|
| 1 | פרומפט פשוט, בלי ענף | 0.81 |
| 2 | הגדרה לכל נושא + ענף `unsure` | 0.81 |

הדאטהסט לא שונה בין הניסויים. צילומים: `screenshots/experiments_table.png`, `screenshots/compare_exp1_vs_exp2.png`, `screenshots/experiments_changes.png`.

## צילומי trace
`screenshots/trace_list.png`, `trace_slow_classify.png`, `trace_detail_other.png`, `trace_summary.png`.

## תובנה מה-trace
הממוצע זהה בשני הניסויים (0.81), אבל התוצאות לא זהות: ניסוי 2 תיקן 3 תמונות ושבר 2, כלומר צילומי VS Code (תיקיית `code`) שנקראו `site_guide`. ב-trace של תמונה מתיקיית `code` המודל ענה `other` בביטחון, ולכן הענף של `read_text` לא רץ, כי הוא מופעל רק כשהמודל עונה `unsure`. נשארת גם תמונה אחת מעורפלת באמת: דף התחברות עם שגיאת חיבור.
