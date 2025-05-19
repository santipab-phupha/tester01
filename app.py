import os
import re
import json
import streamlit as st
import pandas as pd
import google.generativeai as genai

# ==== CONFIG ====
API_KEY = "AIzaSyDGvQId4r_P26jFxA_7hVeirouw8KIppUQ"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Use the files you’ve already provisioned in the app directory
Q_FILE = os.path.join(os.getcwd(), "question.txt")    # now a .txt file
DB_FILE = os.path.join(os.getcwd(), "database.csv")

# ==== AI / JSON UTILS ====
def ask_gemini(prompt: str) -> str:
    convo = model.start_chat(history=[])
    return convo.send_message(prompt).text

def extract_json_array(raw: str) -> str:
    matches = re.findall(r"\[.*\]", raw, flags=re.DOTALL)
    return max(matches, key=len) if matches else raw

def load_questions():
    if not os.path.exists(Q_FILE):
        return None, None
    raw = open(Q_FILE, encoding="utf-8").read()
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        cleaned = extract_json_array(raw)
        try:
            questions = json.loads(cleaned)
            # overwrite the existing file with cleaned JSON
            with open(Q_FILE, "r+", encoding="utf-8") as f:
                f.seek(0)
                f.truncate()
                json.dump(questions, f, ensure_ascii=False, indent=2)
            return questions, None
        except json.JSONDecodeError as e:
            return None, e

def recheck_questionnaire_fit(questions_text, user_summary):
    prompt = f"""
โจทย์ภาพรวมของโปรเจกต์: {user_summary}

คำถามที่ใช้: {questions_text}

ช่วยประเมินดังนี้:
1. แต่ละคำถามตรงกับวัตถุประสงค์หรือไม่? ระบุเป็นรายข้อ
2. คำถามเหมาะสมกับลักษณะของกลุ่มเป้าหมายหรือไม่ (เช่น ภาษา, โทน, สมมติฐาน)?
3. หากพบข้อไม่เหมาะสม ให้เสนอคำถามใหม่ที่ควรใช้แทน พร้อมเหตุผลสั้น ๆ
4. ทำเป็นแค่ตารางพอ ไม่ต้องมีข้อสังเกตุเพิ่มเติม ใช้ตารางพอโอเคมั้ย
5. หัวตารางมี: คำถาม/ตรงกับวัตถุประสงค์?/เหมาะสมกับกลุ่มเป้าหมาย?/คำถามใหม่ (หากไม่เหมาะสม)/เหตุผล
"""
    return ask_gemini(prompt)

# ==== STREAMLIT APP ====
st.set_page_config(page_title="Survey App", layout="wide")
if "current_page" not in st.session_state:
    st.session_state.current_page = "Generate Questions"

with st.sidebar:
    st.markdown("## 📊 Survey Navigation")
    pages = ["Generate Questions", "Answer Poll", "Check Bias", "Fix Survey", "Re-Check"]
    choice = st.selectbox("Select Section", pages, index=pages.index(st.session_state.current_page))
    st.session_state.current_page = choice

page = st.session_state.current_page

# ==== PAGE: GENERATE QUESTIONS ====
if page == "Generate Questions":
    st.title("🧠 Generate Survey Questions")

    if not os.path.exists(Q_FILE):
        st.stop()

    user_summary = st.text_area(
        "🗒️ สรุปหัวข้อแบบสอบถาม",
        height=300,
        value=st.session_state.get("user_summary_default", """
        ช่วยออกแบบแบบสอบถามในหัวข้อ:
        - จุดประสงค์
        • อยากรู้ว่าพฤติกรรมเพื่อนำมาประเมินและพัฒนา

        จำนวนคนที่ต้องการ
        • รวม 500 คน
        • แบ่งเป็น 4 กลุ่ม กลุ่มละประมาณ 125 คน

        กลุ่มที่อยากได้ข้อมูล
        • คนที่เข้าฟิตเนส
        • คนไม่เข้าฟิตเนส แต่รักสุขภาพ
        • คนไม่สนใจสุขภาพ
        • คนที่ชอบใช้แก้วเยติ / แก้วเก็บความเย็น

        คำถามหลัก
        • ปกติใช้แก้วน้ำไหม?
        • ถ้ามีแก้ววัดสารอาหาร+แสดงผลในแอป+ถ่ายอาหารแล้วคำนวณแคล จะยอมจ่ายเท่าไหร่?
        • ตอนนี้คุมอาหารยังไง?
        • ทำงานอะไร?
        • ทำไมถึงพกแก้ว? มองแก้วยังไง?
        """)
    )
    st.session_state.user_summary_default = user_summary

    if st.button("📄 Generate / Update Questions"):
        prompt = f"""
จากข้อมูลต่อไปนี้:

{user_summary}

ใช้ Expectancy Theory + Maslow’s Hierarchy of Needs
สร้างคำถาม 10 ข้อ เพื่อสำรวจพฤติกรรมกลุ่มตัวอย่าง
ให้ output เป็น JSON array:
[
  {{
    "id": "Q-1",
    "question": "ตัวอย่างคำถาม",
    "type": "Likert/Checkbox/Text/Choice/Prescreen",
    "options": ["opt1","opt2",...],
    "reason": "..."
  }},
  ...
]
"""
        with st.spinner("🔁 Generating with Gemini..."):
            raw = ask_gemini(prompt)
            cleaned = extract_json_array(raw)
            try:
                qlist = json.loads(cleaned)
                # overwrite the existing question.txt
                with open(Q_FILE, "r+", encoding="utf-8") as f:
                    f.seek(0)
                    f.truncate()
                    json.dump(qlist, f, ensure_ascii=False, indent=2)
                st.success("✅ อัปเดต `question.txt` เรียบร้อยแล้ว!")
                st.experimental_rerun()
            except Exception:
                st.code(raw, language="json")

# ==== PAGE: ANSWER POLL ====
elif page == "Answer Poll":
    st.title("🗳️ Answer Survey Poll")

    if not os.path.exists(DB_FILE):
        st.error("❌ ไม่พบไฟล์ `database.csv` กรุณาอัปโหลดไฟล์นี้ไปยังไดเรกทอรีของแอปก่อน Deploy")
        st.stop()

    questions, load_error = load_questions()
    if load_error:
        st.error(f"❌ โหลด `question.txt` ผิดพลาด: {load_error}")
        st.stop()
    if questions is None:
        st.warning("ยังไม่มีคำถามใน `question.txt`")
        st.stop()

    # show existing responses
    df_db = pd.read_csv(DB_FILE)
    if not df_db.empty:
        st.download_button(
            "📥 Export Responses (.csv)",
            df_db.to_csv(index=False, encoding="utf-8-sig"),
            file_name="database.csv",
            mime="text/csv"
        )
        st.subheader("📊 สรุปคำตอบ")
        for q in questions:
            qid, text, qtype = q["id"], q["question"], q["type"].lower()
            if qid in df_db.columns:
                if qtype == "checkbox":
                    exploded = df_db[qid].dropna().astype(str).str.split(";").explode()
                    counts = exploded.value_counts()
                else:
                    counts = df_db[qid].value_counts()
                st.markdown(f"**{qid}. {text}**")
                st.bar_chart(counts)

    st.subheader("✍️ กรุณาตอบแบบสอบถาม")
    with st.form("survey_form"):
        answers = {}
        for q in questions:
            qid, text, qtype = q["id"], q["question"], q["type"].lower()
            opts = q.get("options", [])
            label = f"{qid}. {text}"
            if qtype == "checkbox":
                answers[qid] = st.multiselect(label, opts, key=qid)
            elif qtype in ("likert", "choice", "prescreen"):
                answers[qid] = st.radio(label, opts, key=qid)
            else:
                answers[qid] = st.text_input(label, key=qid)
        submitted = st.form_submit_button("🚀 Submit")

    if submitted:
        df_db = pd.read_csv(DB_FILE)
        row = {
            q["id"]: (";".join(answers[q["id"]]) if isinstance(answers[q["id"]], list) else answers[q["id"]])
            for q in questions
        }
        df_db = pd.concat([df_db, pd.DataFrame([row])], ignore_index=True)
        # overwrite the existing CSV without creating a new file
        with open(DB_FILE, "r+", encoding="utf-8") as f:
            f.seek(0)
            f.truncate()
            df_db.to_csv(f, index=False, encoding="utf-8-sig")
        st.success("✅ บันทึกคำตอบเรียบร้อยแล้ว")
        st.experimental_rerun()

# ==== PAGE: CHECK BIAS ====
elif page == "Check Bias":
    st.title("🧐 ตรวจสอบ Bias ในคำถามแบบสอบถาม")
    st.markdown("กรอกคำถามหรืออัปโหลด `.txt` เพื่อวิเคราะห์อคติ")
    uploaded = st.file_uploader("Upload .txt", type=["txt"])
    default = """1. คุณคิดว่าคนที่ไม่ออกกำลังกายขี้เกียจหรือไม่?
2. คุณพอใจกับรูปร่างของคุณแค่ไหน?
3. ถ้าคุณรักสุขภาพจริง คุณจะใช้แก้วน้ำอัจฉริยะหรือไม่?
4. รายได้ของคุณต่ำกว่า 15,000 บาท ใช่หรือไม่?"""
    questions_text = (
        uploaded.read().decode("utf-8") if uploaded else st.text_area("หรือกรอกเอง", height=200, value=default)
    )
    if st.button("🔍 ตรวจสอบ Bias"):
        if "user_summary_default" not in st.session_state:
            st.warning("⚠️ กรุณาสร้างคำถามก่อนในหน้า Generate Questions")
            st.stop()
        prompt = f"""โจทย์วิจัยเดิม: {st.session_state.user_summary_default.strip()}

โปรดตรวจสอบ Bias ในคำถามต่อไปนี้ และสร้างคำถามใหม่แทนที่:
{questions_text.strip()}
"""
        with st.spinner("🤖 กำลังวิเคราะห์..."):
            result = ask_gemini(prompt)
        st.markdown("### ผลลัพธ์")
        st.write(result)

# ==== PAGE: FIX SURVEY ====
elif page == "Fix Survey":
    st.title("🛠️ ปรับปรุงแบบสอบถาม")
    st.markdown("อัปโหลด `.txt` หรือกรอกเพื่อให้ AI ปรับปรุง")
    uploaded = st.file_uploader("Upload .txt", type=["txt"], key="fix_upload")
    default = """1. คุณคิดว่าคนที่ไม่ออกกำลังกายขี้เกียจหรือไม่?
2. คุณพอใจกับรูปร่างของคุณแค่ไหน?
3. ถ้าคุณรักสุขภาพจริง คุณจะใช้แก้วน้ำอัจฉริยะหรือไม่?
4. รายได้ของคุณต่ำกว่า 15,000 บาท ใช่หรือไม่?"""
    questions_text = (
        uploaded.read().decode("utf-8") if uploaded else st.text_area("แบบสอบถามเดิม", height=200, value=default, key="fix_text")
    )
    if st.button("🧠 ปรับปรุงแบบสอบถาม"):
        if "user_summary_default" not in st.session_state:
            st.warning("⚠️ กรุณาสร้างคำถามก่อนในหน้า Generate Questions")
            st.stop()
        prompt = f"""โจทย์วิจัยเดิม: {st.session_state.user_summary_default.strip()}

ปรับปรุงคำถามต่อไปนี้:
{questions_text.strip()}

ให้ผลลัพธ์เป็น JSON array ดังนี้:
[
  {{
    "id": "Q-1",
    "question": "...",
    "type": "...",
    "options": [...],
    "reason": "..."
  }},
  ...
]
"""
        with st.spinner("🤖 กำลังปรับปรุง..."):
            result = ask_gemini(prompt)
        from re import findall
        try:
            json_text = max(findall(r"\[.*\]", result, flags=re.DOTALL), key=len)
            fixed = json.loads(json_text)
            with open("fig_json.json", "r+", encoding="utf-8") as f:
                f.seek(0)
                f.truncate()
                json.dump(fixed, f, ensure_ascii=False, indent=2)
            st.success("✅ บันทึก `fig_json.json` เรียบร้อยแล้ว")
            st.code(json.dumps(fixed, ensure_ascii=False, indent=2), language="json")
        except Exception as e:
            st.error("❌ ไม่สามารถแปลง JSON ได้")
            st.code(result)
            st.exception(e)

# ==== PAGE: RE-CHECK ====
elif page == "Re-Check":
    st.title("🔍 Re-Check ความเหมาะสมของแบบสอบถาม")
    if not os.path.exists(Q_FILE):
        st.error("❌ ไม่พบ `question.txt` กรุณาอัปโหลดก่อน")
        st.stop()
    try:
        question_data = json.load(open(Q_FILE, encoding="utf-8"))
    except Exception as e:
        st.error("❌ อ่าน `question.txt` ผิดพลาด")
        st.exception(e)
        st.stop()

    default_text = "\n".join([f"{q['id']}. {q['question']}" for q in question_data])
    edited = st.text_area("แก้ไขคำถามก่อนประเมิน", height=300, value=default_text)
    if st.button("🔎 ประเมินความเหมาะสม"):
        if "user_summary_default" not in st.session_state:
            st.warning("⚠️ กรุณาสร้างคำถามก่อนในหน้า Generate Questions")
            st.stop()
        with st.spinner("🔍 กำลังวิเคราะห์..."):
            result = recheck_questionnaire_fit(edited, st.session_state.user_summary_default.strip())
        st.markdown("### ผลการประเมิน")
        st.write(result)
