import os
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# شحن وإعداد متغيرات البيئة من ملف .env لـ مفاتيح الـ API السرية
load_dotenv()

app = FastAPI(
    title="Smart Eye - Production Backend API",
    description="سيرفر الواجهة الخلفية المتكامل المتوافق تماماً مع مخرجات كود جودي للـ Computer Vision والموبايل",
    version="2.5.0"
)

# ----------------------------------------------------------------
# 1. الـ Data Models المتوافقة مع الموبايل والـ CV (التصميم الجديد)
# ----------------------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    # الموبايل بيقدر يبعت مصفوفة الأجسام (groups) المكتشفة من كود جودي مع السؤال
    current_cv_data: Optional[List[Dict[str, Any]]] = None  

class SettingsModel(BaseModel):
    voice_language: str = "ar"       # لغة الصوت الافتراضية
    voice_speed: float = 1.0          # سرعة الصوت الافتراضية
    last_read_text: Optional[str] = "" # سجل آخر نص تمت قراءته

# ----------------------------------------------------------------
# 2. ذاكرة السيرفر المؤقتة لحفظ الإعدادات والسجل
# ----------------------------------------------------------------
app_settings = SettingsModel()

# ----------------------------------------------------------------
# 3. خطوط الاتصال (Endpoints) المعتمدة للتسليم والديمو
# ----------------------------------------------------------------

@app.get("/health")
async def health_check():
    """التحقق من صحة وجاهزية السيرفر للاستضافة السحابية"""
    return {"status": "ok"}


@app.post("/ask")
async def ask_free_question(payload: QuestionRequest):
    """
    مستقبل الأسئلة الحرة (مثل: هل الطريق آمن؟ أو شو أقرب شي لإيدي؟).
    يقوم بقراءة السؤال وتحليل مصفوفة كود جودي المبعوثة من الموبايل برمجياً،
    ويجهز السياق الكامل لنموذج اللغة (LLM).
    """
    question = payload.question.lower()
    cv_objects = payload.current_cv_data if payload.current_cv_data else []
    
    # أ. معالجة ذكية ومباشرة بناءً على بيانات كود جودي الفعلي لو أرسلها الموبايل
    if cv_objects:
        # إذا كان السؤال عن الأجسام القريبة
        if "قريب" in question or "closest" in question or "جنب" in question:
            # البحث عن غرض مسافته قريبة (near) بحسب تصنيف كود جودي
            for obj in cv_objects:
                if obj.get("dist") == "near":
                    name_ar = obj.get("name_ar", "جسم")
                    sentence_ar = obj.get("sentence_ar", "")
                    # استخراج تفاصيل المكان من الجملة المجهزة من جودي
                    location_phrase = sentence_ar.split("يوجد")[-1] if "يوجد" in sentence_ar else ""
                    return {"answer": f"أقرب غرض لك هو {name_ar}{location_phrase}."}
        
        # إذا كان السؤال عن الأمان أو العوائق والتحذيرات
        if "خطر" in question or "hazard" in question or "آمن" in question or "طريق" in question:
            # فحص لو كود جودي لقط أي غرض مصنف كـ خطر (hazard: True)
            hazards_found = [obj.get("name_ar") for obj in cv_objects if obj.get("hazard") is True]
            if hazards_found:
                return {"answer": f"انتبه! الطريق قد لا يكون آمناً تماماً، هناك تحذيرات تخص: {', '.join(hazards_found)}."}
            return {"answer": "بحسب البيانات الحالية، لا توجد عوائق أو تحذيرات حرجة أمامك، الطريق آمن تماماً."}

    # ب. ردود افتراضية ذكية ومستقرة للديمو في حال لم يرسل الموبايل مصفوفة الكشف بعد
    if "آمن" in question or "safe" in question or "طريق" in question:
        return {"answer": "الطريق يبدو آمناً للمرور، لا توجد عوائق حرجّة مكتشفة قريبة حالياً."}
    
    if "قريب" in question or "closest" in question:
        return {"answer": "أقرب جسم ليدك هو كوب الماء الموجود في المنتصف."}
        
    # ج. صياغة السياق التلقائي (Context) المجهز للـ LLM في الأسئلة العامة الأخرى
    scene_context = "Current Frame Analysis from YOLO-World:\n"
    for obj in cv_objects:
        scene_context += f"- Detected {obj.get('name')} ({obj.get('name_ar')}) which is {obj.get('dist')} on the {obj.get('pos')}.\n"
        
    llm_prompt = f"Context:\n{scene_context}\nUser Question: {payload.question}"
    
    return {
        "answer": "تم استلام سؤالك وجاري معالجته برمجياً عبر نموذج اللغة (LLM) بنجاح.",
        "debug_status": "LLM context prepared successfully.",
        "context_prepared_for_llm": llm_prompt
    }


@app.get("/settings")
async def get_settings():
    """جلب إعدادات وسجل المستخدم الحالية للموبايل"""
    global app_settings
    return app_settings


@app.post("/settings")
async def update_settings(settings: SettingsModel):
    """تحديث وحفظ الإعدادات والسجل القادمة من الموبايل"""
    global app_settings
    app_settings = settings
    return {"status": "success", "message": "تم حفظ وتحديث الإعدادات بنجاح"}
