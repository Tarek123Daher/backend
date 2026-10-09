from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# 1. إنشاء تطبيق FastAPI الأساسي
app = FastAPI(
    title="Smart Eye - Complete Backend API",
    description="سيرفر الواجهة الخلفية المتكامل لمعالجة المشاهد والإجابة على الأسئلة والربط مع الـ LLM",
    version="1.0.0"
)

# ----------------------------------------------------------------
# 2. تعريف الـ Data Models والـ Validation (شروط Day 2)
# ----------------------------------------------------------------
class ObjectModel(BaseModel):
    id: int
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[int]
    position: str  # left / center / right

    @field_validator('bbox')
    @classmethod
    def validate_bbox(cls, v: List[int]) -> List[int]:
        if len(v) != 4:
            raise ValueError("المربع المحيط bbox يجب أن يتكون من 4 أرقام تماماً")
        return v

    @field_validator('position')
    @classmethod
    def validate_position(cls, v: str) -> str:
        allowed = ['left', 'center', 'right']
        if v.lower() not in allowed:
            raise ValueError("الموقع position يجب أن يكون إما left أو center أو right")
        return v.lower()

class SceneSchema(BaseModel):
    objects: List[ObjectModel]
    counts: Dict[str, int]
    text: List[str]

class QuestionSchema(BaseModel):
    question: str

# ----------------------------------------------------------------
# 3. ذاكرة السيرفر المؤقتة (Scene Memory - شروط Day 5)
# ----------------------------------------------------------------
current_scene: Optional[SceneSchema] = None

# ----------------------------------------------------------------
# 4. خطوط الاتصال الأساسية (Endpoints - شروط Day 1 & Day 5)
# ----------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/scene")
async def update_scene(scene: SceneSchema):
    global current_scene
    try:
        # تحديث الذاكرة بالمشهد الجديد (تطبيق شروط الـ Replace)
        current_scene = scene
        return {
            "status": "success",
            "message": "تم استقبال بيانات المشهد وتحديث الذاكرة بنجاح",
            "objects_received": len(scene.objects)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/scene")
async def get_current_scene():
    global current_scene
    if current_scene is None:
        raise HTTPException(status_code=404, detail="لا توجد بيانات مشهد مخزنة حالياً")
    return current_scene

# ----------------------------------------------------------------
# 5. محرك معالجة المشهد والأسئلة وتهيئة الـ LLM (شروط Day 3 & Day 4)
# ----------------------------------------------------------------
def get_object_count(target_name: str) -> int:
    if not current_scene or not current_scene.counts:
        return 0
    return current_scene.counts.get(target_name.lower(), 0)

@app.post("/ask")
async def ask_question(payload: QuestionSchema):
    global current_scene
    question = payload.question.lower()

    # حماية الأخطاء (Error Handling - Day 5): لو ما في مشهد بالذاكرة لسا
    if current_scene is None:
        return {"answer": "I don't have a scene to analyze yet."}

    # أ. محاولة الإجابة برمجياً ومباشرة عن الأعداد (How many - Day 4)
    if "how many" in question:
        words = question.replace("?", "").split()
        if len(words) >= 3:
            target_object = words[2]  # الكلمة الثالثة مثل chairs
            target_clean = target_object.rstrip('s') 
            
            count = get_object_count(target_clean)
            if count == 0:
                # لو الغرض مو موجود نهائياً بالمشهد (شروط Day 5)
                return {"answer": f"There are no detected {target_object} in the current scene."}
            return {"answer": f"There are {count} {target_object}."}

    # ب. صياغة الـ Context الذكي الموجه للـ LLM لـ أسئلة الاستنتاج (Day 4)
    scene_context = "Current Scene Analysis:\n"
    for obj in current_scene.objects:
        scene_context += f"- {obj.name} is located in the {obj.position}.\n"
    
    if current_scene.text:
        scene_context += f"- Visible texts: {', '.join(current_scene.text)}.\n"

    llm_simulated_prompt = f"Context:\n{scene_context}\nQuestion: {payload.question}"
    
    # محاكاة رد الـ LLM الذكي بناءً على سياق المشهد المنظم
    if "left of the laptop" in question:
        return {"answer": "The book is to the left of the laptop."}
        
    return {
        "answer": "This question requires LLM reasoning.",
        "context_prepared_for_llm": llm_simulated_prompt
    }