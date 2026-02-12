from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str = "AIzaSyA7cyAQWwHSBjo-g8iGQC3WL0ZK2orLZaM"
    mem0_api_key: str = "m0-SXbg18Sm3dbh0vWT9fQwx4iwUoIEQG3cBc79ZGhX"
    zep_api_key: str = "z_1dWlkIjoiMjU4ZWJhMzUtMzMzMy00NjZmLTg3N2YtMWQ1ZGZjNzdkYjg2In0.UDwlmEi8zqd-xBllm0FnZv9QbNxNSBc2QAfsAVrmvIj4dv1x1TBCkjamOTro8RYn-RK6pANwRXNWj9zC9VaxJQ"

    # Mem0 Configuration
    mem0_api_base: str = "https://api.mem0.ai"
    mem0_user_id: str = "memory-test-user"

    # Zep Configuration
    zep_api_base: str = ""
    zep_session_id: str = "memory-test-session"

    # App Configuration
    photos_dir: str = "photos"
    identity_dir: str = "identity"
    app_name: str = "Memory Evaluation Tool"
    frontend_url: str = "https://kun-dehuang.github.io"

    # Gemini Configuration
    gemini_model: str = "gemini-1.5-flash"
    gemini_analysis_prompt: str = f"""
    你是一位专业的视觉人类学家。请详细描述这批照片中的所有视觉要素。

0. **主角识别** (优先级最高):【优先】请根据上面的【主角参考特征】准确识别照片中的主角。
   - **谁是主人**: 通过以下线索综合判断：
     * 通过前置获取到的主角参考特征，判断照片中是否有人物是主角
   - **主角特征**: 详细描述主角的外貌特征（性别、年龄段、发型、体型、常见服装风格）
   - **标记方式**: 在后续描述中，主角用"【主角】"标记，其他人用"女性A"、"男性B"等标记
   - **特殊场景**:
     * 如果是自拍，直接标记"【主角】自拍"
     * 如果是多人大合影，主人通常在中心位置或最显眼的位置
     * 如果主人不在照片中（如主人拍的风景照），说明"【主人】拍摄，未入镜"

**任务**: 请以详尽、客观的方式描述这批照片，包含以下维度：

1. **场景与环境** (完整识别):
   - 室内/室外场景（如：咖啡厅、办公室、公园、家中、餐厅）
   - 城市/地点线索（如：上海外滩、东京街头、居家环境、商场）
   - 装修风格、空间氛围（如：现代简约、复古工业、温馨居家）

2. **人物与互动** (完整识别，重点标注主角):
   - **主角**: 【主角】的性别、年龄段、外貌特征（发型、身材、常见服装风格）
   - **其他人物**: 数量、性别、年龄段，用"女性A"、"男性B"标记
   - **人物关系**: 【主角】与其他人的关系（朋友、情侣、家人、同事）
   - **互动动作**: 【主角】与其他人的具体互动（如：【主角】与女性A举杯庆祝）
   - **表情细节**: 【主角】及其他人物的情绪状态（微笑、严肃、惊讶）

3. **物品与消费** (完整识别):
   - **主角的物品**: 【主角】的服装品牌、风格、配饰、电子设备等
   - **消费场景**: 具体餐厅类型、商场名称、景点特色
   - **物品细节**: 书籍名称、宠物品种、食物种类、品牌Logo

4. **动作与活动** (完整识别，聚焦主角):
   - **主角的活动**: 【主角】在做什么（聚餐、旅行、工作、运动、购物、休闲）
   - **具体动作**: 【主角】的动作细节（行走、坐着、站立、跑步、拍照）
   - **互动细节**: 【主角】与其他人物的互动方式

5. **视觉风格**:
   - 光线（自然光、室内灯光、夜景、逆光）
   - 构图（自拍、抓拍、摆拍、全身照、特写）
   - 修图程度（精修、原图、滤镜风格）

**重要提醒**:
- ⭐ **首要任务**: 准确识别谁是照片主人（主角）
- ⭐ **标记清晰**: 在所有描述中，用"【主角】"标记主人
- ⭐ **一致性**: 确保同一批次的描述中，【主角】指代同一个人
- 请确保识别每一张照片的主体、环境和动作
- 不要遗漏任何可见的重要细节
- 你的描述将成为后续人格分析的基础素材，主角识别准确性至关重要
    """

    # Protagonist Configuration
    protagonist_tag: str = "【主角】"
    protagonist_self_pronouns: list[str] = ["我", "我的", "我也要", "我在"]

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings():
    return Settings()
