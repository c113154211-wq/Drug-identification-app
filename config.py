# config.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_PILL_PATH = 'best_cut0502.pt'
YOLO_ACT_PATH = 'actbest_0413.pt'
RESNET_PATH = 'best_resnet50_448px_0513_1.pth'

# 1. 所有的藥物原始特徵與藥名標籤
CLASS_NAMES = [
    'Abilify_5mg', 'Clopine_25mg', 'Clozaril_100mg', 'Cross',
    'Dogmatyl_50mg', 'Invega_6mg', 'Invega_9mg', 'Latuda_40mg', 
    'Rexulti_1mg', 'Rexulti_2mg', 'Risperdal_2mg', 'Seroquel_25mg', 
    'Seroquel_300mg', 'Seroquel_50mg', 'Smooth', 'Solian_200mg', 
    'Sunpylon_50mg', 'Surin_200mg'
]

LOGIC_NAMES = ['Sulpiride_200mg', 'Binin-U_5mg', 'Zyprexa_5mg']
INSTANT_PASS_LIST = [c for c in CLASS_NAMES if c not in ['Cross', 'Smooth']]

# 💡 核心新增：根據規格書定義十字顏色分流與雙面光滑藥物名稱
CROSS_WHITE_MED = 'Sulpiride_200mg'  # 十字刻痕 + 白色 判定為此藥
CROSS_GREEN_MED = 'Binin-U_5mg'   # 十字刻痕 + 綠色 判定為此藥
DOUBLE_SMOOTH_MED = 'Zyprexa_5mg'   # 雙面皆光滑 判定為此藥

# 💡 核心新增：完整定義 19 種抗精神藥品的總集清單（用來過濾是否為不屬於本次藥單的抗精神藥）
ALL_19_MEDS = [name for name in CLASS_NAMES if name not in ['Cross', 'Smooth']] + LOGIC_NAMES

# 2. 當前指定要吃的藥物 (範例)
SELF_DEFINED_MEDS = ['Abilify_5mg', 'Risperdal_2mg', 'Zyprexa_5mg']

SMOOTH_NEED_FLIP_MED = 'Zyprexa_5mg'

# ========================================================
# 用迴圈自動產生所有藥品的照片對照表
# ========================================================
MED_IMAGE_PATHS = {}
for pill_name in SELF_DEFINED_MEDS:
    MED_IMAGE_PATHS[pill_name] = f"images/{pill_name}.png"
# ========================================================

# 相機設定
BACK_CAM = 1
FRONT_CAM = 0