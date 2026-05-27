import cv2
import time
import config

def is_overlapping(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return max(0, xB - xA) > 0 and max(0, yB - yA) > 0

class PillDecisionManager:
    def __init__(self, target_meds, completed_meds):
        self.remaining_targets = [m for m in target_meds if m not in completed_meds]
        self.completed_meds = completed_meds
        
        # 狀態機核心控制
        self.step_mode = "STEP_1_SCAN"  
        self.wrong_pills_to_warn = []
        
        # 階段一計數器
        self.step1_frames = 0
        self.step1_required_frames = 10
        self.step1_recorded_pills = set()  
        self.saved_flip_box = None
        
        # 翻完面後的 3 秒倒數計時器
        self.post_flip_start_time = 0
        self.flip_delay_duration = 5.0  # 翻完面後等5秒
        
        # 階段二計數器 (第二面)
        self.step2_frames = 0
        self.step2_required_frames = 10

        # 純刻字通道
        self.instant_pass_frames = 0
        self.instant_required_frames = 3  

    def process_multi_pills(self, frame, all_boxes, model_wrapper, ignore_wrong=False):
        # 用來暫存「這一幀所有辨識成功的藥物 (座標, 標籤)」的對應關係
        pills_with_boxes_this_frame = [] 
        
        self.wrong_pills_to_warn = []
        has_smooth_this_frame = False
        smooth_box_this_frame = None  # 🎯 核心新增：專門死鎖 Smooth 藥物座標的變數，拒絕被刻字藥覆寫！
        last_valid_box = None  # 留給純刻字通道或其他常規流程使用

        # ----------------------------------------------------
        # 1. 基礎特徵掃描
        # ----------------------------------------------------
        for box_tensor in all_boxes:
            box = box_tensor.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, box)
            this_box = [x1, y1, x2, y2] # 每顆藥獨立獨立保存自己的座標
            
            pad = int(max(x2-x1, y2-y1) * 0.15)
            crop = frame[max(0,y1-pad):min(frame.shape[0],y2+pad), max(0,x1-pad):min(frame.shape[1],x2+pad)]
            if crop.size == 0: continue
            
            raw_label, conf, diff, crop_pil = model_wrapper.predict_crop(crop)
            if conf < 0.60 or diff < 0.12: continue

            # 根據不同藥物決定標籤
            final_label = None
            is_pure_smooth_now = False  # 標記當前這顆是不是 Smooth 特徵
            
            if raw_label in ["Letter", "OneLine"] or raw_label in config.INSTANT_PASS_LIST:
                final_label = raw_label
            elif raw_label == "Cross":
                color = model_wrapper.get_pill_color(crop_pil)
                final_label = config.CROSS_WHITE_MED if color == "白色" else config.CROSS_GREEN_MED
            elif raw_label == "Smooth":
                has_smooth_this_frame = True
                is_pure_smooth_now = True  # 確立特徵
                if self.step_mode == "STEP_2_SCAN":
                    final_label = config.DOUBLE_SMOOTH_MED

            # 如果這顆藥有成功認出標籤，把它的「專屬座標」跟「標籤」一起打包存起來
            if final_label:
                pills_with_boxes_this_frame.append((this_box, final_label))
                last_valid_box = this_box
            elif has_smooth_this_frame:
                # 如果只是 Smooth（暫無標籤），也把座標留給翻面流程用
                last_valid_box = this_box
                
            # 🎯 關鍵防禦鎖定：只要當前這顆藥是 Smooth，就把它的座標獨立存下來，絕不允許被迴圈中其他刻字藥洗掉！
            if is_pure_smooth_now:
                smooth_box_this_frame = this_box

        # 為了相容原本的狀態機邏輯，把標籤抽出來做成原本的清單
        detected_this_frame = [label for box, label in pills_with_boxes_this_frame]

        # ----------------------------------------------------
        # 2. 異常藥物攔截（精準定位！）
        # ----------------------------------------------------
        for this_box, label in pills_with_boxes_this_frame:
            if label in config.ALL_19_MEDS and label not in self.remaining_targets and label not in self.completed_meds:
                self.wrong_pills_to_warn.append((this_box, label))
                
        if self.wrong_pills_to_warn and not ignore_wrong:
            first_wrong_box = self.wrong_pills_to_warn[0][0]
            return "WARNING", (first_wrong_box, "請將被框取藥品去除")

        # ----------------------------------------------------
        # 3. 翻面時間差狀態機邏輯
        # ----------------------------------------------------
        # 🔷 【階段 1】初次掃描第一面 (辨識分流控制)
        if self.step_mode == "STEP_1_SCAN":
            # 💡 檢查當前有沒有認出「屬於這輪藥單」且「在快速過關清單」中的純刻字/藥名標籤
            valid_instant_meds = [m for m in detected_this_frame if m in config.INSTANT_PASS_LIST and m in self.remaining_targets]
            
            # 🎯 分流 A：如果是純刻字藥，且沒看到 Smooth 攪局，直接走 3 幀快速通關！
            if valid_instant_meds and not has_smooth_this_frame:
                self.step1_frames = 0 # 清除翻面計數
                self.instant_pass_frames += 1
                
                if self.instant_pass_frames >= self.instant_required_frames:
                    self.instant_pass_frames = 0
                    return "MATCHED", (valid_instant_meds, f"已識別純刻字藥: {valid_instant_meds}！直接通關")
                
                return "SCANNING", f"已辨識刻字藥物... 鎖定中 ({self.instant_pass_frames}/{self.instant_required_frames})"

            # 🎯 分流 B：如果偵測到 Smooth，代表此藥需要看兩面，老實走 10 幀翻面準備！
            elif has_smooth_this_frame:
                self.instant_pass_frames = 0 
                self.step1_frames += 1
                
                for label in detected_this_frame:
                    if label in self.remaining_targets:
                        self.step1_recorded_pills.add(label)
                
                if self.step1_frames >= self.step1_required_frames:
                    # 💡 撥亂反正：滿10幀切換時，100% 拿鎖定好的 Smooth 座標，絕對不拿最後被覆寫的 last_valid_box
                    if smooth_box_this_frame:
                        self.saved_flip_box = smooth_box_this_frame
                    elif last_valid_box:
                        self.saved_flip_box = last_valid_box
                        
                    self.step_mode = "STEP_WAIT_FLIP"   # 轉移至翻面強制定時器
                    self.post_flip_start_time = time.time() # 啟動 5 秒計時！
            
            # 🎯 分流 C：啥都沒看到，倒扣計數器
            else:
                self.step1_frames = max(0, self.step1_frames - 1)
                self.instant_pass_frames = max(0, self.instant_pass_frames - 1)
                
            pills_str = "+".join(list(self.step1_recorded_pills)) if self.step1_recorded_pills else "等待特徵..."
            return "SCANNING", f"已識別特徵: {pills_str} (第一面鎖定進度: {self.step1_frames}/{self.step1_required_frames})"

        # 🔷 【綜合階段 2 & 3】強制倒數 5 秒，期間只要有藥就畫橘框
        elif self.step_mode == "STEP_WAIT_FLIP":
            # 💡 終極安全機制：只有當畫面中真正捕獲到 Smooth 的專屬座標時，才准更新橘框！
            if has_smooth_this_frame and smooth_box_this_frame:
                self.saved_flip_box = smooth_box_this_frame
            
            # 計算經過的時間
            elapsed = time.time() - self.post_flip_start_time
            remaining = int(self.flip_delay_duration - elapsed) + 1
            
            # ⏳ 5秒之內：持續顯示橘框，並提示剩餘秒數
            if elapsed < self.flip_delay_duration:
                return "NEED_FLIP", (self.saved_flip_box, f"第一面鎖定！請將橘框內的藥品翻面 (剩餘 {remaining} 秒)")
            
            # 🚀 無論如何，5秒時間到：直接強制切換到【階段 4】第二面嚴格審查！
            else:
                self.step2_frames = 0
                self.step_mode = "STEP_2_SCAN"
                return "SCANNING", "倒數結束，系統開始強制進行第二面結果辨識..."

        # 🔷 【階段 4】第二面嚴格看完 10 幀
        elif self.step_mode == "STEP_2_SCAN":
            if has_smooth_this_frame:
                self.step2_frames += 1
                
                for label in detected_this_frame:
                    if label in self.remaining_targets:
                        self.step1_recorded_pills.add(label)
                
                if self.step2_frames >= self.step2_required_frames:
                    final_release_pills = list(self.step1_recorded_pills)
                    
                    if config.DOUBLE_SMOOTH_MED in self.remaining_targets:
                        if config.DOUBLE_SMOOTH_MED not in final_release_pills:
                            final_release_pills.append(config.DOUBLE_SMOOTH_MED)
                            
                    if final_release_pills:
                        pills_str = "+".join(final_release_pills)
                        
                        self.step_mode = "STEP_1_SCAN"
                        self.step1_frames = 0
                        self.step2_frames = 0
                        self.instant_pass_frames = 0
                        self.step1_recorded_pills.clear()
                        
                        return "MATCHED", (final_release_pills, f"已識別特徵: {pills_str}！開始服藥動作辨識")
            else:
                self.step2_frames = max(0, self.step2_frames - 1)
                
            return "SCANNING", f"【第二面嚴格核對中】請保持不動 (進度: {self.step2_frames}/{self.step2_required_frames})"

        return "SCANNING", "請拍攝藥物開始偵測"
    
class ActionDecisionManager:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.stage = 1  
        self.stable_frames = 0
        self.required_frames = 5  

    def update(self, detected_objects):
        hands = [obj for obj in detected_objects if obj['label'] == 'hand']
        pills = [obj for obj in detected_objects if obj['label'] == 'pill']
        mouths = [obj for obj in detected_objects if obj['label'] in ['open_mouth', 'closed_mouth']]

        if self.stage == 1:
            has_pill_on_hand = False
            for h in hands:
                for p in pills:
                    if is_overlapping(h['box'], p['box']):
                        has_pill_on_hand = True
                        break
                if has_pill_on_hand: break
                
            if has_pill_on_hand:
                self.stage = 2
                self.stable_frames = 0

        elif self.stage == 2:
            is_covered = False
            if hands and mouths:
                for h in hands:
                    for m in mouths:
                        if is_overlapping(h['box'], m['box']):
                            is_covered = True
                            break
                    if is_covered: break
            
            elif hands and not mouths:
                for h in hands:
                    if h['box'][1] < 240: 
                        is_covered = True
                        break

            if is_covered:
                self.stable_frames += 1
                if self.stable_frames >= self.required_frames:
                    self.stage = 3  
            else:
                self.stable_frames = max(0, self.stable_frames - 1)

        return self.stage