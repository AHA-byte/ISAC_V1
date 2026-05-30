import cv2
import mss
import numpy as np
import pydirectinput
import time
import keyboard
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'runs/detect/isac_run_1-7/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.50

# Screen dimensions (Native resolution of Surface Laptop Studio 2)
SCREEN_WIDTH = 2400
SCREEN_HEIGHT = 1600
CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2

# Initialize the YOLO Brain
model = YOLO(MODEL_PATH)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_screen_frame(sct, monitor):
    """Captures the screen instantly using mss."""
    screenshot = sct.grab(monitor)
    return np.array(screenshot)[:, :, :3] 

def aim_at(target_box):
    """Calculates distance to target and moves the mouse."""
    x1, y1, x2, y2 = target_box
    target_center_x = int((x1 + x2) / 2)
    target_center_y = int((y1 + y2) / 2)
    
    delta_x = target_center_x - CENTER_X
    delta_y = target_center_y - CENTER_Y
    
    # Sensitivity Multiplier
    pydirectinput.move(int(delta_x * 0.3), int(delta_y * 0.3))
    
    return target_center_x, target_center_y

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    print("[SYSTEM] ISAC Bot v2 initializing...")
    print("[SYSTEM] Press and hold 'Q' to trigger the emergency kill switch.")
    
    # Setup fast screen capture zone (primary monitor)
    sct = mss.mss()
    monitor = sct.monitors[1] 

    # --- MEMORY & TIMERS ---
    last_rope_time = 0
    ROPE_COOLDOWN = 10 
    
    last_ability_1 = time.time()
    last_ability_2 = time.time()
    last_ability_3 = time.time()
    
    last_search_vault = 0 
    SEARCH_VAULT_COOLDOWN = 3.0 

    # --- STUCK DETECTION MEMORY ---
    stuck_counter = 0
    last_target_x = 0
    last_target_y = 0

    while True:
        # EMERGENCY KILL SWITCH
        if keyboard.is_pressed('q'):
            print("[SYSTEM] Kill switch activated. Shutting down.")
            break

        current_time = time.time()

        # --- PASSIVE ABILITY MANAGEMENT ---
        if current_time - last_ability_3 > 10:
            pydirectinput.press('3')
            last_ability_3 = current_time
        if current_time - last_ability_2 > 30:
            pydirectinput.press('2')
            last_ability_2 = current_time
        if current_time - last_ability_1 > 40:
            pydirectinput.press('1')
            last_ability_1 = current_time

        # --- 1. PERCEPTION ---
        frame = get_screen_frame(sct, monitor)
        results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        current_view = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            current_view.setdefault(class_name, []).append(box.xyxy[0].cpu().numpy())

        # --- 2. DECISION LOGIC (STATE MACHINE) ---
        
        # Priority 1: Combat / Survival
        if 'enemy_aware' in current_view or 'enemy_unaware' in current_view:
            enemy_target = current_view.get('enemy_aware', current_view.get('enemy_unaware'))[0]
            aim_at(enemy_target)
            pydirectinput.mouseDown()
            time.sleep(0.1) # Burst fire
            pydirectinput.mouseUp()
            stuck_counter = 0 # Reset stuck counter while fighting

        # Priority 2: Interaction & Special Traversal
        elif 'interact_prompt' in current_view:
            if current_time - last_rope_time > ROPE_COOLDOWN:
                pydirectinput.press('f')
                time.sleep(1.5) # Wait for animation
                
                # Check what kind of rope we just grabbed
                if 'rope_direction_down' in current_view:
                    print("[STATE] Rope DOWN detected. Sliding...")
                    pydirectinput.keyDown('s')
                    time.sleep(5) 
                    pydirectinput.keyUp('s')
                    last_rope_time = time.time() 
                    
                elif 'rope_direction_up' in current_view:
                    print("[STATE] Rope UP detected. Climbing...")
                    pydirectinput.keyDown('w')
                    time.sleep(5) 
                    pydirectinput.keyUp('w')
                    last_rope_time = time.time() 
                else:
                    print("[STATE] Standard interaction. Waiting...")
                    time.sleep(1)
            else:
                pass # Rope is on cooldown, ignore prompt

        # --- Priority 3: Progression & Smart Vaulting ---
        elif 'waypoint_trail' in current_view or 'waypoint_door' in current_view or 'waypoint' in current_view:
            
            # STRICT PRIORITY: Trail > Door > Standard Waypoint
            if 'waypoint_trail' in current_view:
                target = current_view['waypoint_trail'][0]
            elif 'waypoint_door' in current_view:
                target = current_view['waypoint_door'][0]
            else:
                target = current_view['waypoint'][0]
                
            curr_x, curr_y = aim_at(target)
            
            pydirectinput.keyDown('w')
            time.sleep(0.5) 
            pydirectinput.keyUp('w')

            # STUCK LOGIC: Check if the waypoint coordinates moved
            if abs(curr_x - last_target_x) < 10 and abs(curr_y - last_target_y) < 10:
                stuck_counter += 1
            else:
                stuck_counter = 0 

            last_target_x = curr_x
            last_target_y = curr_y

            # SMART VAULT: Try hopping the cover
            if stuck_counter == 2:
                print("[STATE] Obstacle detected. Attempting to vault...")
                pydirectinput.keyDown('w') 
                pydirectinput.press('ctrl')
                time.sleep(1.0) 
                pydirectinput.keyUp('w')

            # EVASION MANEUVER: The wall is too tall, break line of sight
            elif stuck_counter > 5:
                print("[STATE] Vault failed. Executing evasion maneuver.")
                pydirectinput.move(-400, 0) # Snap look left
                pydirectinput.keyDown('w')
                time.sleep(1.5)
                pydirectinput.keyUp('w')
                stuck_counter = 0

        # --- Priority 4: Scanning & Search Vaulting ---
        else:
            # 1. SLOW rotation to prevent motion blur (reduced from 100 to 40)
            pydirectinput.move(40, 0) 
            pydirectinput.keyDown('w')
            
            # 2. Incremental vaulting check
            if current_time - last_search_vault > SEARCH_VAULT_COOLDOWN:
                pydirectinput.press('ctrl')
                last_search_vault = current_time
                
            # 3. Slightly longer sleep to let the camera settle for the next YOLO frame
            time.sleep(0.15) 
            pydirectinput.keyUp('w')

if __name__ == '__main__':
    main()