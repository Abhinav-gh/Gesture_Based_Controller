import cv2
import mediapipe as mp
import time
import pyautogui
import collections
import math
import yaml
import os
import numpy as np
import platform
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Deque

# Banner display with style
print("\n" + "="*60)
print("💥 GESTURE SWIPER ENGINE 9000+ 💥".center(60))
print("Enhanced Edition (with Mac Support)".center(60))
print("="*60 + "\n")

@dataclass
class HandState:
    """Tracks the state of a detected hand"""
    position_history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: collections.deque(maxlen=15))
    last_swipe_time: float = 0
    alt_tab_active: bool = False
    gesture_cooldown: float = 0
    last_gesture: str = "unknown"
    swipe_state: str = "none"  # none, start, tracking, confirmed
    swipe_direction: str = "none"
    swipe_start_time: float = 0
    swipe_start_pos: Tuple[float, float] = (0, 0)

class GestureConfig:
    """Configuration parameters for gesture detection"""
    def __init__(self):
        # Default configuration
        self.gesture_delay = 1.5           # Time between consecutive gesture actions
        self.swipe_cooldown = 0.4          # REDUCED: quicker consecutive swipes
        self.min_swipe_velocity = 0.3      # REDUCED: lower velocity threshold
        self.min_swipe_distance = 0.08     # REDUCED: smaller movement needed
        self.max_vertical_ratio = 0.7      # INCREASED: allow more vertical movement
        self.detection_confidence = 0.7    # Hand detection confidence threshold
        self.tracking_confidence = 0.7     # Hand tracking confidence threshold
        self.debug_mode = True             # Show debug info on screen
        self.flick_velocity_threshold = 1.0 # REDUCED: easier to trigger flicks
        self.swipe_intent_threshold = 0.2  # REDUCED: easier to start swipe
        self.swipe_max_duration = 1.0      # INCREASED: more time to complete swipe

        # Platform detection for Mac vs Windows/Linux
        self.is_mac = platform.system() == "Darwin"
        print(f"🖥️ Platform detected: {platform.system()} {'(macOS)' if self.is_mac else ''}")

        self.load_config()
        
    def load_config(self):
        """Load configuration from YAML file if it exists"""
        config_file = "gesture_config.yaml"
        
        # Create default config if it doesn't exist
        if not os.path.exists(config_file):
            self.save_config()
            print(f"📝 Created default configuration file: {config_file}")
            return
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                if config:
                    for key, value in config.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️ Error loading configuration: {e}")
            print("Using default settings instead.")
            
    def save_config(self):
        """Save current configuration to YAML file"""
        config_file = "gesture_config.yaml"
        config_dict = {attr: getattr(self, attr) for attr in dir(self) 
                      if not attr.startswith('_') and not callable(getattr(self, attr))
                      and attr != 'is_mac'}  # Don't save platform info
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        except Exception as e:
            print(f"⚠️ Error saving configuration: {e}")


class GestureEngine:
    """Main gesture detection and handling engine"""
    
    def __init__(self):
        self.config = GestureConfig()
        self.hands_state: Dict[str, HandState] = {}
        self.setup_mediapipe()
        self.setup_camera()
        self.running = True
        
        # GUI elements
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {
            'text_primary': (255, 255, 0),    # Yellow
            'text_secondary': (0, 255, 255),  # Cyan
            'gesture_active': (0, 255, 0),    # Green
            'gesture_inactive': (120, 120, 120),  # Gray
            'vector_left': (255, 0, 255),     # Purple
            'vector_right': (0, 165, 255),    # Orange
            'hand_connections': (0, 255, 0),  # Green
            'swipe_intent': (255, 165, 0)     # Orange
        }
        
        # Detect platform for key mappings (Mac vs Windows/Linux)
        self.is_mac = self.config.is_mac
        
        # Define platform-specific key mappings
        if self.is_mac:
            # Mac modifier keys
            self.tab_switch_keys = {
                "next": ["command", "option", "right"],  # Command+Option+Right for next tab on Mac
                "prev": ["command", "option", "left"]    # Command+Option+Left for previous tab on Mac
            }
            self.alt_key = "option"  # Alt is Option on Mac
            self.copy_key = ["command", "c"]
            self.paste_key = ["command", "v"]
            self.undo_key = ["command", "z"]
            self.redo_key = ["command", "shift", "z"]
            self.save_key = ["command", "s"]
            self.close_tab_key = ["command", "w"]
            self.new_tab_key = ["command", "t"]
        else:
            # Windows/Linux modifier keys
            self.tab_switch_keys = {
                "next": ["ctrl", "tab"],             # Ctrl+Tab for next tab
                "prev": ["ctrl", "shift", "tab"]     # Ctrl+Shift+Tab for prev tab
            }
            self.alt_key = "alt"  # Standard Alt key
            self.copy_key = ["ctrl", "c"]
            self.paste_key = ["ctrl", "v"]
            self.undo_key = ["ctrl", "z"]
            self.redo_key = ["ctrl", "y"]
            self.save_key = ["ctrl", "s"]
            self.close_tab_key = ["ctrl", "w"]
            self.new_tab_key = ["ctrl", "t"]
        
    def setup_mediapipe(self):
        """Initialize MediaPipe components"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Enhanced landmark drawing style
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 220, 255),
            thickness=2,
            circle_radius=2
        )
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=self.config.detection_confidence,
            min_tracking_confidence=self.config.tracking_confidence
        )
        
    def setup_camera(self):
        """Initialize webcam capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Webcam not found. Exiting.")
            exit(1)
            
        # Get camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📹 Camera initialized: {self.frame_width}x{self.frame_height}")
        
    def count_fingers(self, landmarks, hand_label: str) -> List[int]:
        """Count extended fingers based on hand landmarks"""
        tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        pip = [3, 6, 10, 14, 18]   # First joints
        mcp = [2, 5, 9, 13, 17]    # Knuckles
        fingers = []
        
        # Thumb detection (different for left/right hand)
        if hand_label == "Right":
            # For right hand, thumb is extended if it's to the left of the MCP joint
            thumb_open = landmarks.landmark[tips[0]].x < landmarks.landmark[pip[0]].x
        else:
            # For left hand, thumb is extended if it's to the right of the MCP joint
            thumb_open = landmarks.landmark[tips[0]].x > landmarks.landmark[pip[0]].x
            
        fingers.append(1 if thumb_open else 0)
        
        # For the other fingers, they're extended if the tip is above the PIP joint
        # We also check if the tip is actually above the MCP (knuckle) joint for more accuracy
        for i in range(1, 5):
            finger_extended = (landmarks.landmark[tips[i]].y < landmarks.landmark[pip[i]].y and
                              landmarks.landmark[tips[i]].y < landmarks.landmark[mcp[i]].y)
            fingers.append(1 if finger_extended else 0)
            
        return fingers
        
    def update_velocity(self, hand_id: str, cx: float, cy: float, now: float) -> Tuple[float, float, float, float]:
        """Calculate hand movement velocity and direction with improved tracking"""
        hand = self.hands_state[hand_id]
        hand.position_history.append((cx, cy, now))
        
        if len(hand.position_history) < 2:
            return 0, 0, 0, 0
            
        # Use more sophisticated velocity calculation
        # Use weighted recent history for smoother velocity
        try:
            # Use first and last position for overall movement
            x0, y0, t0 = hand.position_history[0]
            x1, y1, t1 = hand.position_history[-1]
            
            dx = x1 - x0
            dy = y1 - y0
            dt = t1 - t0 if t1 - t0 > 0 else 1e-6  # Avoid division by zero
            
            # Base velocity calculation
            vx = dx / dt
            vy = dy / dt
            
            # Calculate distance and angle
            dist = math.sqrt(dx**2 + dy**2)
            angle = math.degrees(math.atan2(vy, vx))
            
            return vx, vy, dist, angle
        except Exception as e:
            print(f"Error in velocity calculation: {e}")
            return 0, 0, 0, 0

    def detect_swipe_intent(self, hand_id: str, fingers: List[int]) -> bool:
        """Detect if the hand is in a position that indicates swipe intent"""
        # Check if hand is in a swipe-friendly position (open palm or pointing)
        # This helps distinguish intentional swipes from incidental movements
        return sum(fingers) >= 1 or fingers == [0, 1, 0, 0, 0]  # Open hand or pointing
        
    def detect_gestures(self):
        """Main gesture detection loop"""
        try:
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                    
                # Mirror image for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Process with MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                now = time.time()
                
                # Only draw visualizations if debug mode is on
                if self.config.debug_mode:
                    # Draw status bar
                    self.draw_status_bar(frame)
                
                # Process hand landmarks if detected
                if results.multi_hand_landmarks and results.multi_handedness:
                    for idx, landmarks in enumerate(results.multi_hand_landmarks):
                        try:
                            # Get hand information
                            hand_info = results.multi_handedness[idx].classification[0]
                            label = hand_info.label
                            confidence = hand_info.score
                            hand_id = f"{label}_{idx}"
                            
                            # Create state tracker for new hands
                            if hand_id not in self.hands_state:
                                self.hands_state[hand_id] = HandState()
                                
                            # Only draw hand landmarks if debug mode is on
                            if self.config.debug_mode:
                                # Draw hand landmarks with custom style
                                self.mp_drawing.draw_landmarks(
                                    frame, 
                                    landmarks, 
                                    self.mp_hands.HAND_CONNECTIONS,
                                    self.drawing_spec,
                                    self.drawing_spec
                                )
                            
                            # Calculate finger positions
                            fingers = self.count_fingers(landmarks, label)
                            gesture = self.identify_gesture(fingers)
                            total_fingers = sum(fingers)
                            
                            # Hand center position
                            cx = landmarks.landmark[9].x  # Middle of palm
                            cy = landmarks.landmark[9].y
                            x_pixel = int(cx * frame.shape[1])
                            y_pixel = int(cy * frame.shape[0])
                            
                            # Only draw hand information if debug mode is on
                            if self.config.debug_mode:
                                self.draw_hand_info(frame, hand_id, label, fingers, gesture, 
                                                  (x_pixel, y_pixel), confidence)
                            
                            # Calculate velocity
                            vx, vy, dist, angle = self.update_velocity(hand_id, cx, cy, now)
                            
                            # Process gestures based on which hand
                            if label == "Left":
                                self.process_left_hand(hand_id, total_fingers, fingers, now)
                            elif label == "Right":
                                self.process_right_hand(hand_id, vx, vy, dist, angle, 
                                                     fingers, (x_pixel, y_pixel), frame, now)
                        except Exception as e:
                            print(f"Error processing hand {idx}: {e}")
                            continue
                            
                # Clean up stale hand states (hands no longer detected)
                self.cleanup_stale_hands(results)
                
                # Only display frame if debug mode is on
                if self.config.debug_mode:
                    cv2.imshow("⚔️ Gesture Swiper Engine 9000+", frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("👋 Quitting by user request")
                        break
                else:
                    # Still need to check for keyboard interrupt in non-debug mode
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("👋 Quitting by user request")
                        break
                    
        except KeyboardInterrupt:
            print("🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()
            
    def identify_gesture(self, fingers: List[int]) -> str:
        """Convert finger state to named gesture"""
        if fingers == [0, 0, 0, 0, 0]:
            return "fist"
        elif fingers == [1, 1, 0, 0, 0]:
            return "peace"
        elif fingers == [1, 1, 1, 0, 0]:
            return "three"
        elif fingers == [1, 1, 1, 1, 0]:
            return "four_closed"
        elif fingers == [0, 1, 1, 1, 0]:
            return "three_no_thumb"
        elif fingers == [0, 1, 1, 1, 1]:
            return "four"
        elif fingers == [1, 1, 1, 1, 1]:
            return "open"
        elif fingers == [0, 1, 0, 0, 0]:
            return "point"
        elif fingers == [1, 0, 0, 0, 1]:
            return "rock"
        else:
            # Convert binary to string (e.g. [0,1,1,0,1] -> "01101")
            return ''.join(map(str, fingers))
            
    def process_left_hand(self, hand_id: str, total_fingers: int, fingers: List[int], now: float):
        """Process gestures for left hand (alt-tab functionality)"""
        try:
            hand = self.hands_state[hand_id]
            
            # Detect alt-tab gesture (three fingers up - thumb+index+middle, or index+middle+ring)
            if fingers == [0, 1, 1, 1, 0] or fingers == [1, 1, 1, 0, 0]:  # Three fingers variation
                # Activate alt-tab if not already active
                if not hand.alt_tab_active:
                    print(f"🧠 [Left] 🤘 Holding ALT...")
                    if not self.config.debug_mode:
                        # Use the correct alt key based on platform
                        pyautogui.keyDown(self.alt_key)
                        pyautogui.press("tab")
                    else:
                        print(f"[DEBUG] Would press: {self.alt_key.upper()}+TAB")
                    hand.alt_tab_active = True
                    hand.gesture_cooldown = now
                    hand.last_gesture = "alt-tab"
                # Cycle forward (TAB) when already in alt-tab mode
                elif now - hand.gesture_cooldown > self.config.gesture_delay:
                    print(f"↪️ [Left] Cycling TAB (Forward)")
                    if not self.config.debug_mode:
                        pyautogui.press("tab")
                    else:
                        print(f"[DEBUG] Would press: TAB (while holding {self.alt_key.upper()})")
                    hand.gesture_cooldown = now
                    
            # Detect alt-shift-tab gesture (four fingers up - add ring finger to go backward)
            elif fingers == [0, 1, 1, 1, 1] or fingers == [1, 1, 1, 1, 0]:  # Four fingers variation
                if hand.alt_tab_active and now - hand.gesture_cooldown > self.config.gesture_delay:
                    print(f"↩️ [Left] Cycling SHIFT+TAB (Backward)")
                    if not self.config.debug_mode:
                        pyautogui.keyDown("shift")
                        pyautogui.press("tab")
                        pyautogui.keyUp("shift")
                    else:
                        print(f"[DEBUG] Would press: SHIFT+TAB (while holding {self.alt_key.upper()})")
                    hand.gesture_cooldown = now
                elif not hand.alt_tab_active:
                    # If Alt wasn't active yet, start alt-tab mode and go backward
                    print(f"🧠 [Left] 🖖 Holding ALT (Backward)...")
                    if not self.config.debug_mode:
                        pyautogui.keyDown(self.alt_key)
                        pyautogui.keyDown("shift")
                        pyautogui.press("tab")
                        pyautogui.keyUp("shift")
                    else:
                        print(f"[DEBUG] Would press: {self.alt_key.upper()}+SHIFT+TAB")
                    hand.alt_tab_active = True
                    hand.gesture_cooldown = now
                    hand.last_gesture = "alt-shift-tab"
            
            # Release alt key when gesture changes to something else
            elif hand.alt_tab_active:
                print(f"🔓 [Left] {self.alt_key.upper()} released")
                if not self.config.debug_mode:
                    pyautogui.keyUp(self.alt_key)
                else:
                    print(f"[DEBUG] Would release: {self.alt_key.upper()} key")
                hand.alt_tab_active = False
        except Exception as e:
            print(f"Error in left hand processing: {e}")
            
    def process_right_hand(self, hand_id: str, vx: float, vy: float, dist: float, 
                      angle: float, fingers: List[int], position: Tuple[int, int], frame, now: float):
        """Process gestures for right hand with improved swipe detection"""
        try:
            hand = self.hands_state[hand_id]
            x_pixel, y_pixel = position
            
            # Always define current_direction at the beginning to avoid reference errors
            current_direction = "right" if vx > 0 else "left"
            
            # Check if hand is in a swipe-ready position
            swipe_intent = self.detect_swipe_intent(hand_id, fingers)
            
            # Only draw velocity vector if debug mode is on
            if self.config.debug_mode:
                # Draw velocity vector for visual feedback
                arrow_color = self.colors['vector_right'] if vx > 0 else self.colors['vector_left']
                vector_scale = 200  # Scale factor for arrow length
                cv2.arrowedLine(
                    frame, 
                    (x_pixel, y_pixel),
                    (x_pixel + int(vx * vector_scale), y_pixel + int(vy * vector_scale)),
                    arrow_color, 
                    2
                )
                
                # Add velocity text and show swipe intent
                cv2.putText(
                    frame,
                    f"Vel: {vx:.2f}, {vy:.2f} | Intent: {'Yes' if swipe_intent else 'No'}",
                    (x_pixel + 10, y_pixel + 10),
                    self.font, 
                    0.5, 
                    self.colors['swipe_intent'] if swipe_intent else arrow_color, 
                    1
                )
                
                # Show swipe state
                cv2.putText(
                    frame,
                    f"State: {hand.swipe_state}",
                    (x_pixel + 10, y_pixel + 30),
                    self.font, 
                    0.5, 
                    self.colors['text_secondary'],
                    1
                )
            
            # SIMPLIFIED SWIPE DETECTION
            
            # State 1: Looking for potential swipe start
            if hand.swipe_state == "none":
                # MODIFIED: easier to trigger with any slight horizontal movement
                if abs(vx) > self.config.swipe_intent_threshold:
                    # We detected initial horizontal movement
                    hand.swipe_state = "start"
                    hand.swipe_start_time = now
                    hand.swipe_start_pos = (x_pixel/frame.shape[1], y_pixel/frame.shape[0])
                    hand.swipe_direction = "right" if vx > 0 else "left"
                    if self.config.debug_mode:
                        print(f"👀 Swipe Intent Detected: {hand.swipe_direction}")
            
            # State 2: Tracking ongoing swipe motion
            elif hand.swipe_state == "start":
                # Fast flick detection - immediately confirm on high velocity
                if abs(vx) > self.config.flick_velocity_threshold:
                    hand.swipe_state = "confirmed"
                    # Update direction based on current movement
                    hand.swipe_direction = current_direction
                    if self.config.debug_mode:
                        print(f"⚡ Fast Flick Detected: {hand.swipe_direction}")
                
                # Check for swipe continuation
                elif abs(vx) > self.config.min_swipe_velocity * 0.7:
                    # Continue tracking the swipe
                    normalized_x = x_pixel/frame.shape[1]
                    normalized_y = y_pixel/frame.shape[0]
                    dx = normalized_x - hand.swipe_start_pos[0]
                    dy = normalized_y - hand.swipe_start_pos[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # If we've moved enough distance, confirm the swipe
                    if distance > self.config.min_swipe_distance:
                        hand.swipe_state = "confirmed"
                        # Update direction based on overall movement for more accuracy
                        hand.swipe_direction = "right" if dx > 0 else "left"
                        if self.config.debug_mode:
                            print(f"✓ Swipe Confirmed: {hand.swipe_direction}")
                
                # If swipe takes too long, reset
                elif now - hand.swipe_start_time > self.config.swipe_max_duration:
                    hand.swipe_state = "none"
                    if self.config.debug_mode:
                        print("⏱️ Swipe timed out")
            
            # State 3: Swipe confirmed, trigger action
            elif hand.swipe_state == "confirmed":
                swipe_ready = now - hand.last_swipe_time > self.config.swipe_cooldown
                
                if swipe_ready:
                    direction = "RIGHT ⏩" if hand.swipe_direction == "right" else "LEFT ⏪"
                    print(f"👉 Swipe Executed: {direction} | Vx: {vx:.2f}, Vy: {vy:.2f}, Angle: {angle:.1f}°")
                    
                    # Only draw swipe animation if debug mode is on
                    if self.config.debug_mode:
                        self.draw_swipe_effect(frame, hand.swipe_direction == "right")
                    
                    # Trigger corresponding action unless in debug mode
                    if not self.config.debug_mode:
                        # Use platform-specific key combinations for tab switching
                        if hand.swipe_direction == "right":  # Right swipe
                            self.press_keys(self.tab_switch_keys["next"])
                        else:  # Left swipe
                            self.press_keys(self.tab_switch_keys["prev"])
                    else:
                        # Show debug message for the key combination
                        if hand.swipe_direction == "right":  # Right swipe
                            keys_text = "+".join([k.upper() for k in self.tab_switch_keys["next"]])
                            print(f"[DEBUG] Would press: {keys_text} (next tab)")
                        else:  # Left swipe
                            keys_text = "+".join([k.upper() for k in self.tab_switch_keys["prev"]])
                            print(f"[DEBUG] Would press: {keys_text} (previous tab)")
                    
                    hand.last_swipe_time = now
                    hand.last_gesture = f"swipe-{hand.swipe_direction}"
                
                # Reset state
                hand.swipe_state = "none"
        except Exception as e:
            print(f"Error in right hand processing: {e}")
            hand.swipe_state = "none"  # Reset state in case of error
            
    def press_keys(self, keys):
        """Press keys with platform-aware handling"""
        try:
            # First press all modifier keys in sequence
            for key in keys:
                pyautogui.keyDown(key)
                
            # Then release them in reverse order
            for key in reversed(keys):
                pyautogui.keyUp(key)
        except Exception as e:
            print(f"Error pressing keys: {e}")
            # Safety: try to release all keys
            for key in keys:
                try:
                    pyautogui.keyUp(key)
                except:
                    pass
                
    def draw_swipe_effect(self, frame, is_right: bool):
        """Draw a visual effect when swipe is detected"""
        try:
            height, width = frame.shape[:2]
            color = self.colors['vector_right'] if is_right else self.colors['vector_left']
            
            # Draw arrow across screen
            arrow_start = (width - 100, height // 2) if is_right else (100, height // 2)
            arrow_end = (100, height // 2) if is_right else (width - 100, height // 2)
            
            cv2.arrowedLine(frame, arrow_start, arrow_end, color, 4)
            
            # Add text
            if self.is_mac:
                text = "NEXT TAB ⌘⌥→" if is_right else "⌘⌥← PREV TAB"
            else:
                text = "NEXT TAB ⏩" if is_right else "⏪ PREV TAB"
                
            text_x = width // 4 if is_right else width // 2
            cv2.putText(frame, text, (text_x, height // 2 - 20), 
                       self.font, 1, color, 2)
        except Exception as e:
            print(f"Error drawing swipe effect: {e}")
                   
    def draw_status_bar(self, frame):
        """Draw status bar with app info"""
        try:
            height, width = frame.shape[:2]
            
            # Draw top bar
            cv2.rectangle(frame, (0, 0), (width, 40), (40, 40, 40), -1)
            cv2.putText(frame, "GESTURE SWIPER ENGINE", (10, 30), 
                       self.font, 0.8, self.colors['text_primary'], 2)
            
            # Draw help text
            platform_text = "macOS" if self.is_mac else "Windows/Linux"
            cv2.putText(frame, f"{platform_text} Mode: Left=App Switching | Right=Tab Navigation", 
                       (width // 2 - 230, 30), self.font, 0.6, 
                       self.colors['text_secondary'], 1)
        except Exception as e:
            print(f"Error drawing status bar: {e}")
                   
    def draw_hand_info(self, frame, hand_id: str, label: str, fingers: List[int], 
                      gesture: str, position: Tuple[int, int], confidence: float):
        """Draw information about detected hand"""
        try:
            x, y = position
            hand = self.hands_state[hand_id]
            
            # Gesture name with visual indicator
            if (gesture == "three" or gesture == "three_no_thumb") and label == "Left":
                alt_name = "Option" if self.is_mac else "Alt"
                gesture_text = f"🤘 {alt_name}-Tab Mode (Forward)"
                color = self.colors['gesture_active']
            elif (gesture == "four" or gesture == "four_closed") and label == "Left":
                alt_name = "Option" if self.is_mac else "Alt"
                gesture_text = f"🖖 {alt_name}-Tab Mode (Backward)"
                color = self.colors['gesture_active']
            else:
                emoji_map = {
                    "fist": "✊",
                    "peace": "✌️",
                    "three": "🤟",
                    "three_no_thumb": "🤟",
                    "four": "🖖",
                    "four_closed": "🖖",
                    "open": "🖐️",
                    "point": "👆",
                    "rock": "🤘",
                    "swipe-left": "👈",
                    "swipe-right": "👉"
                }
                emoji = emoji_map.get(gesture, "👋")
                gesture_text = f"{emoji} {gesture}"
                color = self.colors['gesture_inactive']
            
            # Base position for text
            text_y = 40 + (0 if label == "Left" else 40)
            
            # Hand label with confidence
            cv2.putText(frame, f"{label} ({confidence:.2f})", 
                       (10, text_y), self.font, 0.7, color, 2)
            
            # Gesture name
            cv2.putText(frame, gesture_text, 
                       (10, text_y + 25), self.font, 0.6, color, 1)
            
            # Finger status as dots
            for i, finger in enumerate(fingers):
                dot_color = (0, 255, 0) if finger == 1 else (0, 0, 255)
                dot_x = 10 + i * 20
                dot_y = text_y + 45
                cv2.circle(frame, (dot_x, dot_y), 7, dot_color, -1)
        except Exception as e:
            print(f"Error drawing hand info: {e}")
            
    def cleanup_stale_hands(self, results):
        """Remove hand states that are no longer detected"""
        try:
            if not results.multi_handedness:
                # If no hands detected, release alt key if it was pressed
                for hand_id, hand in list(self.hands_state.items()):
                    if hand.alt_tab_active:
                        print(f"🔓 Lost tracking of {hand_id} - releasing {self.alt_key.upper()}")
                        if not self.config.debug_mode:
                            pyautogui.keyUp(self.alt_key)
                        else:
                            print(f"[DEBUG] Would release: {self.alt_key.upper()} key")
                        hand.alt_tab_active = False
        except Exception as e:
            print(f"Error cleaning up stale hands: {e}")
            # Safety measure - release alt key
            pyautogui.keyUp(self.alt_key)
            
    def cleanup(self):
        """Release resources and ensure modifier keys are released"""
        print("🧹 Cleaning up resources...")
        
        try:
            # Release any held keys
            for hand_id, hand in self.hands_state.items():
                if hand.alt_tab_active:
                    print(f"🔓 Releasing {self.alt_key.upper()} key for {hand_id}")
                    if not self.config.debug_mode:
                        pyautogui.keyUp(self.alt_key)
                    else:
                        print(f"[DEBUG] Would release: {self.alt_key.upper()} key")
                    hand.alt_tab_active = False
                    
            # Make sure all modifier keys are released
            if not self.config.debug_mode:
                # Define platform-specific modifier keys to ensure they're released
                if self.is_mac:
                    for key in ["command", "option", "shift", "control"]:
                        try:
                            pyautogui.keyUp(key)
                        except:
                            pass
                else:
                    for key in ["alt", "ctrl", "shift", "win"]:
                        try:
                            pyautogui.keyUp(key)
                        except:
                            pass
                
            # Release camera and close windows
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                
            cv2.destroyAllWindows()
            
            if hasattr(self, 'hands') and self.hands is not None:
                self.hands.close()
                
            print("💀 Exited gracefully. Peace out.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            # Last resort cleanup
            if not self.config.debug_mode:
                if self.is_mac:
                    for key in ["command", "option", "shift", "control"]:
                        try:
                            pyautogui.keyUp(key)
                        except:
                            pass
                else:
                    for key in ["alt", "ctrl", "shift", "win"]:
                        try:
                            pyautogui.keyUp(key)
                        except:
                            pass


# Main program entry point
if __name__ == "__main__":
    # Check required packages
    try:
        import yaml
    except ImportError:
        print("⚠️ PyYAML not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "pyyaml"])
        import yaml
    
    try:
        import numpy
    except ImportError:
        print("⚠️ NumPy not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "numpy"])
        import numpy
        
    # Run the gesture engine
    try:
        engine = GestureEngine()
        engine.detect_gestures()
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")