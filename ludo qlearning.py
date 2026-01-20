import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import random
import time
import threading
import json
import os

# --- Constants ---
CELL_SIZE = 40
BOARD_SIZE = 15
PANEL_WIDTH = 340
WINDOW_WIDTH = (CELL_SIZE * BOARD_SIZE) + PANEL_WIDTH
WINDOW_HEIGHT = CELL_SIZE * BOARD_SIZE

COLORS = {
    'R': "#E74C3C", 'G': "#2ECC71", 'Y': "#F1C40F", 'B': "#3498DB",
    'grid': "#BDC3C7", 'bg': "#ECF0F1", 'panel_bg': "#ffffff"
}

# --- Shared Brain (The Q-Table) ---
# We use one Q-table for all players to speed up learning.
# State key: "sorted_piece_positions|roll"
class QBrain:
    def __init__(self):
        self.q_table = {} 
        self.alpha = 0.5    # Learning Rate
        self.gamma = 0.9    # Discount Factor
        self.epsilon = 0.2  # Exploration Rate (Training Mode)

    def get_state_key(self, pieces, roll):
        # We sort pieces so [0,5] is treated same as [5,0] (reduces state space)
        # Note: We must track which original index corresponds to which value for actions
        # But for simple lookup, we just use string representation.
        # To keep it simple and robust for this demo, we won't sort, 
        # so specific pieces retain identity (Piece 1 vs Piece 2).
        p_str = ",".join(map(str, pieces))
        return f"{p_str}|{roll}"

    def choose_action(self, pieces, roll, valid_moves, training=True):
        if not valid_moves: return None
        
        state = self.get_state_key(pieces, roll)
        
        # If Training: Explore sometimes
        # If Playing (Not training): Always Exploit (epsilon = 0)
        eff_epsilon = self.epsilon if training else 0.05
        
        if random.random() < eff_epsilon:
            return random.choice(valid_moves)
        
        # Exploitation (Use learned values)
        if state not in self.q_table:
            return random.choice(valid_moves) # No knowledge yet
        
        # Find move with highest Q-value
        best_action = valid_moves[0]
        max_q = -float('inf')
        
        for move in valid_moves:
            # move is the index of the piece (0-3)
            # Since q_table stores keys as strings, we check retrieval
            q_val = self.q_table[state].get(str(move), 0.0) # Json keys are strings
            if q_val > max_q:
                max_q = q_val
                best_action = move
        return best_action

    def learn(self, state, action, reward):
        # Update Q-Value
        if state not in self.q_table: self.q_table[state] = {}
        
        action_key = str(action) # specific piece index
        old_q = self.q_table[state].get(action_key, 0.0)
        
        # Simplified Bellman: NewQ = OldQ + Alpha * (Reward - OldQ)
        # We omit Gamma*MaxFutureQ here for synchronous simplicity 
        # (as we don't know the next roll yet).
        new_q = old_q + self.alpha * (reward - old_q)
        
        self.q_table[state][action_key] = new_q

    def save_brain(self, filename):
        try:
            with open(filename, 'w') as f:
                json.dump(self.q_table, f)
            return True
        except Exception as e:
            print(e)
            return False

    def load_brain(self, filename):
        try:
            with open(filename, 'r') as f:
                self.q_table = json.load(f)
            return True
        except Exception as e:
            print(e)
            return False

# --- Game Logic ---
class LudoLogic:
    def __init__(self, brain):
        self.brain = brain # Shared Brain
        self.players = [{'id': i, 'pieces': [-1]*4, 'score': 0} for i in range(4)]
        self.turn = 0 
        self.global_path = self._generate_path()
        self.last_roll = 0
        self.waiting_for_move = False 
        self.game_over = False
        self.winner = None

    def _generate_path(self):
        path = []
        for c in range(1, 6): path.append((c, 6))
        for r in range(5, -1, -1): path.append((6, r))
        path.append((7, 0))
        for r in range(6): path.append((8, r))
        for c in range(9, 15): path.append((c, 6))
        path.append((14, 7))
        for c in range(14, 8, -1): path.append((c, 8))
        for r in range(9, 15): path.append((8, r))
        path.append((7, 14))
        for r in range(14, 8, -1): path.append((6, r))
        for c in range(5, -1, -1): path.append((c, 8))
        path.append((0, 7))
        for c in range(6): path.append((c, 7)) 
        return path

    def get_piece_coords(self, player_idx, piece_idx):
        pos = self.players[player_idx]['pieces'][piece_idx]
        if pos == -1:
            base = [(0,0), (9,0), (9,9), (0,9)][player_idx]
            off = [(1.5,1.5), (3.5,1.5), (1.5,3.5), (3.5,3.5)][piece_idx]
            return (base[0] + off[0], base[1] + off[1])
        if pos < 51: return self.global_path[(player_idx * 13 + pos) % 52]
        elif pos < 57: 
            d = pos - 51
            if player_idx == 0: return (1 + d, 7)
            if player_idx == 1: return (7, 1 + d)
            if player_idx == 2: return (13 - d, 7)
            if player_idx == 3: return (7, 13 - d)
        return (7, 7)

    def get_valid_moves(self, roll):
        if self.game_over: return []
        valid = []
        for i, pos in enumerate(self.players[self.turn]['pieces']):
            if pos == -1 and roll == 6: valid.append(i)
            elif pos != -1 and pos + roll <= 57: valid.append(i)
        return valid

    def move_piece(self, piece_idx, roll, training_mode=True):
        p_id = self.turn
        curr = self.players[p_id]['pieces'][piece_idx]
        new_pos = 0 if curr == -1 else curr + roll
        
        # State BEFORE move (for learning)
        prev_pieces = self.players[p_id]['pieces'].copy()
        state_key = self.brain.get_state_key(prev_pieces, roll)

        # Execute Move
        self.players[p_id]['pieces'][piece_idx] = new_pos
        reward = 1 + new_pos * 0.1
        log_txt = f"Moved P{piece_idx+1} to {new_pos}"

        # Kill Logic
        if new_pos < 51:
            my_glob = (p_id * 13 + new_pos) % 52
            if new_pos not in [0, 8, 13, 21, 26, 34, 39, 47]: 
                for oid in range(4):
                    if oid == p_id: continue
                    for opi, opos in enumerate(self.players[oid]['pieces']):
                        if opos != -1 and opos < 51:
                            if (oid * 13 + opos) % 52 == my_glob:
                                self.players[oid]['pieces'][opi] = -1
                                reward += 50
                                log_txt += " [KILL]"
        
        if new_pos == 57:
            reward += 100
            log_txt += " [HOME]"
            if sum(1 for p in self.players[p_id]['pieces'] if p == 57) == 4:
                self.game_over = True
                self.winner = ['Red','Green','Yellow','Blue'][p_id]
                log_txt += " [WINNER]"

        self.players[p_id]['score'] += reward
        
        # LEARN: Only learn if in training mode
        if training_mode:
            self.brain.learn(state_key, piece_idx, reward)
                                         
        return reward, log_txt

# --- GUI ---
class LudoGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ludo - Train & Play Model")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.resizable(False, False)
        
        self.brain = QBrain() # The Shared Brain
        self.logic = LudoLogic(self.brain)
        
        self.ai_running = False
        self._init_ui()
        self.canvas.bind("<Button-1>", self.on_board_click)

    def _init_ui(self):
        self.canvas = tk.Canvas(self, width=CELL_SIZE * BOARD_SIZE, height=WINDOW_HEIGHT, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)
        
        self.panel = tk.Frame(self, width=PANEL_WIDTH, bg=COLORS['panel_bg'])
        self.panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(self.panel, text="AI Model Manager", font=("Arial", 14, "bold"), bg=COLORS['panel_bg']).pack(pady=(15,5))
        
        # Mode
        self.mode_var = tk.StringVar(value="Human vs Human")
        cb = ttk.Combobox(self.panel, textvariable=self.mode_var, values=["Human vs Human", "AI vs AI (Training)", "Human(Red) vs AI (Play)"], state="readonly", width=22)
        cb.pack(pady=5)
        cb.bind("<<ComboboxSelected>>", self.reset_game)

        # Controls
        bf = tk.Frame(self.panel, bg=COLORS['panel_bg'])
        bf.pack(pady=5)
        tk.Button(bf, text="Start Game", command=self.start_ai_loop, bg="#2ECC71", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(bf, text="Stop", command=self.stop_ai, bg="#E74C3C", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(bf, text="Reset Board", command=self.reset_game, bg="#95A5A6", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Brain IO
        tk.Label(self.panel, text="Model File (The Brain):", bg=COLORS['panel_bg'], font=("Arial", 10, "bold")).pack(pady=(15,2))
        iof = tk.Frame(self.panel, bg=COLORS['panel_bg'])
        iof.pack(pady=5)
        tk.Button(iof, text="Save Model", command=self.save_model, bg="#3498DB", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(iof, text="Load Model", command=self.load_model, bg="#F39C12", fg="white").pack(side=tk.LEFT, padx=5)

        tk.Label(self.panel, text="Speed:", bg=COLORS['panel_bg']).pack(pady=(5,0))
        self.speed_slider = tk.Scale(self.panel, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, bg=COLORS['panel_bg'])
        self.speed_slider.set(0.1) # Fast default for training
        self.speed_slider.pack(fill=tk.X, padx=20)

        self.turn_lbl = tk.Label(self.panel, text="Red's Turn", font=("Arial", 12, "bold"), bg=COLORS['R'], fg="white", width=25, pady=10)
        self.turn_lbl.pack(pady=10)
        
        self.dice_frame = tk.Canvas(self.panel, width=60, height=60, bg=COLORS['panel_bg'], highlightthickness=0)
        self.dice_frame.pack()
        self.dice_frame.bind("<Button-1>", self.on_dice_click_human)
        self.draw_dice(0)
        
        self.log_box = tk.Text(self.panel, height=14, width=35, font=("Consolas", 8), relief=tk.FLAT, bg="#f9f9f9")
        self.log_box.pack(padx=10, pady=5)
        self.log_box.tag_config("INFO", foreground="gray")

        self._draw_board_static()
        self._refresh_pieces()

    # --- Draw & Logic Helpers (Standard) ---
    def _draw_board_static(self):
        self.canvas.delete("all")
        self._draw_base_rect(0, 0, COLORS['R'])
        self._draw_base_rect(9, 0, COLORS['G'])
        self._draw_base_rect(9, 9, COLORS['Y'])
        self._draw_base_rect(0, 9, COLORS['B'])
        mid = 7.5 * CELL_SIZE
        self.canvas.create_polygon(6*CELL_SIZE, 6*CELL_SIZE, 9*CELL_SIZE, 6*CELL_SIZE, mid, mid, fill=COLORS['G'], outline="black")
        self.canvas.create_polygon(9*CELL_SIZE, 6*CELL_SIZE, 9*CELL_SIZE, 9*CELL_SIZE, mid, mid, fill=COLORS['Y'], outline="black")
        self.canvas.create_polygon(9*CELL_SIZE, 9*CELL_SIZE, 6*CELL_SIZE, 9*CELL_SIZE, mid, mid, fill=COLORS['B'], outline="black")
        self.canvas.create_polygon(6*CELL_SIZE, 9*CELL_SIZE, 6*CELL_SIZE, 6*CELL_SIZE, mid, mid, fill=COLORS['R'], outline="black")
        
        def fill(coords, c):
            for x,y in coords: 
                x1, y1 = x*CELL_SIZE, y*CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x1+CELL_SIZE, y1+CELL_SIZE, fill=c, outline=COLORS['grid'])
        fill([(i,7) for i in range(1,6)], COLORS['R'])
        fill([(7,i) for i in range(1,6)], COLORS['G'])
        fill([(i,7) for i in range(9,14)], COLORS['Y'])
        fill([(7,i) for i in range(9,14)], COLORS['B'])
        fill([(1,6),(8,1),(13,8),(6,13)], "#999") 
        for r in range(15):
            for c in range(15):
                if not ((r<6 and c<6) or (r<6 and c>8) or (r>8 and c<6) or (r>8 and c>8)):
                    self.canvas.create_rectangle(c*CELL_SIZE, r*CELL_SIZE, (c+1)*CELL_SIZE, (r+1)*CELL_SIZE, width=1)
        for c, r in [(2,6), (6,1), (8,2), (12,6), (13,8), (8,12), (6,13), (2,8)]:
             self.canvas.create_text(c*CELL_SIZE+20, r*CELL_SIZE+22, text="★", font=("Arial", 20), fill="#7F8C8D")

    def _draw_base_rect(self, c, r, color):
        x, y = c * CELL_SIZE, r * CELL_SIZE
        self.canvas.create_rectangle(x, y, x + 6*CELL_SIZE, y + 6*CELL_SIZE, fill=color, outline="black")
        self.canvas.create_rectangle(x + CELL_SIZE, y + CELL_SIZE, x + 5*CELL_SIZE, y + 5*CELL_SIZE, fill="white", outline="black")

    def draw_dice(self, num):
        self.dice_frame.delete("all")
        self.dice_frame.create_rectangle(5, 5, 55, 55, fill="white", outline="#333", width=2)
        if num == 0:
            self.dice_frame.create_text(30, 30, text="ROLL", font=("Arial", 9, "bold"), fill="#999")
            return
        dots = {1:[(30,30)], 2:[(15,15),(45,45)], 3:[(15,15),(30,30),(45,45)], 4:[(15,15),(15,45),(45,15),(45,45)], 5:[(15,15),(15,45),(45,15),(45,45),(30,30)], 6:[(15,15),(15,45),(45,15),(45,45),(15,30),(45,30)]}
        for x, y in dots[num]:
            self.dice_frame.create_oval(x-3, y-3, x+3, y+3, fill="black")

    def _refresh_pieces(self):
        self.canvas.delete("piece")
        self.canvas.delete("hl")
        names = ["Red", "Green", "Yellow", "Blue"]
        cols = [COLORS['R'], COLORS['G'], COLORS['Y'], COLORS['B']]
        
        if self.logic.game_over:
            self.turn_lbl.config(text=f"WINNER: {self.logic.winner}", bg="#8E44AD")
        else:
            self.turn_lbl.config(text=f"{names[self.logic.turn]}'s Turn", bg=cols[self.logic.turn])

        for pid, p in enumerate(self.logic.players):
            for i in range(4):
                cx, cy = self.logic.get_piece_coords(pid, i)
                x, y = cx*CELL_SIZE, cy*CELL_SIZE
                overlap = 0
                for opid in range(4):
                    for opi in range(4):
                        if (opid < pid or (opid==pid and opi < i)) and \
                           self.logic.players[opid]['pieces'][opi] == p['pieces'][i] and \
                           p['pieces'][i] != -1:
                            overlap += 1
                off = overlap * 4
                
                # Highlight if valid
                if pid == self.logic.turn and self.logic.waiting_for_move and \
                   i in self.logic.get_valid_moves(self.logic.last_roll):
                    self.canvas.create_oval(x+2, y+2, x+38, y+38, outline="magenta", width=3, tags="hl")
                
                self.canvas.create_oval(x+5+off, y+5+off, x+35+off, y+35+off, fill=cols[pid], outline="white", width=2, tags="piece")
                self.canvas.create_text(x+20+off, y+20+off, text=str(i+1), fill="white", font=("Arial", 9, "bold"), tags="piece")

    def log(self, msg, tag=None):
        self.log_box.insert(tk.END, "> "+msg+"\n", tag)
        self.log_box.see(tk.END)

    # --- BRAIN IO ---
    def save_model(self):
        # We save the SHARED QBrain
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if filename:
            if self.brain.save_brain(filename):
                self.log(f"Model saved! Size: {len(self.brain.q_table)} states.", "INFO")
            else: self.log("Save failed.")

    def load_model(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filename:
            if self.brain.load_brain(filename):
                self.log(f"Model Loaded! States: {len(self.brain.q_table)}", "INFO")
                self.log("AI will now use these learned moves.")
            else: self.log("Load failed.")

    # --- Game Loop ---
    def reset_game(self, event=None):
        self.stop_ai()
        self.logic = LudoLogic(self.brain) # Keep the brain, reset the board!
        self.draw_dice(0)
        self.log_box.delete(1.0, tk.END)
        self.log("Board Reset. Brain kept.")
        self._refresh_pieces()

    def stop_ai(self):
        self.ai_running = False

    def start_ai_loop(self):
        if not self.ai_running:
            self.ai_running = True
            threading.Thread(target=self._ai_loop, daemon=True).start()

    def _ai_loop(self):
        while self.ai_running and not self.logic.game_over:
            # Check who plays
            mode = self.mode_var.get()
            is_ai_turn = False
            
            if mode == "AI vs AI (Training)": 
                is_ai_turn = True
            elif mode == "Human(Red) vs AI (Play)":
                if self.logic.turn != 0: is_ai_turn = True # AI plays Green, Yellow, Blue
            
            if is_ai_turn:
                # Is training allowed?
                # Yes if AI vs AI. No if playing against Human (Inference mode)
                train = (mode == "AI vs AI (Training)")
                self.play_turn_ai(training=train)
                time.sleep(float(self.speed_slider.get()))
            else:
                time.sleep(0.5)

    def play_turn_ai(self, training):
        roll = random.randint(1, 6)
        self.draw_dice(roll)
        self.logic.last_roll = roll
        valid = self.logic.get_valid_moves(roll)
        
        if not valid:
            self._next_turn(roll)
            return

        # Use Brain
        action = self.brain.choose_action(self.logic.players[self.logic.turn]['pieces'], roll, valid, training=training)
        
        # Execute (pass training flag to update weights or not)
        rwd, msg = self.logic.move_piece(action, roll, training_mode=training)
        
        p_tag = ["R","G","Y","B"][self.logic.turn]
        self.log(f"[{p_tag}] {msg} (Rwd:{rwd:.0f})", p_tag)
        self._refresh_pieces()
        self._next_turn(roll)

    # --- Human Input ---
    def on_dice_click_human(self, event):
        # Allow click only if Human Turn
        mode = self.mode_var.get()
        if mode == "AI vs AI (Training)": return
        if mode == "Human(Red) vs AI (Play)" and self.logic.turn != 0: return # Only Red is human
        
        if self.logic.game_over or self.logic.waiting_for_move: return
        
        roll = random.randint(1, 6)
        self.draw_dice(roll)
        self.logic.last_roll = roll
        valid = self.logic.get_valid_moves(roll)
        
        c = ["R","G","Y","B"][self.logic.turn]
        self.log(f"[{c}] Rolled {roll}")
        
        if not valid:
            self._next_turn(roll)
        else:
            self.logic.waiting_for_move = True
            self._refresh_pieces()

    def on_board_click(self, event):
        if not self.logic.waiting_for_move: return
        
        cx, cy = event.x // CELL_SIZE, event.y // CELL_SIZE
        valid = self.logic.get_valid_moves(self.logic.last_roll)
        
        chosen = -1
        for idx in valid:
            px, py = self.logic.get_piece_coords(self.logic.turn, idx)
            if abs(px-cx) < 1 and abs(py-cy) < 1:
                chosen = idx
                break
        
        if chosen != -1:
            # Human never trains the model, just plays
            self.logic.move_piece(chosen, self.logic.last_roll, training_mode=False)
            self.logic.waiting_for_move = False
            self._refresh_pieces()
            self._next_turn(self.logic.last_roll)

    def _next_turn(self, roll):
        if not self.logic.game_over:
            if roll != 6:
                self.logic.turn = (self.logic.turn + 1) % 4
            self.after(50, self._refresh_pieces)

if __name__ == "__main__":
    app = LudoGUI()
    app.mainloop()