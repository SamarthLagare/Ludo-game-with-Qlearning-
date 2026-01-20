A Python Ludo game featuring a Q-Learning AI agent. Train the AI, save the model, and play Human vs. AI. Built with Tkinter.

**Long Description (for `README.md`):**

# 🎲 Ludo with Q-Learning AI

This project is a Python-based implementation of the board game Ludo, integrated with a **Reinforcement Learning (Q-Learning)** agent. The AI agent learns optimal strategies through self-play and can be saved/loaded to play against human users.

## 🚀 Features

* **Interactive GUI:** A complete, playable Ludo board built using `tkinter`.
* **Q-Learning Agent:** AI uses a Q-Table to learn states (piece positions) and actions (which piece to move).
* **Training Mode:** "AI vs AI" mode with a speed slider to fast-track the learning process.
* **Save/Load Brain:** Export the trained Q-Table to a `.json` file and load it later to play against a "smart" bot.
* **Game Mechanics:** Fully implemented rules including safe spots, killing opponents, and home-run logic.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:** `tkinter` (GUI), `random`, `json`, `threading`
* **Algorithm:** Q-Learning (Tabular RL)

## 🧠 How the AI Learns

The agent receives rewards based on its actions:

* **Move:** +1 (Small incentive to move forward)
* **Kill Opponent:** +50 (High incentive to capture)
* **Reach Goal:** +100 (Maximum reward)

During **Training Mode**, the agent explores random moves (-greedy). During **Play Mode**, it exploits the learned Q-Values from the saved JSON model to make the best decision.
