import numpy as np
import tkinter as tk
import random

GRID_SIZE = 5
ACTIONS = ['up', 'down', 'left', 'right']
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9

CELL_SIZE = 100
MARGIN = 30  # space for labels

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.init_q_table()

    def init_q_table(self):
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                self.q_table[(x, y)] = {action: 0 for action in ACTIONS}

    def choose_action(self, state):
        if np.random.rand() < EPSILON:
            return random.choice(ACTIONS)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
        self.q_table[state][action] = new_value


class GridWorld(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Q-Learning GridWorld Improved GUI")
        self.geometry(f"{GRID_SIZE * CELL_SIZE + MARGIN*2}x{GRID_SIZE * CELL_SIZE + MARGIN*3 + 50}")

        self.canvas = tk.Canvas(self, width=GRID_SIZE * CELL_SIZE + MARGIN*2,
                                     height=GRID_SIZE * CELL_SIZE + MARGIN*2, bg='white')
        self.canvas.pack()

        self.status_label = tk.Label(self, text="", font=("Arial", 14))
        self.status_label.pack(pady=5)

        self.restart_button = tk.Button(self, text="Restart Training", command=self.restart_training)
        self.restart_button.pack(pady=5)

        self.agent = QLearningAgent()
        self.agent_pos = (0, 0)
        self.goal_pos = (GRID_SIZE-1, GRID_SIZE-1)
        self.episode = 0
        self.step = 0
        self.path = []  # To track the current episode path

        self.running = True
        self.draw_grid()
        self.after(1000, self.run_training)  # wait 1 sec before start

    def draw_grid(self):
        self.canvas.delete("all")
        # Draw grid and labels
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1 = MARGIN + j * CELL_SIZE
                y1 = MARGIN + i * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='gray', width=1)
                # Coordinates label
                self.canvas.create_text(x1 + 10, y1 + 10, anchor='nw', text=f"({i},{j})", fill='darkgray', font=("Arial", 8))

        # Draw path highlights
        for pos in self.path:
            px, py = pos
            x1 = MARGIN + py * CELL_SIZE + 20
            y1 = MARGIN + px * CELL_SIZE + 20
            x2 = x1 + CELL_SIZE - 40
            y2 = y1 + CELL_SIZE - 40
            self.canvas.create_rectangle(x1, y1, x2, y2, fill='#cceeff', outline='')

        # Draw goal
        gx, gy = self.goal_pos
        self.canvas.create_rectangle(
            MARGIN + gy * CELL_SIZE + 10,
            MARGIN + gx * CELL_SIZE + 10,
            MARGIN + gy * CELL_SIZE + 90,
            MARGIN + gx * CELL_SIZE + 90,
            fill='green'
        )

        # Draw agent
        ax, ay = self.agent_pos
        self.canvas.create_oval(
            MARGIN + ay * CELL_SIZE + 15,
            MARGIN + ax * CELL_SIZE + 15,
            MARGIN + ay * CELL_SIZE + 85,
            MARGIN + ax * CELL_SIZE + 85,
            fill='blue'
        )

    def move(self, action):
        x, y = self.agent_pos
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < GRID_SIZE - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < GRID_SIZE - 1:
            y += 1
        return (x, y)

    def run_training(self):
        if not self.running:
            return

        if self.episode >= 300:
            self.status_label.config(text=f"Training Complete after {self.episode} episodes!")
            return

        if self.step == 0:
            self.agent_pos = (0, 0)
            self.path = [self.agent_pos]

        if self.step < 50:
            state = self.agent_pos
            action = self.agent.choose_action(state)
            next_state = self.move(action)
            reward = 1 if next_state == self.goal_pos else -0.1
            self.agent.update(state, action, reward, next_state)
            self.agent_pos = next_state
            self.path.append(self.agent_pos)
            self.step += 1

            self.draw_grid()
            self.status_label.config(text=f"Episode: {self.episode + 1} / 300 | Step: {self.step} / 50")

            if self.agent_pos == self.goal_pos:
                self.episode += 1
                self.step = 0
                self.after(500, self.run_training)  # short pause after goal reached
            else:
                self.after(100, self.run_training)  # continue current episode

        else:
            # Episode ended without reaching goal
            self.episode += 1
            self.step = 0
            self.after(100, self.run_training)

    def restart_training(self):
        self.agent = QLearningAgent()
        self.agent_pos = (0, 0)
        self.episode = 0
        self.step = 0
        self.path = []
        self.running = True
        self.status_label.config(text="Training restarted!")
        self.draw_grid()
        self.after(500, self.run_training)


if __name__ == "__main__":
    app = GridWorld()
    app.mainloop()
