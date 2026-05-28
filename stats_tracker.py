import json
import os
import time
from datetime import date
from collections import deque

# ============================================================
# PILLAR 4 — EWMA FOCUS SCORE + COGNITIVE LOAD PREDICTOR
# ============================================================

class EWMAFocusScorer:
    """
    PILLAR 4: Exponentially Weighted Moving Average focus score.

    The classic approach simply counts events and deducts.
    EWMA instead keeps a running weighted estimate that decays
    old events and reacts quickly to recent ones.

    Score drivers (weighted deductions applied per event):
      posture_break   : −12 pts
      privacy_trigger : −6  pts
      phone_detected  : −16 pts
      eye_fatigue     : −8  pts  (triggered when blink rate < threshold)
      task_switch     : −4  pts  (mode change while in Focus)

    Micro-break recommendation:
      When predicted_cognitive_load > 0.85 for 3+ consecutive minutes,
      a micro-break is recommended.
    """
    WEIGHTS = {
        'posture_break':   12.0,
        'privacy_trigger':  6.0,
        'phone_detected':  16.0,
        'eye_fatigue':      8.0,
        'task_switch':      4.0,
    }
    EWMA_ALPHA   = 0.15    # smoothing factor (lower = smoother, slower)
    RECOVER_RATE = 0.3     # pts recovered per minute without incident

    def __init__(self):
        self.score = 100.0           # current EWMA score
        self._load_history = deque(maxlen=180)  # 1 sample/sec × 3 min
        self._break_recommended = False
        self._session_events = []
        self._last_update = time.time()

    def apply_event(self, event_type: str):
        """Deduct weight for event_type using EWMA."""
        weight = self.WEIGHTS.get(event_type, 5.0)
        # EWMA deduction: blend deduction into current score
        target = max(0.0, self.score - weight)
        self.score = self.EWMA_ALPHA * target + (1 - self.EWMA_ALPHA) * self.score
        self._session_events.append((time.time(), event_type))

    def tick(self):
        """
        Call every second (or every N frames) to:
          1. Apply passive recovery
          2. Update cognitive-load history
          3. Determine micro-break recommendation
        """
        now = time.time()
        elapsed_min = (now - self._last_update) / 60.0
        self._last_update = now

        # Passive recovery
        self.score = min(100.0, self.score + self.RECOVER_RATE * elapsed_min)

        # Cognitive load is inverse of score (0=relaxed, 1=overloaded)
        cog_load = max(0.0, (100.0 - self.score) / 100.0)
        self._load_history.append(cog_load)

        # Recommend break if high load persists ≥ 3 min (180 samples)
        if len(self._load_history) >= 180:
            avg_load = sum(self._load_history) / len(self._load_history)
            self._break_recommended = avg_load > 0.85
        else:
            self._break_recommended = False

        return round(self.score, 1), self._break_recommended

    def get_session_score(self) -> int:
        """Return final rounded session score and reset."""
        final = int(round(self.score))
        self._session_events.clear()
        return final

    def reset_session(self):
        self.score = 100.0
        self._load_history.clear()
        self._break_recommended = False
        self._session_events.clear()
        self._last_update = time.time()


class StatsTracker:
    def __init__(self, filepath='stats.json'):
        self.filepath = filepath
        self.data = self._load_data()
        # PILLAR 4 — EWMA scorer instance
        self.ewma = EWMAFocusScorer()

    def _load_data(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_data(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=4)

    def log_stat(self, category, value=1):
        today = str(date.today())
        if today not in self.data:
            self.data[today] = {
                "focus_minutes": 0, "posture_alerts": 0,
                "dashboard_taps": 0, "gestures_used": 0,
                "focus_score": 100, "micro_breaks_suggested": 0
            }

        if category == "focus_score":
            old_score = self.data[today].get("focus_score", 100)
            self.data[today]["focus_score"] = (old_score + value) // 2
        elif category in self.data[today]:
            self.data[today][category] += value

        self._save_data()

    # PILLAR 4 — feed EWMA events directly
    def record_focus_event(self, event_type: str):
        """Call this from HabitEngine instead of manual deductions."""
        self.ewma.apply_event(event_type)

    def ewma_tick(self):
        """Returns (current_score, break_recommended). Call every second."""
        return self.ewma.tick()

    def finalise_session(self) -> int:
        score = self.ewma.get_session_score()
        self.log_stat("focus_score", score)
        return score

    def get_today_stats(self):
        today = str(date.today())
        return self.data.get(today, {
            "focus_minutes": 0, "posture_alerts": 0,
            "dashboard_taps": 0, "gestures_used": 0
        })

    def generate_report(self, html_path='stats.html'):
        rows = ""
        total_focus = 0
        total_alerts = 0

        for day, stats in sorted(self.data.items(), reverse=True):
            f = stats.get('focus_minutes', 0)
            a = stats.get('posture_alerts', 0)
            fs = stats.get('focus_score', '-')
            total_focus += f
            total_alerts += a
            rows += f"""
            <tr>
                <td>{day}</td>
                <td>{f}</td>
                <td>{a}</td>
                <td>{stats.get('dashboard_taps', 0)}</td>
                <td>{stats.get('gestures_used', 0)}</td>
                <td>{fs}/100</td>
            </tr>
            """

        html_content = f"""
        <html>
        <head>
            <title>Omni-Desk Insights</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1a1a2e; color: #e0e0e0; padding: 40px; line-height: 1.6; }}
                .container {{ max-width: 1100px; margin: auto; }}
                .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #30475e; padding-bottom: 20px; margin-bottom: 30px; }}
                h1 {{ color: #00adb5; margin: 0; font-size: 2.5em; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
                .card {{ background: #16213e; padding: 25px; border-radius: 15px; border: 1px solid #0f3460; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
                .card h3 {{ margin: 0; color: #00adb5; font-size: 1.1em; text-transform: uppercase; }}
                .card .value {{ font-size: 2.5em; font-weight: bold; margin: 10px 0; color: #fff; }}
                table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: #16213e; border-radius: 15px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}
                th, td {{ padding: 18px; text-align: left; border-bottom: 1px solid #0f3460; }}
                th {{ background-color: #0f3460; color: #00adb5; text-transform: uppercase; letter-spacing: 1px; }}
                tr:last-child td {{ border-bottom: none; }}
                tr:hover {{ background-color: #1f4068; transition: 0.3s; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>OMNI-DESK INSIGHTS</h1>
                    <div style="text-align: right;">v2.0 — AI Enhanced</div>
                </div>
                <div class="stats-grid">
                    <div class="card">
                        <h3>Total Focus Time</h3>
                        <div class="value">{total_focus}</div>
                        <div>Minutes</div>
                    </div>
                    <div class="card">
                        <h3>Posture Guardian</h3>
                        <div class="value" style="color: #e94560;">{total_alerts}</div>
                        <div>Alerts</div>
                    </div>
                    <div class="card">
                        <h3>AI Interactions</h3>
                        <div class="value">{sum(s.get('gestures_used', 0) + s.get('dashboard_taps', 0) for s in self.data.values())}</div>
                        <div>Actions Performed</div>
                    </div>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th><th>Focus (Min)</th><th>Posture Alerts</th>
                            <th>Dashboard Taps</th><th>Gestures Used</th><th>EWMA Focus Score</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </body>
        </html>
        """
        with open(html_path, 'w') as f:
            f.write(html_content)
