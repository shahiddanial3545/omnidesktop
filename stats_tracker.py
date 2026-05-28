import json
import os
from datetime import date

class StatsTracker:
    def __init__(self, filepath='stats.json'):
        self.filepath = filepath
        self.data = self._load_data()

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
            self.data[today] = {"focus_minutes": 0, "posture_alerts": 0, "dashboard_taps": 0, "gestures_used": 0, "focus_score": 100}

        if category == "focus_score":
            # Average score for the day
            old_score = self.data[today].get("focus_score", 100)
            self.data[today]["focus_score"] = (old_score + value) // 2
        elif category in self.data[today]:
            self.data[today][category] += value

        self._save_data()

    def get_today_stats(self):
        today = str(date.today())
        return self.data.get(today, {"focus_minutes": 0, "posture_alerts": 0, "dashboard_taps": 0, "gestures_used": 0})

    def generate_report(self, html_path='stats.html'):
        rows = ""
        total_focus = 0
        total_alerts = 0

        for day, stats in sorted(self.data.items(), reverse=True):
            f = stats.get('focus_minutes', 0)
            a = stats.get('posture_alerts', 0)
            total_focus += f
            total_alerts += a
            rows += f"""
            <tr>
                <td>{day}</td>
                <td>{f}</td>
                <td>{a}</td>
                <td>{stats.get('dashboard_taps', 0)}</td>
                <td>{stats.get('gestures_used', 0)}</td>
            </tr>
            """

        html_content = f"""
        <html>
        <head>
            <title>Omni-Desk Insights</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1a1a2e; color: #e0e0e0; padding: 40px; line-height: 1.6; }}
                .container {{ max-width: 1000px; margin: auto; }}
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
                    <div style="text-align: right;">v1.0 Production Layer</div>
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
                        <div>Alerts Avoided</div>
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
                            <th>Date</th>
                            <th>Focus (Min)</th>
                            <th>Posture Alerts</th>
                            <th>Dashboard Taps</th>
                            <th>Gestures Used</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        with open(html_path, 'w') as f:
            f.write(html_content)
