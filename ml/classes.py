"""AnySport hockey class ids. Fixed across datasets, the hub and the training scripts."""

CLASSES = ["player", "goaltender", "referee", "puck", "goal", "faceoff_dot", "center_ice"]
ID = {c: i for i, c in enumerate(CLASSES)}

# HockeyAI class names → ours (their data.yaml order can differ; we match by name, case-insensitive)
HOCKEYAI_MAP = {
    "players": "player", "player": "player",
    "goaltender": "goaltender", "goalie": "goaltender",
    "referee": "referee",
    "puck": "puck",
    "goal frame": "goal", "goal": "goal", "goalpost": "goal",
    "faceoff dots": "faceoff_dot", "faceoff": "faceoff_dot", "face-off": "faceoff_dot",
    "center ice": "center_ice", "centerice": "center_ice",
}
