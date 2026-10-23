"""Run with: uv run python scripts/check_environment.py"""
import json
from as_personas.environment import check_environment, project_root

if __name__ == "__main__":
    report = check_environment()
    path = project_root() / "outputs/reports/environment_check.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {path}")
