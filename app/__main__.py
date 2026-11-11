"""`python -m app` -> Flask's server with debug OFF (debug would expose an interactive code-execution console)."""
import os

from . import create_app

app = create_app()
if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "5000")), debug=False)
