import argparse

from flask import Flask, render_template

from utils.startup import prepare_directories


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("delos/index.html")


def main():
    parser = argparse.ArgumentParser(description="Delos server")
    parser.add_argument("--root", required=True, help="Root directory for knowledge, skills, storage, logs")
    args = parser.parse_args()

    prepare_directories(args.root)

    app.run(host="0.0.0.0", port=5052, debug=True)


if __name__ == "__main__":
    main()
