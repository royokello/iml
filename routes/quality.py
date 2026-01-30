from __future__ import annotations

from typing import Any, Dict

from flask import render_template, request

from video.analysis import run_analysis


def register(app, ensure_root_configured, strip_outer_quotes) -> None:
    @app.route("/quality", methods=["GET", "POST"])
    def quality_page():
        context: Dict[str, Any] = {}
        if request.method == "POST":
            source = strip_outer_quotes(request.form.get("source", "")).strip()
            resolution = int(request.form.get("resolution", 768))
            qualities_str = request.form.get("qualities", "")
            # Parse qualities
            crfs = []
            for q in qualities_str.split(","):
                q = q.strip()
                if q.isdigit():
                    crfs.append(int(q))

            sample_len = float(request.form.get("sample_len", 8.0))
            num_samples = int(request.form.get("num_samples", 8))

            context["last_input"] = {
                "source": source,
                "resolution": resolution,
                "qualities": qualities_str,
                "sample_len": sample_len,
                "num_samples": num_samples,
            }

            if not source or not crfs:
                context["error"] = "Please provide valid source and qualities."
            else:
                try:
                    results = run_analysis(
                        ensure_root_configured(),
                        source,
                        resolution,
                        0,  # start_crf unused in this signature
                        crfs,
                        sample_len,
                        num_samples,
                    )
                    if "error" in results:
                        context["error"] = results["error"]
                    else:
                        context["results"] = results
                except Exception as e:
                    context["error"] = f"Analysis failed: {str(e)}"

        return render_template("quality.html", **context)
