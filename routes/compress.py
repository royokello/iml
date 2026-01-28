from flask import render_template, request

from video.compress import run_batch_compression


def register(app, ensure_root_configured):
    @app.route("/compress", methods=["GET", "POST"])
    def compress_page():
        context = {}
        if request.method == "POST":
            source = request.form.get("source", "").strip()
            output_dir = request.form.get("output", "").strip()
            resolution = int(request.form.get("resolution", 768))
            crf = int(request.form.get("crf", 25))

            context["last_input"] = {
                "source": source,
                "output": output_dir,
                "resolution": resolution,
                "crf": crf,
            }

            if not source or not output_dir:
                context["error"] = "Please provide valid source path and output directory."
            else:
                try:
                    results_list = run_batch_compression(
                        ensure_root_configured(),
                        source,
                        output_dir,
                        resolution,
                        crf,
                    )

                    if results_list and results_list[0].get("error") and len(results_list) == 1:
                        context["error"] = results_list[0]["error"]
                    elif not results_list:
                        context["error"] = "No results returned."
                    else:
                        total_source_size = 0
                        total_output_size = 0
                        processed_results = []

                        for r in results_list:
                            if r.get("success"):
                                s_size = r.get("source_size", 0)
                                o_size = r.get("output_size", 0)
                                total_source_size += s_size
                                total_output_size += o_size

                                r["source_size_mb"] = f"{(s_size / 1048576):.2f}"
                                r["output_size_mb"] = f"{(o_size / 1048576):.2f}"
                                r["filename"] = r.get("source", "").split("\\")[-1]
                            else:
                                r["filename"] = r.get("source", "Unknown").split("\\")[-1]
                            processed_results.append(r)

                        context["results"] = processed_results
                        context["summary"] = {
                            "total_files": len(processed_results),
                            "total_source_mb": f"{(total_source_size / 1048576):.2f}",
                            "total_output_mb": f"{(total_output_size / 1048576):.2f}",
                            "saved_mb": f"{((total_source_size - total_output_size) / 1048576):.2f}",
                        }

                except Exception as e:
                    context["error"] = f"Compression failed: {str(e)}"

        return render_template("compress.html", **context)
