from __future__ import annotations

from flask import jsonify, request
from image.dedup.grouping import components_for_items, dhash_from_cache

_MAX_CACHE = 32


def init_app(app, sessions, sessions_lock) -> None:
    @app.route("/api/filter", methods=["POST"])
    def filter_groups():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400

        sid = data.get("sid")
        if not sid:
            return jsonify({"error": "Missing 'sid' in request body."}), 400

        with sessions_lock:
            session = sessions.get(sid)

        if session is None:
            return jsonify({"error": f"Session not found: {sid}"}), 400

        items = session.get("items")
        if items is None:
            return jsonify({"error": "Scan not yet completed for this session."}), 400

        hash_size = data.get("hash_size", 16)
        mode = data.get("mode", "grey")
        threshold = data.get("threshold", 8)
        min_group_size = data.get("min_group_size", 2)

        # Recompute dhash at the requested hash_size from pixel cache if available
        pixel_cache = session.get("pixel_cache")
        if pixel_cache:
            mode_key = "edge" if mode == "edge" else "grey"
            recomputed = []
            for item in items:
                path = item[0]
                entry = pixel_cache.get(str(path))
                pixels = entry.get(mode_key) if entry else None
                if pixels:
                    dh = dhash_from_cache(pixels, _MAX_CACHE, hash_size)
                else:
                    dh = item[4]
                recomputed.append((path, item[1], item[2], item[3], dh))
            items = recomputed

        # Get near-duplicate groups
        groups = components_for_items(items, threshold)

        # Filter by minimum group size
        filtered = [g for g in groups if len(g) >= min_group_size]

        # Sort: most members first, then by first item's path
        filtered.sort(key=lambda g: (-len(g), g[0][1]))

        # Build response
        result_groups = []
        for group_index, group in enumerate(filtered):
            sorted_group = sorted(group, key=lambda item: item[1])

            images_payload = []
            for item in sorted_group:
                images_payload.append({
                    "path": str(item[0]),
                    "rel_path": item[1],
                    "width": item[2],
                    "height": item[3],
                })

            result_groups.append({
                "group_id": group_index,
                "images": images_payload,
            })

        total_clustered = sum(len(g) for g in filtered)

        return jsonify({
            "groups": result_groups,
            "total_groups": len(filtered),
            "total_clustered": total_clustered,
            "total_images": len(items),
        })
