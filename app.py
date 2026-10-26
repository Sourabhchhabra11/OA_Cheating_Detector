
from flask import Flask, render_template, jsonify, Response

from monitor import MonitoringService


app = Flask(__name__)
monitor = MonitoringService()


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    return jsonify(monitor.get_status())


@app.route("/api/start", methods=["POST"])
def start_monitoring():
    success, message = monitor.start()

    return jsonify({
        "success": success,
        "message": message
    }), 200 if success else 409


@app.route("/api/stop", methods=["POST"])
def stop_monitoring():
    monitor.stop()

    return jsonify({
        "success": True,
        "message": "Monitoring stopped."
    })


@app.route("/api/reset", methods=["POST"])
def reset_monitoring():
    monitor.reset()

    return jsonify({
        "success": True,
        "message": "Session reset."
    })


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = monitor.get_jpeg()

            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
