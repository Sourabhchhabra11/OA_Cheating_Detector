
const $ = id => document.getElementById(id);

let monitoring = false;
let scoreChart;
let eventChart;
let previousTimelineSecond = -1;

const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
}[char]));

const eventNames = {
    looking_away: "Looking Away",
    phone: "Phone",
    multiple_faces: "Multiple Faces"
};

function initializeCharts() {
    scoreChart = new Chart($("score-chart"), {
        type: "line",
        data: {
            labels: ["Start"],
            datasets: [{
                label: "Suspicion Score",
                data: [0],
                borderColor: "#8b7cff",
                backgroundColor: "rgba(139,124,255,0.12)",
                fill: true,
                tension: 0.35,
                pointRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    ticks: { color: "#8c9ab4", maxTicksLimit: 8 },
                    grid: { color: "#273249" }
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: "#8c9ab4", precision: 0 },
                    grid: { color: "#273249" }
                }
            }
        }
    });

    eventChart = new Chart($("event-chart"), {
        type: "doughnut",
        data: {
            labels: ["Looking Away", "Phone", "Multiple Faces"],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ["#8b7cff", "#ffb45c", "#64a9ff"],
                borderColor: "#121a2c",
                borderWidth: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: "#aab7cd",
                        padding: 16,
                        usePointStyle: true,
                        font: { size: 10 }
                    }
                }
            }
        }
    });
}

async function apiRequest(url, method = "GET") {
    const response = await fetch(url, { method });
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.message || `Request failed: ${response.status}`);
    }

    return data;
}

function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remaining = seconds % 60;

    return String(minutes).padStart(2, "0") + ":" +
        String(remaining).padStart(2, "0");
}

function setBehaviorState(id, active) {
    const element = $(id);
    element.textContent = active ? "DETECTED" : "INACTIVE";
    element.classList.toggle("active", Boolean(active));
}

function updateTimeline(score, duration, running) {
    const second = Math.floor(duration);

    if (second === previousTimelineSecond) return;

    previousTimelineSecond = second;

    if (!running && duration === 0) {
        scoreChart.data.labels = ["Start"];
        scoreChart.data.datasets[0].data = [0];
    } else {
        scoreChart.data.labels.push(formatDuration(second));
        scoreChart.data.datasets[0].data.push(score);

        if (scoreChart.data.labels.length > 60) {
            scoreChart.data.labels.shift();
            scoreChart.data.datasets[0].data.shift();
        }
    }

    scoreChart.update("none");
}

function renderEvents(events, summary) {
    const table = $("event-table");
    const safeEvents = Array.isArray(events) ? events : [];

    $("event-total").textContent = summary.total ?? safeEvents.length;
    $("event-count-label").textContent =
        `${safeEvents.length} EVENT${safeEvents.length === 1 ? "" : "S"}`;

    if (safeEvents.length === 0) {
        table.innerHTML = `
            <tr>
                <td colspan="5" class="empty-row">
                    No suspicious events recorded in this session.
                </td>
            </tr>`;
    } else {
        table.innerHTML = safeEvents.slice().reverse().map(event => {
            const type = eventNames[event.event] || event.event;
            const time = event.timestamp || event.time || "—";

            return `
                <tr>
                    <td>${escapeHTML(time)}</td>
                    <td><span class="event-pill">${escapeHTML(type)}</span></td>
                    <td>${escapeHTML(type)} event recorded by the detector</td>
                    <td>${escapeHTML(event.score)}</td>
                    <td>Recorded</td>
                </tr>`;
        }).join("");
    }

    eventChart.data.datasets[0].data = [
        summary.looking_away ?? 0,
        summary.phone ?? 0,
        summary.multiple_faces ?? 0
    ];

    eventChart.update("none");
}

function updateDashboard(data) {
    monitoring = Boolean(data.running);

    $("score-value").textContent = data.suspicion_score ?? 0;
    $("score-progress").style.width =
        `${Math.min(100, Math.max(0, (data.suspicion_score ?? 0) * 10))}%`;

    $("face-count").textContent = data.face_count ?? 0;
    $("duration").textContent = formatDuration(data.duration ?? 0);

    $("gaze-label").textContent = `Gaze: ${data.gaze_direction ?? "—"}`;
    $("head-label").textContent = `Head pose: ${Number(data.yaw ?? 0).toFixed(1)}° yaw`;

    setBehaviorState("gaze-state", data.looking_away);
    setBehaviorState("phone-state", data.phone_detected);
    setBehaviorState("faces-state", data.multiple_faces);

    $("risk-status").textContent =
        data.running ? (data.status || "NORMAL") :
        data.error ? "DETECTION ERROR" : "NOT STARTED";

    $("risk-description").textContent = data.error
        ? data.error
        : data.running
            ? "Live results from the Python detection pipeline. Events indicate behavior requiring review, not proof of cheating."
            : "Start monitoring to begin receiving real detection results.";

    $("session-status").textContent = data.error
        ? "Monitoring error"
        : data.running
            ? "Monitoring active"
            : "Monitoring stopped";

    $("camera-status").textContent = data.running
        ? "CAMERA LIVE"
        : data.error
            ? "CAMERA ERROR"
            : "CAMERA OFF";

    $("feed-label").textContent = data.running
        ? "Camera connected"
        : data.error
            ? "Camera unavailable"
            : "Waiting for camera";

    $("start-btn").textContent = data.running
        ? "■ Stop Monitoring"
        : "▶ Start Monitoring";

    if (data.running) {
        $("camera-placeholder").hidden = true;
        $("camera-feed").hidden = false;

        if (!$("camera-feed").getAttribute("src")) {
            $("camera-feed").src = `/video_feed?t=${Date.now()}`;
        }
    } else {
        $("camera-feed").hidden = true;
        $("camera-feed").removeAttribute("src");
        $("camera-placeholder").hidden = false;

        if (data.error) {
            $("camera-placeholder").querySelector("h3").textContent =
                "Camera could not start";
            $("camera-placeholder").querySelector("p").textContent =
                data.error;
        } else {
            $("camera-placeholder").querySelector("h3").textContent =
                "Camera feed not connected";
            $("camera-placeholder").querySelector("p").textContent =
                "Start monitoring to connect the webcam.";
        }
    }

    $("report-message").textContent = data.error
        ? `Backend error: ${data.error}`
        : data.running
            ? "Live session data is being received from Python."
            : "Stop monitoring to save the session report.";

    renderEvents(data.events, data.summary || {});
    updateTimeline(
        data.suspicion_score ?? 0,
        data.duration ?? 0,
        data.running
    );
}

async function refreshStatus() {
    try {
        const data = await apiRequest("/api/status");
        updateDashboard(data);
    } catch (error) {
        $("session-status").textContent = "Backend connection failed";
        $("risk-status").textContent = "CONNECTION ERROR";
        $("risk-description").textContent =
            "Check that app.py is running and refresh this page.";
        $("report-message").textContent = error.message;
    }
}

$("start-btn").addEventListener("click", async () => {
    const button = $("start-btn");
    button.disabled = true;

    try {
        if (monitoring) {
            await apiRequest("/api/stop", "POST");
        } else {
            const result = await apiRequest("/api/start", "POST");

            if (!result.success) {
                throw new Error(result.message);
            }
        }

        await refreshStatus();
    } catch (error) {
        $("risk-description").textContent = error.message;
        await refreshStatus();
    } finally {
        button.disabled = false;
    }
});

$("reset-btn").addEventListener("click", async () => {
    const button = $("reset-btn");
    button.disabled = true;

    try {
        await apiRequest("/api/reset", "POST");
        previousTimelineSecond = -1;
        await refreshStatus();
    } catch (error) {
        $("risk-description").textContent = error.message;
    } finally {
        button.disabled = false;
    }
});

$("export-btn").addEventListener("click", async () => {
    try {
        const data = await apiRequest("/api/status");
        const headers = ["timestamp", "event", "score"];

        const rows = (data.events || []).map(event => [
            event.timestamp || event.time || "",
            event.event || "",
            event.score ?? ""
        ]);

        const csv = [headers, ...rows]
            .map(row => row.map(value =>
                `"${String(value).replace(/"/g, '""')}"`
            ).join(","))
            .join("\n");

        const url = URL.createObjectURL(
            new Blob([csv], { type: "text/csv;charset=utf-8;" })
        );

        const link = document.createElement("a");
        link.href = url;
        link.download = "oa_session_events.csv";
        link.click();

        URL.revokeObjectURL(url);
    } catch (error) {
        $("report-message").textContent = error.message;
    }
});

initializeCharts();
refreshStatus();
setInterval(refreshStatus, 1000);
