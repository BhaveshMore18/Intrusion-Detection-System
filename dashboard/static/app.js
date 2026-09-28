const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const scanStatus = document.getElementById("scanStatus");
const resultCard = document.getElementById("resultCard");
const verdictBadge = document.getElementById("verdictBadge");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const featureTable = document.getElementById("featureTable");

const FEATURE_LABELS = {
  file_size: "File size (bytes)",
  byte_entropy: "Byte entropy (0-8)",
  printable_ratio: "Printable byte ratio",
  null_byte_ratio: "Null byte ratio",
  high_byte_ratio: "High byte (>=0x80) ratio",
  is_pe: "Is PE executable",
  num_sections: "PE sections",
  num_imports: "Imported functions",
  num_suspicious_sections: "Suspicious section names",
  has_high_entropy_section: "Has high-entropy section",
  entry_point_outside_first_section: "Entry point outside 1st section",
};

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) scanFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) scanFile(fileInput.files[0]);
});

async function scanFile(file) {
  scanStatus.textContent = `Analyzing ${file.name}...`;
  resultCard.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/scan", { method: "POST", body: formData });

    let data;
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      data = await res.json();
    } else {
      // Server returned something unexpected (e.g. a proxy/dev-server error
      // page) -- don't let JSON.parse blow up with a cryptic message.
      const text = await res.text();
      scanStatus.textContent = `Error: server returned an unexpected response (HTTP ${res.status}).`;
      console.error("Non-JSON response from /api/scan:", text.slice(0, 500));
      return;
    }

    if (!res.ok) {
      scanStatus.textContent = `Error: ${data.error || "scan failed"}`;
      return;
    }

    scanStatus.textContent = `Last scan: ${file.name}`;
    renderResult(data.result, data.features);
    refreshHistory();
  } catch (err) {
    scanStatus.textContent = `Error: ${err.message}`;
  }
}

function renderResult(result, features) {
  resultCard.classList.remove("hidden");
  verdictBadge.textContent = result.label.toUpperCase();
  verdictBadge.className = `badge ${result.label}`;
  confidenceText.textContent = `${result.confidence}% malicious-confidence`;
  confidenceBar.style.width = `${result.confidence}%`;

  featureTable.innerHTML = "";
  for (const [key, value] of Object.entries(features)) {
    const row = document.createElement("tr");
    const label = FEATURE_LABELS[key] || key;
    const display = typeof value === "number" ? Number(value.toFixed(4)) : value;
    row.innerHTML = `<td>${label}</td><td>${display}</td>`;
    featureTable.appendChild(row);
  }
}

async function refreshHistory() {
  const res = await fetch("/api/history");
  const data = await res.json();

  document.getElementById("statTotal").textContent = data.stats.total_scans;
  document.getElementById("statMalicious").textContent = data.stats.malicious_found;
  document.getElementById("statBenign").textContent = data.stats.benign_found;
  document.getElementById("statAvgConf").textContent = `${data.stats.avg_confidence}%`;

  const body = document.getElementById("historyBody");
  if (!data.scans.length) {
    body.innerHTML = `<tr><td colspan="5" class="hint">No scans yet — upload a file above.</td></tr>`;
    return;
  }

  body.innerHTML = data.scans.map(scan => `
    <tr>
      <td>${escapeHtml(scan.filename)}</td>
      <td><span class="verdict-pill ${scan.label}">${scan.label.toUpperCase()}</span></td>
      <td>${scan.confidence}%</td>
      <td>${scan.entropy}</td>
      <td>${scan.timestamp}</td>
    </tr>
  `).join("");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

refreshHistory();
