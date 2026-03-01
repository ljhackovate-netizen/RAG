// ── Toast notification ─────────────────────────────────────────
function showToast(msg, type = 'success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast ${type}`;
  setTimeout(() => { el.className = 'toast hidden'; }, 3500);
}

// ── Q&A (only if enabled) ──────────────────────────────────────
async function askQA() {
  const clientId   = document.getElementById('client-id')?.value;
  const clientName = document.getElementById('client-name')?.value;
  const question   = document.getElementById('qa-q')?.value.trim();
  if (!clientId || !question) {
    showToast('Select client and enter a question', 'error');
    return;
  }
  const r = await fetch(
    `/generate/qa?client_id=${encodeURIComponent(clientId)}&client_name=${encodeURIComponent(clientName)}&question=${encodeURIComponent(question)}`,
    { method: 'POST' }
  );
  const data = await r.json();
  const box = document.getElementById('qa-answer');
  box.style.display = '';
  box.textContent = data.answer || data.detail || 'No answer returned.';
}
