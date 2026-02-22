/**
 * SiteIQ Dashboard - Single Page Chat Interface
 */

const API_BASE = '/api';

// State
let selectedFile = null;
let videoLoaded = false;
let videoThumbnail = null;

// Elements
const chatMessages = document.getElementById('chatMessages');
const fileInput = document.getElementById('fileInput');
const promptInput = document.getElementById('promptInput');
const submitBtn = document.getElementById('submitBtn');
const attachBtn = document.getElementById('attachBtn');
const filePreview = document.getElementById('filePreview');
const fileName = document.getElementById('fileName');
const removeFileBtn = document.getElementById('removeFile');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Attach button - open file picker
    attachBtn.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('video/')) {
            selectedFile = file;
            fileName.textContent = file.name;
            filePreview.classList.remove('hidden');

            // Generate thumbnail
            generateThumbnail(file);
        }
    });

    // Remove file
    removeFileBtn.addEventListener('click', clearFile);

    // Submit
    submitBtn.addEventListener('click', handleSubmit);

    // Enter key
    promptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSubmit();
    });
});

function clearFile() {
    selectedFile = null;
    videoThumbnail = null;
    fileInput.value = '';
    filePreview.classList.add('hidden');
}

function generateThumbnail(file) {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.src = URL.createObjectURL(file);

    video.onloadeddata = () => {
        video.currentTime = 1; // Seek to 1 second
    };

    video.onseeked = () => {
        const canvas = document.createElement('canvas');
        canvas.width = 320;
        canvas.height = 180;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        videoThumbnail = canvas.toDataURL('image/jpeg', 0.7);
        URL.revokeObjectURL(video.src);
    };
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ============================================================================
// Message Handling
// ============================================================================

function addUserMessage(text) {
    // Remove welcome message and show header
    const welcome = document.getElementById('welcomeMessage');
    if (welcome) {
        welcome.remove();
        document.getElementById('topHeader').classList.remove('hidden');
    }

    const msg = document.createElement('div');
    msg.className = 'message user';
    msg.innerHTML = `<div class="message-content">${text}</div>`;
    chatMessages.appendChild(msg);
    scrollToBottom();
}

function addUserMessageWithVideo(thumbnail) {
    // Remove welcome message and show header
    const welcome = document.getElementById('welcomeMessage');
    if (welcome) {
        welcome.remove();
        document.getElementById('topHeader').classList.remove('hidden');
    }

    const msg = document.createElement('div');
    msg.className = 'message user';
    msg.innerHTML = `
        <div class="message-content video-message">
            ${thumbnail ? `<img src="${thumbnail}" alt="Video thumbnail" class="video-thumbnail">` : ''}
            <span>Analyze video</span>
        </div>
    `;
    chatMessages.appendChild(msg);
    scrollToBottom();
}

function addLoadingMessage() {
    const msg = document.createElement('div');
    msg.className = 'message assistant loading';
    msg.id = 'loadingMsg';
    msg.innerHTML = `<div class="message-content">Analyzing</div>`;
    chatMessages.appendChild(msg);
    scrollToBottom();
    return msg;
}

function removeLoadingMessage() {
    const loading = document.getElementById('loadingMsg');
    if (loading) loading.remove();
}

function addAssistantMessage(content) {
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    msg.innerHTML = `<div class="message-content">${content}</div>`;
    chatMessages.appendChild(msg);
    scrollToBottom();
    return msg;
}

function addRichResponse(summaryHtml, videoUrl) {
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    // Add cache-busting timestamp
    const videoSrc = `${videoUrl}?t=${Date.now()}`;
    msg.innerHTML = `
        <div class="message-content response-with-video">
            <video id="mainVideo" controls playsinline preload="auto">
                <source src="${videoSrc}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <div class="summary-content">${summaryHtml}</div>
        </div>
    `;
    chatMessages.appendChild(msg);

    // Ensure video loads
    const video = document.getElementById('mainVideo');
    video.load();

    scrollToBottom();
}

function addAnswerWithEvidence(answer, evidence, code) {
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    const msgId = `msg-${Date.now()}`;

    let html = `<div class="message-content response-with-evidence">`;

    // Add video clips if evidence exists
    if (evidence && evidence.length > 0) {
        html += `<div class="evidence-videos">`;
        evidence.forEach((ev, index) => {
            const clipUrl = `${API_BASE}/video/clip?start=${ev.start_time}&duration=2&t=${Date.now()}_${index}`;
            html += `
                <div class="evidence-video-item">
                    <video controls playsinline preload="auto">
                        <source src="${clipUrl}" type="video/mp4">
                    </video>
                    <span class="evidence-time">${formatTime(ev.start_time)}</span>
                </div>
            `;
        });
        html += `</div>`;
    }

    // Add content area with both text and code (code hidden by default)
    html += `
        <div class="content-area">
            <div class="answer-text" id="${msgId}-text">${answer}</div>
            <pre class="code-block hidden" id="${msgId}-code"><code class="language-python">${escapeHtml(code || '')}</code></pre>
        </div>
    `;

    // Add toggle if code exists
    if (code && code.trim()) {
        html += `
            <div class="toggle-section">
                <button class="view-toggle" id="${msgId}-toggle" onclick="toggleView('${msgId}')">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="16 18 22 12 16 6"></polyline>
                        <polyline points="8 6 2 12 8 18"></polyline>
                    </svg>
                    <span>Code</span>
                </button>
            </div>
        `;
    }

    html += '</div>';

    msg.innerHTML = html;
    chatMessages.appendChild(msg);

    // Apply syntax highlighting
    if (code && code.trim()) {
        const codeEl = document.querySelector(`#${msgId}-code code`);
        if (codeEl && window.hljs) {
            hljs.highlightElement(codeEl);
        }
    }

    scrollToBottom();
}

function toggleView(msgId) {
    const textEl = document.getElementById(`${msgId}-text`);
    const codeEl = document.getElementById(`${msgId}-code`);
    const toggleBtn = document.getElementById(`${msgId}-toggle`);

    const showingCode = codeEl.classList.contains('hidden');

    if (showingCode) {
        textEl.classList.add('hidden');
        codeEl.classList.remove('hidden');
        toggleBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
            </svg>
            <span>Text</span>
        `;
        toggleBtn.classList.add('active');
    } else {
        textEl.classList.remove('hidden');
        codeEl.classList.add('hidden');
        toggleBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="16 18 22 12 16 6"></polyline>
                <polyline points="8 6 2 12 8 18"></polyline>
            </svg>
            <span>Code</span>
        `;
        toggleBtn.classList.remove('active');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function seekVideo(timestamp) {
    const video = document.getElementById('mainVideo');
    if (video) {
        const startTime = Math.max(0, timestamp - 60);
        video.currentTime = startTime;
        video.play();
        video.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ============================================================================
// Submit Handling
// ============================================================================

async function handleSubmit() {
    const text = promptInput.value.trim();

    // If file is selected, process video
    if (selectedFile) {
        addUserMessageWithVideo(videoThumbnail);
        clearFile();
        promptInput.value = '';

        await processVideo();
        return;
    }

    // If text question and video is loaded, ask question
    if (text && videoLoaded) {
        addUserMessage(text);
        promptInput.value = '';
        await askQuestion(text);
        return;
    }

    // If no file and no video loaded, prompt to upload
    if (!videoLoaded) {
        addUserMessage(text || 'Help me analyze a video');
        addAssistantMessage('Please upload a construction video using the + button, and I\'ll analyze it for productivity insights.');
        promptInput.value = '';
    }
}

async function processVideo() {
    const loading = addLoadingMessage();

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 5000));

    try {
        // Get summary
        const summaryRes = await fetch(`${API_BASE}/summary`);
        if (!summaryRes.ok) throw new Error('Failed to load summary');
        const summaryData = await summaryRes.json();
        const summaryHtml = renderMarkdown(summaryData.markdown);

        // Get video URL
        const videoUrl = `${API_BASE}/video/annotated`;

        removeLoadingMessage();
        addRichResponse(summaryHtml, videoUrl);

        videoLoaded = true;
        promptInput.placeholder = 'Ask a question about the video...';

    } catch (error) {
        removeLoadingMessage();
        addAssistantMessage('Sorry, I encountered an error loading the analysis. Please try again.');
        console.error(error);
    }
}

async function askQuestion(question) {
    const loading = addLoadingMessage();

    try {
        // Get answer and code
        const answerRes = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        if (!answerRes.ok) throw new Error('Failed to get answer');
        const answerData = await answerRes.json();
        const answer = answerData.answer;
        const code = answerData.code || '';

        // Get evidence
        let evidence = [];
        try {
            const evidenceRes = await fetch(`${API_BASE}/evidence`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, answer })
            });
            if (evidenceRes.ok) {
                const evidenceData = await evidenceRes.json();
                evidence = evidenceData.evidence || [];
            }
        } catch (e) {
            console.log('Evidence fetch failed:', e);
        }

        removeLoadingMessage();
        addAnswerWithEvidence(answer, evidence, code);

    } catch (error) {
        removeLoadingMessage();
        addAssistantMessage('Sorry, I encountered an error. Please try again.');
        console.error(error);
    }
}

// ============================================================================
// Markdown Rendering
// ============================================================================

function renderMarkdown(md) {
    if (!md) return '';

    let html = md;

    // Process tables FIRST (before any line break conversion)
    const tableRegex = /\|(.+)\|\n\|[-:\s|]+\|\n((?:\|.+\|\n?)+)/g;
    html = html.replace(tableRegex, (match, header, body) => {
        const headerCells = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
        const bodyRows = body.trim().split('\n').map(row => {
            const cells = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        return `<table><thead><tr>${headerCells}</tr></thead><tbody>${bodyRows}</tbody></table>`;
    });

    // Headers
    html = html
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold and italic
    html = html
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/gim, '<em>$1</em>');

    // Horizontal rules
    html = html.replace(/^---$/gim, '<hr>');

    // Lists (process before line breaks)
    html = html.replace(/^- (.+)$/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

    // Line breaks
    html = html
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');

    // Clean up
    html = html.replace(/<br><\/p>/g, '</p>');
    html = html.replace(/<p><br>/g, '<p>');
    html = html.replace(/<ul><br>/g, '<ul>');
    html = html.replace(/<br><\/ul>/g, '</ul>');

    if (!html.startsWith('<')) {
        html = '<p>' + html + '</p>';
    }

    return html;
}
