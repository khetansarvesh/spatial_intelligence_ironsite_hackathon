/**
 * SiteIQ Dashboard - Revamped Clean Interface
 * Flow: UPLOAD → PROCESSING → RESULTS → CHAT
 */

const API_BASE = '/api';

// Global state
let currentState = 'upload';
let summaryMarkdown = null;
let annotatedVideoUrl = null;

// ============================================================================
// State Management
// ============================================================================

const AppState = {
    UPLOAD: 'upload',
    PROCESSING: 'processing',
    RESULTS: 'results',
    CHAT: 'chat'
};

function setState(newState) {
    currentState = newState;

    // Hide all states
    document.getElementById('uploadState').style.display = 'none';
    document.getElementById('processingState').style.display = 'none';
    document.getElementById('resultsState').style.display = 'none';
    document.getElementById('chatState').style.display = 'none';

    // Show current state
    switch (newState) {
        case AppState.UPLOAD:
            document.getElementById('uploadState').style.display = 'flex';
            break;
        case AppState.PROCESSING:
            document.getElementById('processingState').style.display = 'flex';
            break;
        case AppState.RESULTS:
            document.getElementById('resultsState').style.display = 'flex';
            break;
        case AppState.CHAT:
            document.getElementById('chatState').style.display = 'flex';
            break;
    }
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    // Upload dropzone
    const dropzone = document.getElementById('uploadDropzone');
    const videoInput = document.getElementById('videoInput');

    dropzone.addEventListener('click', () => videoInput.click());
    videoInput.addEventListener('change', handleFileSelect);

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Results → Chat button
    document.getElementById('askQuestionsBtn').addEventListener('click', () => {
        setState(AppState.CHAT);
        // Copy video source to chat video
        const chatVideo = document.getElementById('chatVideo');
        chatVideo.src = annotatedVideoUrl;
        chatVideo.load();
    });

    // Chat → Results button
    document.getElementById('backToResultsBtn').addEventListener('click', () => {
        setState(AppState.RESULTS);
    });

    // Chat input
    const chatInput = document.getElementById('chatInput');
    const chatSend = document.getElementById('chatSend');

    chatSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Suggested questions
    document.querySelectorAll('.suggestion-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const question = chip.dataset.question;
            if (question) {
                document.getElementById('chatInput').value = question;
                sendMessage();
            }
        });
    });

    // Demo button - skip upload and show existing results
    document.getElementById('viewDemoBtn').addEventListener('click', () => {
        loadResults();
    });
}

// ============================================================================
// File Upload
// ============================================================================

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
}

async function handleFileUpload(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please select a video file');
        return;
    }

    // Switch to processing state
    setState(AppState.PROCESSING);
    updateProgress(0, 'Uploading video...');

    const formData = new FormData();
    formData.append('video', file);

    try {
        updateProgress(10, 'Uploading video...');

        const response = await fetch(`${API_BASE}/process-video`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        updateProgress(30, 'Processing frames...');
        await simulateProgress(30, 70, 'Analyzing activities and productivity...');

        updateProgress(80, 'Generating summary...');

        const result = await response.json();

        updateProgress(100, 'Analysis complete!');

        // Wait a moment then load results
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Load the results
        await loadResults();

    } catch (error) {
        console.error('Upload failed:', error);
        alert('Failed to process video. Please try again.');
        setState(AppState.UPLOAD);
    }
}

function updateProgress(percent, text) {
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = `${percent}%`;
    document.getElementById('processingStatus').textContent = text;
}

async function simulateProgress(start, end, text) {
    for (let i = start; i < end; i += 5) {
        await new Promise(resolve => setTimeout(resolve, 500));
        updateProgress(i, text);
    }
}

// ============================================================================
// Results Loading
// ============================================================================

async function loadResults() {
    try {
        // 1. Get markdown summary from API
        const summaryResponse = await fetch(`${API_BASE}/summary`);
        if (!summaryResponse.ok) throw new Error('Failed to load summary');
        const summaryData = await summaryResponse.json();
        summaryMarkdown = summaryData.markdown;

        // 2. Render markdown to summary container
        const summaryContent = document.getElementById('summaryContent');
        summaryContent.innerHTML = renderMarkdown(summaryMarkdown);

        // 3. Set annotated video URL
        annotatedVideoUrl = `${API_BASE}/video/annotated`;
        const annotatedVideo = document.getElementById('annotatedVideo');
        annotatedVideo.src = annotatedVideoUrl;
        annotatedVideo.load();

        // 4. Switch to results state
        setState(AppState.RESULTS);

    } catch (error) {
        console.error('Failed to load results:', error);
        alert('Failed to load analysis results. Please try again.');
        setState(AppState.UPLOAD);
    }
}

// ============================================================================
// Markdown Rendering
// ============================================================================

function renderMarkdown(markdown) {
    if (!markdown) return '';

    let html = markdown
        // Headers
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        // Bold
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.*?)\*/gim, '<em>$1</em>')
        // Code
        .replace(/`(.*?)`/gim, '<code>$1</code>')
        // Horizontal rules
        .replace(/^---$/gim, '<hr>')
        // Line breaks
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');

    // Handle lists
    html = html.replace(/(<br>)?- (.*?)(<br>|<\/p>|$)/g, '<li>$2</li>');
    html = html.replace(/(<li>.*<\/li>)+/g, '<ul>$&</ul>');

    // Handle tables (basic)
    const tableRegex = /\|(.+)\|[\r\n]+\|[-:|]+\|[\r\n]+((?:\|.+\|[\r\n]*)+)/g;
    html = html.replace(tableRegex, (match, header, body) => {
        const headerCells = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
        const bodyRows = body.trim().split('\n').map(row => {
            const cells = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        return `<table><thead><tr>${headerCells}</tr></thead><tbody>${bodyRows}</tbody></table>`;
    });

    // Wrap in paragraph if not already
    if (!html.startsWith('<')) {
        html = '<p>' + html + '</p>';
    }

    return html;
}

// ============================================================================
// Chat Interface
// ============================================================================

function addMessage(type, content) {
    const messagesContainer = document.getElementById('chatMessages');
    const message = document.createElement('div');
    message.className = `chat-message ${type}`;
    message.textContent = content;
    messagesContainer.appendChild(message);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return message;
}

function addLoadingMessage() {
    return addMessage('loading', 'Analyzing data...');
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();

    if (!question) return;

    // Add user message
    addMessage('user', question);
    input.value = '';

    // Add loading message
    const loadingMsg = addLoadingMessage();

    try {
        // 1. Get answer from ask agent
        const answerResponse = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        if (!answerResponse.ok) throw new Error('Failed to get answer');
        const answerData = await answerResponse.json();
        const answer = answerData.answer;

        // 2. Get evidence timestamps
        const evidenceResponse = await fetch(`${API_BASE}/evidence`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer })
        });

        let evidence = [];
        if (evidenceResponse.ok) {
            const evidenceData = await evidenceResponse.json();
            evidence = evidenceData.evidence || [];
        }

        // Remove loading message
        loadingMsg.remove();

        // 3. Add assistant response
        addMessage('assistant', answer);

        // 4. Handle evidence timestamps
        if (evidence.length > 0) {
            displayEvidence(evidence);
            // Auto-seek to first evidence
            seekToEvidence(evidence[0]);
        } else {
            hideEvidence();
        }

    } catch (error) {
        loadingMsg.remove();
        addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
        console.error('Query failed:', error);
    }
}

// ============================================================================
// Evidence Display & Seeking
// ============================================================================

function displayEvidence(evidenceList) {
    const container = document.getElementById('evidenceTimestamps');
    const indicator = document.getElementById('evidenceIndicator');

    container.innerHTML = '';

    evidenceList.forEach((ev, index) => {
        const chip = document.createElement('button');
        chip.className = 'evidence-chip';
        chip.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            ${formatTimestamp(ev.start_time)} - ${ev.description}
        `;
        chip.addEventListener('click', () => {
            // Remove active from all chips
            document.querySelectorAll('.evidence-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            seekToEvidence(ev);
        });
        container.appendChild(chip);
    });

    container.style.display = 'flex';
    indicator.style.display = 'flex';
}

function hideEvidence() {
    document.getElementById('evidenceTimestamps').innerHTML = '';
    document.getElementById('evidenceIndicator').style.display = 'none';
}

function seekToEvidence(evidence) {
    const video = document.getElementById('chatVideo');
    const targetTime = evidence.start_time;

    // Calculate window: targetTime - 1 minute (or 0)
    const startWindow = Math.max(0, targetTime - 60);

    video.currentTime = startWindow;
    video.play();
}

function formatTimestamp(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ============================================================================
// Helper Functions
// ============================================================================

function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}
