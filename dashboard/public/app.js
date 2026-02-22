/**
 * SiteIQ Dashboard - Chat-First Interface
 * Revamped for modern 2026 design
 */

const API_BASE = '/api';
let currentReport = null;
let currentReportFile = null;
let notifications = [];
let backgroundProcessing = null;

// ============================================================================
// State Management
// ============================================================================

const AppState = {
    UPLOAD: 'upload',
    PROCESSING: 'processing',
    CHAT: 'chat'
};

let currentState = AppState.UPLOAD;

function setState(newState) {
    currentState = newState;

    // Hide all states
    document.getElementById('uploadState').style.display = 'none';
    document.getElementById('processingState').style.display = 'none';
    document.getElementById('chatInterface').style.display = 'none';

    // Show current state
    switch (newState) {
        case AppState.UPLOAD:
            document.getElementById('uploadState').style.display = 'flex';
            break;
        case AppState.PROCESSING:
            document.getElementById('processingState').style.display = 'flex';
            break;
        case AppState.CHAT:
            document.getElementById('chatInterface').style.display = 'grid';
            break;
    }
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadReports();
});

function setupEventListeners() {
    // Report selection
    document.getElementById('reportSelect').addEventListener('change', handleReportChange);

    // New analysis button - opens modal instead of changing state
    document.getElementById('newAnalysisBtn').addEventListener('click', openUploadModal);

    // Modal controls
    document.getElementById('modalCloseBtn').addEventListener('click', closeUploadModal);
    document.getElementById('uploadModal').addEventListener('click', (e) => {
        if (e.target.id === 'uploadModal') {
            closeUploadModal();
        }
    });

    // Modal upload dropzone
    const modalDropzone = document.getElementById('modalUploadDropzone');
    const modalVideoInput = document.getElementById('modalVideoInput');

    modalDropzone.addEventListener('click', () => modalVideoInput.click());
    modalVideoInput.addEventListener('change', handleModalFileSelect);

    // Modal drag and drop
    modalDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        modalDropzone.classList.add('dragover');
    });

    modalDropzone.addEventListener('dragleave', () => {
        modalDropzone.classList.remove('dragover');
    });

    modalDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        modalDropzone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // Original dropzone for upload state (keep for backward compatibility)
    const dropzone = document.getElementById('uploadDropzone');
    if (dropzone) {
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
    }

    // Chat input
    const chatInput = document.getElementById('chatInput');
    const chatSend = document.getElementById('chatSend');

    chatSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !chatSend.disabled) {
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

    // Notification bell
    const notificationBtn = document.getElementById('notificationBtn');
    const notificationDropdown = document.getElementById('notificationDropdown');

    notificationBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        notificationDropdown.classList.toggle('active');
    });

    document.addEventListener('click', () => {
        notificationDropdown.classList.remove('active');
    });

    notificationDropdown.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    document.getElementById('clearNotifications').addEventListener('click', clearAllNotifications);

    // Background processing buttons
    document.getElementById('processInBackgroundBtn').addEventListener('click', processInBackground);
    document.getElementById('cancelProcessingBtn').addEventListener('click', cancelProcessing);

    // Panel toggle (for mobile)
    const panelToggle = document.getElementById('panelToggle');
    if (panelToggle) {
        panelToggle.addEventListener('click', () => {
            const panel = document.getElementById('panelContent');
            panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
        });
    }

    // Scroll indicator
    const panelContent = document.getElementById('panelContent');
    const scrollIndicator = document.getElementById('scrollIndicator');

    if (panelContent && scrollIndicator) {
        scrollIndicator.addEventListener('click', () => {
            panelContent.scrollTo({
                top: panelContent.scrollHeight,
                behavior: 'smooth'
            });
        });

        panelContent.addEventListener('scroll', () => {
            const hasScrolled = panelContent.scrollTop > 50;
            const isAtBottom = panelContent.scrollHeight - panelContent.scrollTop <= panelContent.clientHeight + 50;

            if (hasScrolled || isAtBottom) {
                scrollIndicator.classList.add('hidden');
            } else {
                scrollIndicator.classList.remove('hidden');
            }
        });

        // Check scroll on load
        setTimeout(() => {
            const needsScroll = panelContent.scrollHeight > panelContent.clientHeight;
            if (!needsScroll) {
                scrollIndicator.classList.add('hidden');
            }
        }, 500);
    }
}

// ============================================================================
// API Functions
// ============================================================================

async function api(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

async function loadReports() {
    try {
        const data = await api('/reports');
        const select = document.getElementById('reportSelect');

        select.innerHTML = '<option value="">Select Analysis Session...</option>';

        data.reports.forEach(report => {
            const option = document.createElement('option');
            option.value = report.name;
            option.textContent = report.name.replace('_report.json', '').replace(/_/g, ' ');
            select.appendChild(option);
        });

        // Auto-select first report if available
        if (data.reports.length > 0) {
            select.value = data.reports[0].name;
            handleReportChange();
        }
    } catch (error) {
        console.error('Failed to load reports:', error);
    }
}

function handleReportChange() {
    const filename = document.getElementById('reportSelect').value;
    if (filename) {
        loadReport(filename);
    } else {
        setState(AppState.UPLOAD);
    }
}

async function loadReport(filename) {
    try {
        const report = await api(`/report/${filename}`);
        currentReport = report;
        currentReportFile = filename;

        // Switch to chat interface
        setState(AppState.CHAT);

        // Clear previous chat
        document.getElementById('chatMessages').innerHTML = '';

        // Update metrics panel
        updateMetricsPanel(report);

        // Enable chat
        document.getElementById('chatInput').disabled = false;
        document.getElementById('chatSend').disabled = false;

        // Add system message with analysis summary
        addSystemMessage(generateAnalysisSummary(report));

    } catch (error) {
        console.error('Failed to load report:', error);
        alert('Failed to load analysis report');
    }
}

// ============================================================================
// Notification Functions
// ============================================================================

function addNotification(title, message, status = 'processing') {
    const notification = {
        id: Date.now(),
        title,
        message,
        status,
        time: new Date().toLocaleTimeString(),
        unread: true
    };

    notifications.unshift(notification);
    updateNotificationUI();
    return notification.id;
}

function updateNotification(id, updates) {
    const notification = notifications.find(n => n.id === id);
    if (notification) {
        Object.assign(notification, updates);
        updateNotificationUI();
    }
}

function updateNotificationUI() {
    const list = document.getElementById('notificationList');
    const badge = document.getElementById('notificationBadge');
    const btn = document.getElementById('notificationBtn');

    if (notifications.length === 0) {
        list.innerHTML = '<div class="notification-empty">No notifications yet</div>';
        badge.style.display = 'none';
        btn.classList.remove('has-notifications');
        return;
    }

    const unreadCount = notifications.filter(n => n.unread).length;
    if (unreadCount > 0) {
        badge.textContent = unreadCount;
        badge.style.display = 'block';
        btn.classList.add('has-notifications');
    } else {
        badge.style.display = 'none';
        btn.classList.remove('has-notifications');
    }

    list.innerHTML = notifications.map(n => `
        <div class="notification-item ${n.unread ? 'unread' : ''}" onclick="markAsRead(${n.id})">
            <div class="notification-item-header">
                <span class="notification-title">${n.title}</span>
                <span class="notification-time">${n.time}</span>
            </div>
            <div class="notification-message">${n.message}</div>
            <span class="notification-status ${n.status}">${n.status === 'success' ? '✓ Complete' : n.status === 'processing' ? '⟳ Processing' : '✗ Failed'}</span>
        </div>
    `).join('');
}

function markAsRead(id) {
    const notification = notifications.find(n => n.id === id);
    if (notification) {
        notification.unread = false;
        updateNotificationUI();

        // If it's a completed notification, load that report
        if (notification.status === 'success' && notification.reportFile) {
            document.getElementById('reportSelect').value = notification.reportFile;
            loadReport(notification.reportFile);
            document.getElementById('notificationDropdown').classList.remove('active');
        }
    }
}

window.markAsRead = markAsRead; // Make it globally accessible

function clearAllNotifications() {
    notifications = [];
    updateNotificationUI();
}

// ============================================================================
// Modal Functions
// ============================================================================

function openUploadModal() {
    document.getElementById('uploadModal').classList.add('active');
    // Reset modal to upload view
    document.getElementById('modalUploadArea').style.display = 'block';
    document.getElementById('modalProcessingArea').style.display = 'none';
}

function closeUploadModal() {
    document.getElementById('uploadModal').classList.remove('active');
}

function processInBackground() {
    closeUploadModal();
    // Processing continues in background
    // Notification will show when complete
}

function cancelProcessing() {
    if (backgroundProcessing) {
        // Cancel the background processing
        backgroundProcessing.cancelled = true;
    }
    closeUploadModal();
    updateNotification(backgroundProcessing.notificationId, {
        status: 'error',
        message: 'Processing cancelled by user'
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

function handleModalFileSelect(e) {
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

    // Check if we're in modal or full page mode
    const isModal = document.getElementById('uploadModal').classList.contains('active');

    // Create notification
    const notificationId = addNotification(
        'Video Analysis Started',
        `Processing ${file.name}...`,
        'processing'
    );

    if (isModal) {
        // Switch modal to processing view
        document.getElementById('modalUploadArea').style.display = 'none';
        document.getElementById('modalProcessingArea').style.display = 'block';
        updateModalProgress(0, 'Uploading video...');

        // Set up background processing
        backgroundProcessing = {
            notificationId,
            file: file,
            cancelled: false
        };
    } else {
        // Switch to processing state
        setState(AppState.PROCESSING);
        updateProgress(0, 'Uploading video...');
    }

    const formData = new FormData();
    formData.append('video', file);

    const quickProcessing = isModal
        ? document.getElementById('modalQuickProcessing').checked
        : document.getElementById('quickProcessing').checked;

    if (quickProcessing) {
        formData.append('max_frames', '300');
    }

    try {
        if (isModal) {
            updateModalProgress(10, 'Uploading video...');
        } else {
            updateProgress(10, 'Uploading video...');
        }

        const response = await fetch(`${API_BASE}/process-video`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        if (isModal) {
            updateModalProgress(50, 'Processing frames and detecting activities...');
            await simulateModalProgress();
        } else {
            updateProgress(50, 'Processing frames and detecting activities...');
            await simulateProgress();
        }

        // Check if cancelled
        if (backgroundProcessing && backgroundProcessing.cancelled) {
            return;
        }

        const result = await response.json();

        if (isModal) {
            updateModalProgress(100, 'Analysis complete!');
        } else {
            updateProgress(100, 'Analysis complete!');
        }

        // Update notification
        updateNotification(notificationId, {
            title: 'Analysis Complete',
            message: `${file.name} processed successfully`,
            status: 'success',
            reportFile: result.report_file
        });

        // Wait a moment then load the report
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Reload reports and select the new one
        await loadReports();
        document.getElementById('reportSelect').value = result.report_file;
        await loadReport(result.report_file);

        // Close modal if in modal mode
        if (isModal) {
            closeUploadModal();
        }

        backgroundProcessing = null;

    } catch (error) {
        console.error('Upload failed:', error);

        // Update notification
        updateNotification(notificationId, {
            title: 'Analysis Failed',
            message: error.message || 'Failed to process video',
            status: 'error'
        });

        if (!backgroundProcessing || !backgroundProcessing.cancelled) {
            alert('Failed to process video. Please try again.');
        }

        if (isModal) {
            closeUploadModal();
        } else {
            setState(AppState.UPLOAD);
        }

        backgroundProcessing = null;
    }
}

function updateProgress(percent, text) {
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = `${percent}%`;
    document.getElementById('processingStatus').textContent = text;
}

async function simulateProgress() {
    for (let i = 50; i < 95; i += 5) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        updateProgress(i, 'Analyzing activities and productivity...');
    }
}

function updateModalProgress(percent, text) {
    document.getElementById('modalProgressFill').style.width = `${percent}%`;
    document.getElementById('modalProgressText').textContent = `${percent}%`;
    document.getElementById('modalProcessingStatus').textContent = text;
}

async function simulateModalProgress() {
    for (let i = 50; i < 95; i += 5) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        updateModalProgress(i, 'Analyzing activities and productivity...');
    }
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

function addSystemMessage(content) {
    return addMessage('system', content);
}

function addLoadingMessage() {
    return addMessage('loading', 'Analyzing data...');
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();

    if (!question || !currentReportFile) return;

    // Add user message
    addMessage('user', question);
    input.value = '';

    // Add loading message
    const loadingMsg = addLoadingMessage();

    try {
        const data = await api('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                report_file: currentReportFile,
                question: question,
                provider: 'openai'
            })
        });

        // Remove loading message
        loadingMsg.remove();

        // Add assistant response
        addMessage('assistant', data.answer);

    } catch (error) {
        loadingMsg.remove();
        addMessage('assistant', 'Sorry, I encountered an error analyzing your question. Please try again.');
        console.error('Query failed:', error);
    }
}

// ============================================================================
// Metrics Panel
// ============================================================================

function updateMetricsPanel(report) {
    // Primary KPIs
    const score = (report.productivity_score * 100).toFixed(1);
    document.getElementById('productivityScore').textContent = `${score}%`;
    document.getElementById('productivityRating').textContent = getProductivityRating(report.productivity_score);

    document.getElementById('sessionDuration').textContent = formatTime(report.session_duration);

    const activeTime = report.productive_time || (report.session_duration - report.idle_time);
    document.getElementById('activeTime').textContent = formatTime(activeTime);
    document.getElementById('activePercentage').textContent = `${((activeTime / report.session_duration) * 100).toFixed(1)}% of session`;

    document.getElementById('idleTime').textContent = formatTime(report.idle_time);
    document.getElementById('idlePercentage').textContent = `${report.idle_percentage.toFixed(1)}% of session`;

    // Tool Usage
    updateToolList(report.tool_usage || {});

    // Highlights
    updateHighlights(report.insights || [], report.recommendations || []);

    // Activity Breakdown
    updateActivityBars(report.activity_breakdown || {});
}

function updateToolList(toolUsage) {
    const container = document.getElementById('toolList');
    container.innerHTML = '';

    const tools = Object.values(toolUsage).sort((a, b) => b.total_time - a.total_time);

    if (tools.length === 0) {
        container.innerHTML = '<div class="loading-placeholder">No tool data available</div>';
        return;
    }

    tools.forEach(tool => {
        const item = document.createElement('div');
        item.className = 'tool-item';
        item.innerHTML = `
            <span class="tool-name">${tool.tool_name}</span>
            <span class="tool-time">${formatTime(tool.total_time)}</span>
        `;
        container.appendChild(item);
    });
}

function updateHighlights(insights, recommendations) {
    const container = document.getElementById('highlightsList');
    container.innerHTML = '';

    // Show top 3 insights
    insights.slice(0, 3).forEach(insight => {
        const item = document.createElement('div');
        item.className = 'highlight-item';
        item.textContent = insight;
        container.appendChild(item);
    });

    // Show top 2 recommendations
    recommendations.slice(0, 2).forEach(rec => {
        const item = document.createElement('div');
        item.className = 'highlight-item warning';
        item.textContent = rec;
        container.appendChild(item);
    });

    if (insights.length === 0 && recommendations.length === 0) {
        container.innerHTML = '<div class="loading-placeholder">No highlights available</div>';
    }
}

function updateActivityBars(activityBreakdown) {
    const container = document.getElementById('activityBars');
    container.innerHTML = '';

    const activities = Object.values(activityBreakdown)
        .sort((a, b) => b.percentage - a.percentage)
        .slice(0, 5); // Top 5 activities

    if (activities.length === 0) {
        container.innerHTML = '<div class="loading-placeholder">No activity data available</div>';
        return;
    }

    activities.forEach(activity => {
        const item = document.createElement('div');
        item.className = 'activity-bar-item';

        const name = formatActivityName(activity.activity);
        const percentage = activity.percentage.toFixed(1);

        item.innerHTML = `
            <div class="activity-bar-header">
                <span class="activity-name">${name}</span>
                <span class="activity-percentage">${percentage}%</span>
            </div>
            <div class="activity-bar">
                <div class="activity-bar-fill" style="width: ${percentage}%"></div>
            </div>
        `;
        container.appendChild(item);
    });
}

// ============================================================================
// Analysis Summary Generation
// ============================================================================

function generateAnalysisSummary(report) {
    const score = (report.productivity_score * 100).toFixed(1);
    const duration = formatTime(report.session_duration);
    const rating = getProductivityRating(report.productivity_score);

    const topActivity = Object.values(report.activity_breakdown || {})
        .sort((a, b) => b.total_time - a.total_time)[0];

    const topTool = Object.values(report.tool_usage || {})
        .sort((a, b) => b.total_time - a.total_time)[0];

    let summary = `📊 Analysis Complete\n\n`;
    summary += `Productivity Score: ${score}% (${rating})\n`;
    summary += `Session Duration: ${duration}\n`;
    summary += `Idle Time: ${formatTime(report.idle_time)} (${report.idle_percentage.toFixed(1)}%)\n\n`;

    if (topActivity) {
        summary += `Primary Activity: ${formatActivityName(topActivity.activity)} (${topActivity.percentage.toFixed(1)}%)\n`;
    }

    if (topTool) {
        summary += `Most Used Tool: ${topTool.tool_name} (${formatTime(topTool.total_time)})\n`;
    }

    summary += `\nYou can now ask questions about this analysis session.`;

    return summary;
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

function formatActivityName(name) {
    return name
        .split('_')
        .map(word => word.charAt(0) + word.slice(1).toLowerCase())
        .join(' ');
}

function getProductivityRating(score) {
    if (score >= 0.8) return 'Excellent';
    if (score >= 0.7) return 'Good';
    if (score >= 0.6) return 'Fair';
    if (score >= 0.5) return 'Below Average';
    return 'Needs Improvement';
}
